import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
import warnings
import librosa
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import seaborn as sns
from torchviz import make_dot

warnings.filterwarnings('ignore')

# Basic settings
dataset_path = 'D:/UCalgary/ENEL645/Final_Project/UrbanSound8K'
sample_rate = 31250
n_mel = 20
batch_size = 128
num_epochs = 10
learning_rate = 0.001
num_classes = 10
num_workers = 16

# Settings for spectrogram
n_fft = 1024
hop_length = 512
win_length = 1024

# Cache directory for precomputed data
cache_dir = os.path.join(dataset_path, 'mel_cache')
os.makedirs(cache_dir, exist_ok=True)

# Load metadata and folds
metadata = pd.read_csv(os.path.join(dataset_path, 'metadata', 'UrbanSound8K.csv'))
unique_folds = np.unique(metadata['fold'])

def get_max_time_steps_mel(metadata, dataset_path, sr=31250, n_fft=1024, hop_length=512, n_mel=20):
    # Find the largest time dimension across all audio files
    print("Calculating max time steps...")
    max_time_steps = 0
    mel_transform = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mel)
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc='Max Time Steps'):
        file_path = os.path.join(dataset_path, 'audio', f'fold{row["fold"]}', row["slice_file_name"])
        audio, _ = librosa.load(file_path, sr=sr)
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        mel_spec = mel_transform(audio_tensor)
        time_steps = mel_spec.shape[1]
        if time_steps > max_time_steps:
            max_time_steps = time_steps
    print(f"Max time steps: {max_time_steps}")
    return max_time_steps

class UrbanSoundDataset(Dataset):
    # Dataset that loads precomputed mel-spectrograms or computes them if missing
    def __init__(self, metadata, dataset_path, fold_list, sr=31250, n_fft=1024, hop_length=512,
                 n_mel=20, max_time_steps=6960, cache_dir='', augment=False):
        self.metadata = metadata[metadata['fold'].isin(fold_list)].copy()
        self.dataset_path = dataset_path
        self.sr = sr
        self.n_mel = n_mel
        self.max_time_steps = max_time_steps
        self.cache_dir = cache_dir
        self.augment = augment

        # List of audio file paths and their labels
        self.file_paths = self.metadata.apply(
            lambda row: os.path.join(self.dataset_path, 'audio', f'fold{row["fold"]}', row["slice_file_name"]), axis=1
        ).tolist()
        self.labels = self.metadata['classID'].tolist()

        # Transform audio to mel-spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mel)

        # Simple augmentation
        self.augmentation_transforms = nn.Sequential(
            torchaudio.transforms.TimeMasking(time_mask_param=10),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=5)
        ) if self.augment else None

        # Load or compute mel-spectrograms
        self.mel_specs = []
        print("Loading or creating mel-spectrograms...")
        for fp in tqdm(self.file_paths, desc='Load/Precompute MelSpectrogram'):
            cache_filename = os.path.join(self.cache_dir, os.path.basename(fp) + '.pt')
            if os.path.exists(cache_filename):
                # Use cached data
                mel = torch.load(cache_filename)
            else:
                # Compute mel-spectrogram from audio
                audio, _ = librosa.load(fp, sr=self.sr)
                audio_tensor = torch.tensor(audio, dtype=torch.float32)
                mel = self.mel_transform(audio_tensor)
                # Apply log and normalize
                mel = torch.log(mel + 1e-9)
                mel = (mel - mel.mean()) / (mel.std() + 1e-9)
                # Pad or trim to max_time_steps
                time_steps = mel.shape[1]
                if time_steps < self.max_time_steps:
                    pad_width = self.max_time_steps - time_steps
                    mel = torch.nn.functional.pad(mel, (0, pad_width))
                else:
                    mel = mel[:, :self.max_time_steps]
                torch.save(mel, cache_filename)
            self.mel_specs.append(mel)
        print("Done loading mel-spectrograms.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mel = self.mel_specs[idx].clone()
        label = self.labels[idx]
        # Apply augmentation if chosen
        if self.augment and self.augmentation_transforms is not None:
            mel = self.augmentation_transforms(mel)
        # Add channel dimension
        mel = mel.unsqueeze(0)
        return mel, label

class DepthwiseSeparableConv(nn.Module):
    # A lightweight convolution block
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class LightweightESCNN(nn.Module):
    # A simple CNN model for audio classification
    def __init__(self, num_classes=10, in_channels=1, n_mel=20, time_steps=0):
        super(LightweightESCNN, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, 16, kernel_size=3, stride=2)
        self.conv2 = DepthwiseSeparableConv(16, 32, kernel_size=3, stride=2)
        self.conv3 = DepthwiseSeparableConv(32, 64, kernel_size=3, stride=2)
        # Find the output size after these layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, n_mel, time_steps)
            out = self.conv1(dummy_input)
            out = self.conv2(out)
            out = self.conv3(out)
            flatten_size = out.view(1, -1).size(1)
        self.fc = nn.Linear(flatten_size, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_model(model, device, train_loader, criterion, optimizer, epoch):
    # Train one epoch
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1} Training', leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def evaluate_model(model, device, val_loader, criterion, epoch):
    # Evaluate one epoch
    model.eval()
    running_loss = 0.0
    preds = []
    truths = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1} Validation', leave=False):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            truths.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = accuracy_score(truths, preds)
    return epoch_loss, accuracy, truths, preds

def plot_loss_accuracy(train_losses, val_losses, val_accuracies, fold_number):
    # Plot loss and accuracy for one fold
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 4))
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title(f'Fold {fold_number} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g-', label='Val Accuracy')
    plt.title(f'Fold {fold_number} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'fold_{fold_number}_metrics.png')
    plt.close()

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    # Plot and save the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_classification_report(report, classes):
    # Plot precision, recall, and f1-score for each class
    precision = [report[str(i)]['precision'] for i in range(len(classes))]
    recall = [report[str(i)]['recall'] for i in range(len(classes))]
    f1 = [report[str(i)]['f1-score'] for i in range(len(classes))]

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-Score')

    plt.xticks(x, classes, rotation=45)
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Classification Report')
    plt.legend()
    plt.tight_layout()
    plt.savefig('classification_report.png')
    plt.close()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Enable cudnn benchmark for possible speedup
    if device.type == 'cuda':
        cudnn.benchmark = True

    # Compute max_time_steps for all audio
    max_time_steps = get_max_time_steps_mel(metadata, dataset_path, sr=sample_rate,
                                            n_fft=n_fft, hop_length=hop_length, n_mel=n_mel)

    print("Starting 10-fold cross-validation...")
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}, Num workers: {num_workers}")

    all_preds = []
    all_truths = []
    fold_accuracies = []
    all_conf_matrices = np.zeros((num_classes, num_classes))

    best_model_path = "best_model.pth"
    use_augmentation = False
    early_stop_patience = 5
    class_names = [str(i) for i in range(num_classes)]

    # Loop through each fold
    for current_fold in unique_folds:
        print(f"\n========================================")
        print(f"Starting Fold {current_fold} as validation...")

        train_folds = [f for f in unique_folds if f != current_fold]
        val_folds = [current_fold]

        # Load datasets for training and validation
        train_dataset = UrbanSoundDataset(metadata, dataset_path, train_folds, sr=sample_rate,
                                          n_fft=n_fft, hop_length=hop_length, n_mel=n_mel,
                                          max_time_steps=max_time_steps, cache_dir=cache_dir,
                                          augment=use_augmentation)
        val_dataset = UrbanSoundDataset(metadata, dataset_path, val_folds, sr=sample_rate,
                                        n_fft=n_fft, hop_length=hop_length, n_mel=n_mel,
                                        max_time_steps=max_time_steps, cache_dir=cache_dir,
                                        augment=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, persistent_workers=True)

        model = LightweightESCNN(num_classes=num_classes, in_channels=1, n_mel=n_mel, time_steps=max_time_steps).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        fold_best_acc = 0.0
        fold_best_val_loss = float('inf')
        no_improvement_count = 0

        train_losses_per_fold = []
        val_losses_per_fold = []
        val_accuracies_per_fold = []

        # Training loop for each epoch
        for epoch in range(num_epochs):
            print(f"--- Fold {current_fold}, Epoch {epoch+1}/{num_epochs} ---")
            train_loss = train_model(model, device, train_loader, criterion, optimizer, epoch)
            val_loss, val_acc, truths, preds = evaluate_model(model, device, val_loader, criterion, epoch)
            print(f"Fold {current_fold}, Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

            train_losses_per_fold.append(train_loss)
            val_losses_per_fold.append(val_loss)
            val_accuracies_per_fold.append(val_acc)

            # Check for improvement in accuracy or tie in accuracy with better loss
            if (val_acc > fold_best_acc) or ((val_acc == fold_best_acc) and (val_loss < fold_best_val_loss)):
                fold_best_acc = val_acc
                fold_best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved! Acc = {fold_best_acc:.4f}, Loss = {fold_best_val_loss:.4f}")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                print(f"No improvement this epoch. Count = {no_improvement_count}")

            # Early stopping if no improvement
            if no_improvement_count >= early_stop_patience:
                print(f"No improvement for {early_stop_patience} epochs. Stopping early.")
                break

            scheduler.step()

        # Plot metrics for this fold
        plot_loss_accuracy(train_losses_per_fold, val_losses_per_fold, val_accuracies_per_fold, current_fold)

        fold_accuracies.append(fold_best_acc)
        cm = confusion_matrix(truths, preds, labels=range(num_classes))
        all_conf_matrices += cm
        all_truths.extend(truths)
        all_preds.extend(preds)

        print(f"Completed Fold {current_fold}. Best Fold accuracy: {fold_best_acc:.4f}")

    # After all folds
    overall_accuracy = np.mean(fold_accuracies)
    print("\n========================================")
    print(f"Overall 10-Fold Accuracy: {overall_accuracy:.4f}")
    classification_rep = classification_report(all_truths, all_preds, digits=4, output_dict=True)
    print("Classification Report (all folds):")
    print(classification_report(all_truths, all_preds, digits=4))
    print("Confusion Matrix (all folds):")
    print(all_conf_matrices)
    print("========================================\nDone with all folds.")

    # Plot final confusion matrix and classification report
    plot_confusion_matrix(all_conf_matrices, class_names, title='Aggregated Confusion Matrix')
    plot_classification_report(classification_rep, class_names)

    # Load best model for ONNX export and model graph
    model = LightweightESCNN(num_classes=num_classes, in_channels=1, n_mel=n_mel, time_steps=max_time_steps)
    model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
    model.eval()

    dummy_input = torch.randn(1, 1, n_mel, max_time_steps)
    y = model(dummy_input)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render("model_architecture", format="png")
    print("Saved model_architecture.png")

    # Export model to ONNX
    onnx_path = "lightweight_esc_model.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, export_params=True,
                      opset_version=11, do_constant_folding=True,
                      input_names=['input'], output_names=['output'])
    print(f"ONNX model saved to {onnx_path}. Use STM32Cube.AI for final deployment.")
    print("Done.")
