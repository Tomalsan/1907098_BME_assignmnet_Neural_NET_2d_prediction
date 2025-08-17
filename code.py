import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWYX')
SS3_CLASSES = list('CEH')

# Create one-hot encoders
aa_encoder = OneHotEncoder(sparse_output=False, categories=[AMINO_ACIDS])
ss3_encoder = OneHotEncoder(sparse_output=False, categories=[SS3_CLASSES])

# Fit encoders
aa_encoder.fit(np.array(AMINO_ACIDS).reshape(-1, 1))
ss3_encoder.fit(np.array(SS3_CLASSES).reshape(-1, 1))

class ProteinDataset(Dataset):
    def __init__(self, train_file):
        self.train_csv = pd.read_csv(train_file)
        seq_list = []
        label_list = []

        for _, row in tqdm(self.train_csv.iterrows(), total=len(self.train_csv)):
            seq = row['seq']
            sst3 = row['sst3']

            if len(seq) == len(sst3):
                # One-hot encode immediately
                seq_encoded = aa_encoder.transform(np.array(list(seq)).reshape(-1, 1))
                label_encoded = ss3_encoder.transform(np.array(list(sst3)).reshape(-1, 1))

                seq_list.append(seq_encoded)
                label_list.append(label_encoded)

        # Concatenate all sequences into one tensor in memory
        self.sequences = torch.FloatTensor(np.concatenate(seq_list, axis=0))
        self.labels = torch.FloatTensor(np.concatenate(label_list, axis=0))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class ProteinModel(nn.Module):
  def __init__(self, input_dim=21, hidden_dim=216, output_dim=3, num_layers=10):
      super(ProteinModel, self).__init__()

      self.input_layer = nn.Sequential(
          nn.Linear(input_dim, hidden_dim),
          nn.ReLU(),
          nn.Dropout(0.3)
      )

      self.hidden_layers = nn.ModuleList()
      for _ in range(num_layers - 2):
          layer = nn.Sequential(
              nn.Linear(hidden_dim, hidden_dim),
              nn.Linear(hidden_dim,hidden_dim),
              nn.Linear(hidden_dim,hidden_dim),
              nn.BatchNorm1d(hidden_dim),
              nn.LayerNorm(hidden_dim),
              nn.LeakyReLU(0.2),
              nn.Dropout(0.5)
          )
          self.hidden_layers.append(layer)

      self.output_layer = nn.Linear(hidden_dim, output_dim)


  def forward(self, x):
    if x.dim() == 2:  # (batch_size, input_dim)
        out = self.input_layer(x)
        for layer in self.hidden_layers:
            residual = out
            out = layer(out)
            out += residual
        out = self.output_layer(out)
        return out  # (batch_size, output_dim)

    elif x.dim() == 3:  # (batch_size, seq_len, input_dim)
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = x.view(-1, x.shape[2])
        out = self.input_layer(x)
        for layer in self.hidden_layers:
            residual = out
            out = layer(out)
            out += residual
        out = self.output_layer(out)
        out = out.view(batch_size, seq_len, -1)
        return out

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

def train_model(train_loader, val_loader, epochs=20, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training started for device = {device}")

    model = ProteinModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float('inf')
    print(f"Model params are: {next(model.parameters()).device}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Training: {epoch+1}/{epochs}")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            output = output.view(-1, 3)
            target_flat = torch.argmax(target.view(-1, 3), dim=1)

            loss = criterion(output, target_flat)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate training accuracy
            pred = torch.argmax(output, dim=1)
            train_correct += (pred == target_flat).sum().item()
            train_total += target_flat.size(0)

            pbar.set_postfix({'Batch Loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                output = output.view(-1, 3)
                target_flat = torch.argmax(target.view(-1, 3), dim=1)

                loss = criterion(output, target_flat)
                val_loss += loss.item()

                pred = torch.argmax(output, dim=1)
                correct += (pred == target_flat).sum().item()
                total += target_flat.size(0)

        val_loss /= len(val_loader)
        accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)

        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_accuracy:.4f}, Val Acc: {accuracy:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    # Plotting the curves
    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    return model, train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_test_set(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            output = output.view(-1, 3)
            target_flat = torch.argmax(target.view(-1, 3), dim=1)

            pred = torch.argmax(output, dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target_flat.cpu().numpy())

    # Calculate metrics
    print(classification_report(all_targets, all_preds))

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1', 'Class 2'],
                yticklabels=['Class 0', 'Class 1', 'Class 2'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.show()

    return all_preds, all_targets

"""## Training"""

# Load data
train_file = "data/training_secondary_structure_train.csv"
valid_file = "data/validation_secondary_structure_valid.csv"
test_file = "data/test_secondary_structure_casp12.csv"

# Create datasets and dataloaders
train_dataset = ProteinDataset(train_file)
val_dataset = ProteinDataset(valid_file)
test_dataset = ProteinDataset(test_file)

batch_size = 1000
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

model, train_losses, val_losses, train_accs, val_accs = train_model(train_loader, val_loader, epochs=5)

test_preds, test_targets = evaluate_test_set(model, test_loader)