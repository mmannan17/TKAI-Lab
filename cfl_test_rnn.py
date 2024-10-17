# train_rnn.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import (
    prepare_data, shuffle_data, SequenceDataset, collate_fn,
    char_to_int, vocab_size, PAD_IDX
)
import time
import random


torch.manual_seed(42)
random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# n_values for train, validation, and test sets
n_train = list(range(1, 501))
n_val = list(range(501, 1001))
n_test = list(range(1001, 1501))

random.shuffle(n_train)
random.shuffle(n_val)
random.shuffle(n_test)

# data with unique negative examples
used_negatives = set()

train_data, train_labels, used_negatives = prepare_data(n_train, num_samples=500, used_negatives=used_negatives)
val_data, val_labels, used_negatives = prepare_data(n_val, num_samples=500, used_negatives=used_negatives)
test_data, test_labels, used_negatives = prepare_data(n_test, num_samples=500, used_negatives=used_negatives)


train_data, train_labels = shuffle_data(train_data, train_labels)
val_data, val_labels = shuffle_data(val_data, val_labels)
test_data, test_labels = shuffle_data(test_data, test_labels)

# datasets
train_dataset = SequenceDataset(train_data, train_labels)
val_dataset = SequenceDataset(val_data, val_labels)
test_dataset = SequenceDataset(test_data, test_labels)

# data loaders
batch_size = 4  # Adjust as needed

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

# RNN Model
class RNNModel(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size, vocab_size, PAD_IDX):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        packed_output, hidden = self.rnn(packed_input)
       
        out = self.fc(hidden[-1])
        return torch.sigmoid(out)

# Hyperparameters
embedding_dim = 8
hidden_size = 64  
output_size = 1
num_epochs = 50  
learning_rate = 0.001

# Initialize model
rnn_model = RNNModel(embedding_dim, hidden_size, output_size, vocab_size, PAD_IDX).to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)

# Training and evaluation functions
def train(model, optimizer, train_loader, val_loader, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for sequences, lengths, labels in train_loader:
            sequences, lengths, labels = sequences.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences, lengths).view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            epoch_train_loss += loss.item()

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for sequences, lengths, labels in val_loader:
                sequences, lengths, labels = sequences.to(device), lengths.to(device), labels.to(device)
                outputs = model(sequences, lengths).view(-1)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_train_loss / len(train_loader):.4f}, Val Loss: {epoch_val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.4f}')

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, lengths, labels in test_loader:
            sequences, lengths, labels = sequences.to(device), lengths.to(device), labels.to(device)
            outputs = model(sequences, lengths).view(-1)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    # Train the RNN model
    start_time = time.time()
    print("\nTraining RNN Model")
    train(rnn_model, optimizer, train_loader, val_loader, num_epochs)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

    # Evaluate on test set
    print("\nEvaluating RNN Model")
    evaluate(rnn_model, test_loader)
