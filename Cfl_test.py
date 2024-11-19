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
import torch.nn.functional as F

torch.manual_seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# n_values for train, validation, and test sets
n_train = list(range(1, 501))
n_val = list(range(501, 1001))
n_test = list(range(1001, 1501))

random.shuffle(n_train)
random.shuffle(n_val)
random.shuffle(n_test)

# Data with unique negative examples
used_negatives = set()

train_data, train_labels, used_negatives = prepare_data(n_train, num_samples=500, used_negatives=used_negatives)
val_data, val_labels, used_negatives = prepare_data(n_val, num_samples=500, used_negatives=used_negatives)
test_data, test_labels, used_negatives = prepare_data(n_test, num_samples=500, used_negatives=used_negatives)

train_data, train_labels = shuffle_data(train_data, train_labels)
val_data, val_labels = shuffle_data(val_data, val_labels)
test_data, test_labels = shuffle_data(test_data, test_labels)

# Datasets
train_dataset = SequenceDataset(train_data, train_labels)
val_dataset = SequenceDataset(val_data, val_labels)
test_dataset = SequenceDataset(test_data, test_labels)

# Data loaders
batch_size = 4  

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input gate weights
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # Forget gate weights
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # Cell gate weights
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size))

        # Output gate weights
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        std = 1.0 / (self.hidden_size ** 0.5)
        for param in self.parameters():
            nn.init.uniform_(param, -std, std)

    def forward(self, x, hx):
        h_prev, c_prev = hx

        i_t = torch.sigmoid(F.linear(x, self.W_ii, self.b_i) + F.linear(h_prev, self.W_hi))
        f_t = torch.sigmoid(F.linear(x, self.W_if, self.b_f) + F.linear(h_prev, self.W_hf))
        g_t = torch.tanh(F.linear(x, self.W_ig, self.b_g) + F.linear(h_prev, self.W_hg))
        o_t = torch.sigmoid(F.linear(x, self.W_io, self.b_o) + F.linear(h_prev, self.W_ho))

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

# Custom LSTM Model
class CustomLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size, vocab_size, PAD_IDX):
        super(CustomLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.hidden_size = hidden_size
        self.lstm_cell = CustomLSTMCell(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        batch_size = embedded.size(0)
        h_t = torch.zeros(batch_size, self.hidden_size).to(embedded.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(embedded.device)
        
        max_length = embedded.size(1)
        outputs = torch.zeros(batch_size, self.hidden_size).to(embedded.device)
        
        for t in range(max_length):
            h_t, c_t = self.lstm_cell(embedded[:, t, :], (h_t, c_t))
            # Collect hidden states at the end of sequences
            mask = (lengths == t + 1).unsqueeze(1).float().to(embedded.device)
            outputs = outputs * (1 - mask) + h_t * mask
        
        out = self.fc(outputs)
        return out

# Hyperparameters
embedding_dim = 8
hidden_size = 64
output_size = 1
num_epochs = 50
learning_rate = 0.001

# Initialize model
lstm_model = CustomLSTM(embedding_dim, hidden_size, output_size, vocab_size, PAD_IDX).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)

# Training and evaluation functions
def train(model, optimizer, train_loader, val_loader, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for sequences, lengths, labels in train_loader:
            sequences, lengths, labels = sequences.to(device), lengths.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(sequences, lengths).view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            epoch_train_loss += loss.item()

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for sequences, lengths, labels in val_loader:
                sequences, lengths, labels = sequences.to(device), lengths.to(device), labels.to(device).float()
                outputs = model(sequences, lengths).view(-1)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_train_loss / len(train_loader):.4f}, '
              f'Val Loss: {epoch_val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.4f}')

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, lengths, labels in test_loader:
            sequences, lengths, labels = sequences.to(device), lengths.to(device), labels.to(device).float()
            outputs = model(sequences, lengths).view(-1)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    # Train the LSTM model
    start_time = time.time()
    print("\nTraining LSTM Model")
    train(lstm_model, optimizer, train_loader, val_loader, num_epochs)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

    # Evaluate on test set
    print("\nEvaluating LSTM Model")
    evaluate(lstm_model, test_loader)
