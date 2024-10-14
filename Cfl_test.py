import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)

# Function to generate positive examples of a^n b^n grammar
def generate_positive_examples(n_values):
    return ['a' * n + 'b' * n for n in n_values]

# Function to generate negative examples
def generate_negative_examples(n_values):
    neg_examples = []
    for n in n_values:
        if n > 1:
            m = n - random.randint(1, n - 1)
            if m != n:
                neg_examples.append('a' * n + 'b' * m)  # n > m
                neg_examples.append('a' * m + 'b' * n)  # n < m
        else:
            neg_examples.append('a' * n + 'b' * (n + 1))
            neg_examples.append('a' * (n + 1) + 'b' * n)
    return neg_examples

# Generate n values for each set
n_train = list(range(1, 71))
n_val = list(range(71, 141))
n_test = list(range(141, 211))


# Random sampling without replacement
random.shuffle(n_train)
random.shuffle(n_val)
random.shuffle(n_test)

# Data preparation
train_positive = generate_positive_examples(n_train)
train_negative = generate_negative_examples(n_train)[:len(train_positive)]  # Balance negative examples
train_data = train_positive + train_negative
train_labels = [1] * len(train_positive) + [0] * len(train_negative)

val_positive = generate_positive_examples(n_val)
val_negative = generate_negative_examples(n_val)[:len(val_positive)]  # Balance negative examples
val_data = val_positive + val_negative
val_labels = [1] * len(val_positive) + [0] * len(val_negative)

test_positive = generate_positive_examples(n_test)
test_negative = generate_negative_examples(n_test)[:len(test_positive)]  # Balance negative examples
test_data = test_positive + test_negative
test_labels = [1] * len(test_positive) + [0] * len(test_negative)

# Shuffle the datasets
def shuffle_data(data, labels):
    combined = list(zip(data, labels))
    random.shuffle(combined)
    data[:], labels[:] = zip(*combined)

shuffle_data(train_data, train_labels)
shuffle_data(val_data, val_labels)
shuffle_data(test_data, test_labels)

# Tokenization
char_to_int = {'a': 0, 'b': 1}
vocab_size = len(char_to_int) + 1  # +1 for padding

# Constants for padding
PAD_IDX = 2

# Dataset class
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        # Encode the sequence
        encoded_seq = [char_to_int[char] for char in seq]
        length = len(encoded_seq)
        return torch.tensor(encoded_seq, dtype=torch.long), length, torch.tensor(label, dtype=torch.float)

# Collate function with sorting by length
def collate_fn(batch):
    sequences, lengths, labels = zip(*batch)
    lengths = torch.tensor(lengths)
    sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)
    labels = torch.tensor(labels)
    # Sort by lengths in descending order
    lengths, perm_idx = lengths.sort(0, descending=True)
    sequences_padded = sequences_padded[perm_idx]
    labels = labels[perm_idx]
    return sequences_padded, lengths, labels


batch_size = 16  

train_dataset = SequenceDataset(train_data, train_labels)
val_dataset = SequenceDataset(val_data, val_labels)
test_dataset = SequenceDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size, vocab_size, PAD_IDX):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, lengths):
        x = self.embedding(x)
        # Pack the padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        packed_output, (hidden, _) = self.lstm(packed_input)
        # Unpack the sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # Get the outputs at the last valid time steps
        idx = (lengths - 1).unsqueeze(1).unsqueeze(1).expand(-1, 1, output.size(2))
        output_at_last_timestep = output.gather(1, idx.to(device)).squeeze(1)
        out = self.fc(output_at_last_timestep)
        return torch.sigmoid(out)

# RNN model
class RNNModel(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size, vocab_size, PAD_IDX):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, lengths):
        x = self.embedding(x)
        # Pack the padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        packed_output, hidden = self.rnn(packed_input)
        # Unpack the sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        idx = (lengths - 1).unsqueeze(1).unsqueeze(1).expand(-1, 1, output.size(2))
        output_at_last_timestep = output.gather(1, idx.to(device)).squeeze(1)
        out = self.fc(output_at_last_timestep)
        return torch.sigmoid(out)

# Hyperparameters
embedding_dim = 8
hidden_size = 256  # Increased hidden size
output_size = 1
num_epochs = 200  
learning_rate = 0.001

# Initialize models
lstm_model = LSTMModel(embedding_dim, hidden_size, output_size, vocab_size, PAD_IDX)
rnn_model = RNNModel(embedding_dim, hidden_size, output_size, vocab_size, PAD_IDX)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm_model.to(device)
rnn_model.to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=learning_rate)
optimizer_rnn = optim.Adam(rnn_model.parameters(), lr=learning_rate)

# Training and evaluation functions
def train(model, optimizer, train_loader, val_loader, num_epochs):
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for sequences, lengths, labels in train_loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            outputs = outputs.view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            epoch_train_loss += loss.item()
        
        train_losses.append(epoch_train_loss / len(train_loader))
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for sequences, lengths, labels in val_loader:
                sequences = sequences.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                outputs = model(sequences, lengths)
                outputs = outputs.view(-1)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_losses.append(epoch_val_loss / len(val_loader))
        val_acc = correct / total
        val_accuracies.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.4f}')
    
    return train_losses, val_losses, val_accuracies

def plot_curves(train_losses, val_losses, val_accuracies, model_name):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, lengths, labels in test_loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            outputs = model(sequences, lengths)
            outputs = outputs.view(-1)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')

# Train the LSTM model
print("\nTraining LSTM Model")
lstm_train_losses, lstm_val_losses, lstm_val_accuracies = train(
    lstm_model, optimizer_lstm, train_loader, val_loader, num_epochs)

# Plot results for LSTM
plot_curves(lstm_train_losses, lstm_val_losses, lstm_val_accuracies, "LSTM")

# Evaluate on test set
print("\nEvaluating LSTM Model")
evaluate(lstm_model, test_loader)

# Train the RNN model
print("\nTraining RNN Model")
rnn_train_losses, rnn_val_losses, rnn_val_accuracies = train(
    rnn_model, optimizer_rnn, train_loader, val_loader, num_epochs)

# Plot results for RNN
plot_curves(rnn_train_losses, rnn_val_losses, rnn_val_accuracies, "RNN")

# Evaluate on test set
print("\nEvaluating RNN Model")
evaluate(rnn_model, test_loader)
