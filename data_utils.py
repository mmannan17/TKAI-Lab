# data_utils.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)

# Constants and mappings
char_to_int = {'a': 0, 'b': 1}
int_to_char = {v: k for k, v in char_to_int.items()}
vocab_size = len(char_to_int) + 1  # +1 for padding
PAD_IDX = len(char_to_int)

# Function to generate positive examples of a^n b^n grammar
def generate_positive_examples(n_values):
    return ['a' * n + 'b' * n for n in n_values]

# Function to generate negative examples without duplicates across splits
def generate_negative_examples(n_values, num_negatives, used_negatives):
    neg_examples = set()
    min_n = min(n_values)
    max_n = max(n_values)
    while len(neg_examples) < num_negatives:
        n = random.choice(n_values)
        m = random.randint(min_n, max_n)
        if n != m:
            # Randomly decide whether to have extra 'a's or 'b's
            if random.random() < 0.5:
                neg_example = 'a' * n + 'b' * m
            else:
                neg_example = 'a' * m + 'b' * n
            if neg_example not in used_negatives:
                neg_examples.add(neg_example)
                used_negatives.add(neg_example)
    return list(neg_examples), used_negatives

def prepare_data(n_values, num_samples=500, used_negatives=None):
    if used_negatives is None:
        used_negatives = set()
    positive_n_values = n_values[:num_samples]
    positive = generate_positive_examples(positive_n_values)
    negative, used_negatives = generate_negative_examples(n_values, num_samples, used_negatives)
    data = positive + negative
    labels = [1] * len(positive) + [0] * len(negative)
    return data, labels, used_negatives

def shuffle_data(data, labels):
    combined = list(zip(data, labels))
    random.shuffle(combined)
    data[:], labels[:] = zip(*combined)
    return data, labels

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
        encoded_seq = [char_to_int[char] for char in seq]
        length = len(encoded_seq)
        return torch.tensor(encoded_seq, dtype=torch.long), length, torch.tensor(label, dtype=torch.float)

# Collate function with sorting by length
def collate_fn(batch):
    sequences, lengths, labels = zip(*batch)
    lengths = torch.tensor(lengths)
    sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)
    labels = torch.tensor(labels)
    lengths, perm_idx = lengths.sort(0, descending=True)
    sequences_padded = sequences_padded[perm_idx]
    labels = labels[perm_idx]
    return sequences_padded, lengths, labels
