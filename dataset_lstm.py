import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from collections import Counter
import re
import logging

logger = logging.getLogger(__name__)

class LSTMTokenizer:
    """
    Simple tokenizer for LSTM models
    """
    def __init__(self, max_vocab_size=30000, max_seq_length=512):
        self.word2idx = {}
        self.idx2word = {}
        self.word2idx['<pad>'] = 0
        self.word2idx['<unk>'] = 1
        self.idx2word[0] = '<pad>'
        self.idx2word[1] = '<unk>'
        self.vocab_size = 2  # Start with pad and unk tokens
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
    
    def fit(self, texts):
        """Build vocabulary from texts"""
        word_counts = Counter()
        
        # Clean and tokenize texts
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)
        
        # Sort by frequency and take most common words
        vocab_words = [word for word, count in word_counts.most_common(self.max_vocab_size - 2)]
        
        # Add words to vocabulary
        for word in vocab_words:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
                
        logger.info(f"Vocabulary size: {self.vocab_size}")
        return self
    
    def _tokenize(self, text):
        """Simple tokenization by splitting on whitespace and removing punctuation"""
        text = text.lower()
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    def encode(self, text, padding=True, truncation=True):
        """Convert text to token ids"""
        words = self._tokenize(text)
        
        # Truncate if needed
        if truncation and len(words) > self.max_seq_length:
            words = words[:self.max_seq_length]
        
        # Convert to indices
        ids = [self.word2idx.get(word, self.word2idx['<unk>']) for word in words]
        
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = [1] * len(ids)
        
        # Pad if needed
        if padding and len(ids) < self.max_seq_length:
            padding_length = self.max_seq_length - len(ids)
            ids = ids + [self.word2idx['<pad>']] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def from_json(self, json_file):
        """Load tokenizer from JSON file container word2idx dict"""
        import json
        with open(json_file, 'r') as f:
            data = json.load(f)
            self.word2idx = data
            self.idx2word = {v: k for k, v in self.word2idx.items()}
            # Add pad and unk tokens
            self.word2idx['<pad>'] = 0
            self.word2idx['<unk>'] = 1
            self.idx2word[0] = '<pad>'
            self.idx2word[1] = '<unk>'
            # Update vocab size
            self.vocab_size = len(self.word2idx)
            logger.info(f"Loaded tokenizer with {self.vocab_size} tokens")

    def save(self, json_file):
        """Save tokenizer word2idx to JSON file"""
        import json
        with open(json_file, 'w') as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=4)
            logger.info(f"Tokenizer saved to {json_file}")

class LSTMDataset(Dataset):
    """Dataset for LSTM model"""
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer.encode(text)
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def get_text_(self, idx):
        """Get original text for a given index"""
        return {
            'text': self.texts[idx],
            'label': self.labels[idx]
        }

def prepare_lstm_data(data_path, text_col='text', label_col='label', 
                     max_vocab_size=30000, max_seq_length=512,
                     val_split=0.1, test_split=0.1, batch_size=32, seed=42, tokenizer=None, return_datasets=False, return_tokenizer=False):
    """
    Load data and prepare for LSTM model
    tokenizer: Custom LSTMTokenizer to use instead of creating a new one
    """
    # Load data
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.tsv'):
        df = pd.read_csv(data_path, sep='\t')
    else:
        raise ValueError("Unsupported file format. Please provide CSV or TSV file.")
    
    # Convert labels to numeric if they aren't already
    if not np.issubdtype(df[label_col].dtype, np.number):
        label_map = {label: idx for idx, label in enumerate(sorted(df[label_col].unique()))}
        df['label_numeric'] = df[label_col].map(label_map)
        labels = df['label_numeric'].values
        logger.info(f"Label mapping: {label_map}")
    else:
        labels = df[label_col].values
        # Make sure labels start from 0
        min_label = labels.min()
        if min_label != 0:
            label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
            labels = np.array([label_map[label] for label in labels])
    
    texts = df[text_col].values
    
    # Split data
    np.random.seed(seed)
    indices = np.random.permutation(len(texts))
    
    test_size = int(test_split * len(texts))
    val_size = int(val_split * len(texts))
    train_size = len(texts) - test_size - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_texts, train_labels = texts[train_indices], labels[train_indices]
    val_texts, val_labels = texts[val_indices], labels[val_indices]
    test_texts, test_labels = texts[test_indices], labels[test_indices]
    
    # Create tokenizer and fit on training data
    if tokenizer is None:
        tokenizer = LSTMTokenizer(max_vocab_size=max_vocab_size, max_seq_length=max_seq_length)
        tokenizer.fit(train_texts)
    
    # Create datasets
    train_dataset = LSTMDataset(train_texts, train_labels, tokenizer)
    val_dataset = LSTMDataset(val_texts, val_labels, tokenizer)
    test_dataset = LSTMDataset(test_texts, test_labels, tokenizer)

    if return_datasets and not return_tokenizer:
        return train_dataset, val_dataset, test_dataset, tokenizer.vocab_size
    elif return_tokenizer and not return_datasets:
        return train_dataset, val_dataset, test_dataset, tokenizer
    
    # Create data loaders
    if len(train_dataset.texts) == 0:
        logger.warning("Training dataset is empty. Please check your data.")
        train_loader = None
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if len(val_dataset.texts) == 0:
        logger.warning("Validation dataset is empty. Please check your data.")
        val_loader = None
    else:
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    if len(test_dataset.texts) == 0:
        logger.warning("Test dataset is empty. Please check your data.")
        test_loader = None
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    if not return_tokenizer:
        return train_loader, val_loader, test_loader, tokenizer.vocab_size
    else:
        return train_loader, val_loader, test_loader, tokenizer