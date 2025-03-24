import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
import numpy as np

class DocumentDataset(Dataset):
    """
    Dataset class for document classification
    with improved preprocessing and batching
    """
    def __init__(self, texts, labels, tokenizer_name='bert-base-uncased', max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize the text with attention mask and truncation
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_path, text_col='text', label_col='label', validation_split=0.1, test_split=0.1, seed=42):
    """
    Load data from CSV/TSV and split into train, validation and test sets
    """
    # Determine file format based on extension
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.tsv'):
        df = pd.read_csv(data_path, sep='\t')
    else:
        raise ValueError("Unsupported file format. Please provide CSV or TSV file.")
    
    # Convert labels to numeric if they aren't already
    if not np.issubdtype(df[label_col].dtype, np.number):
        label_map = {label: idx for idx, label in enumerate(df[label_col].unique())}
        df['label_numeric'] = df[label_col].map(label_map)
        labels = df['label_numeric'].values
    else:
        labels = df[label_col].values
    
    # Create a DataFrame with text and numeric labels
    texts = df[text_col].values
    
    # Shuffle and split the data
    np.random.seed(seed)
    indices = np.random.permutation(len(texts))
    
    test_size = int(test_split * len(texts))
    val_size = int(validation_split * len(texts))
    train_size = len(texts) - test_size - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_texts, train_labels = texts[train_indices], labels[train_indices]
    val_texts, val_labels = texts[val_indices], labels[val_indices]
    test_texts, test_labels = texts[test_indices], labels[test_indices]
    
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)

def create_data_loaders(train_data, val_data, test_data, tokenizer_name='bert-base-uncased', 
                       max_length=512, batch_size=16):
    """
    Create DataLoader objects for training, validation and testing
    """
    train_texts, train_labels = train_data
    val_texts, val_labels = val_data
    test_texts, test_labels = test_data
    
    # Create datasets
    train_dataset = DocumentDataset(train_texts, train_labels, tokenizer_name, max_length)
    val_dataset = DocumentDataset(val_texts, val_labels, tokenizer_name, max_length)
    test_dataset = DocumentDataset(test_texts, test_labels, tokenizer_name, max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader