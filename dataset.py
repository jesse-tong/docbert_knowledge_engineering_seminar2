import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DocumentDataset(Dataset):
    """
    Dataset class for document classification
    with improved preprocessing and batching
    """
    def __init__(self, texts, labels, tokenizer_name='bert-base-uncased', max_length=512, num_classes=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Validate labels
        unique_labels = set(labels)
        min_label = min(unique_labels) if unique_labels else 0
        max_label = max(unique_labels) if unique_labels else 0
        
        # Log warning if labels might be out of range
        if num_classes is not None and (min_label < 0 or max_label >= num_classes):
            logger.warning(f"LABEL RANGE ERROR: Labels must be between 0 and {num_classes-1}, "
                          f"but found range [{min_label}, {max_label}]")
            logger.warning(f"Unique label values: {sorted(unique_labels)}")
            
            # Fix labels by remapping them to start from 0 (some datasets might have labels starting from 1)
            if min_label != 0:
                logger.warning(f"Auto-correcting labels to be zero-indexed...")
                label_map = {original: idx for idx, original in enumerate(sorted(unique_labels))}
                self.labels = np.array([label_map[label] for label in labels])
                logger.warning(f"New unique label values: {sorted(set(self.labels))}")

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

    def get_text_(self, idx):
        """Get original text for a given index"""
        return {
            'text': self.texts[idx],
            'label': self.labels[idx]
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
        label_map = {label: idx for idx, label in enumerate(sorted(df[label_col].unique()))}
        df['label_numeric'] = df[label_col].map(label_map)
        labels = df['label_numeric'].values
        
        # Log the mapping for reference
        logger.info(f"Label mapping: {label_map}")
    else:
        labels = df[label_col].values
        
        # Check if labels start from 0
        min_label = labels.min()
        if min_label != 0:
            logger.warning(f"Labels don't start from 0 (min={min_label}). Converting to zero-indexed...")
            label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
            labels = np.array([label_map[label] for label in labels])
    
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
    
    # Log stats about the dataset
    logger.info(f"Dataset splits: train={train_size}, val={val_size}, test={test_size}")
    logger.info(f"Label distribution in train set: {np.bincount(train_labels)}")
    
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)

def create_data_loaders(train_data, val_data, test_data, tokenizer_name='bert-base-uncased', 
                       max_length=512, batch_size=16, num_classes=None):
    """
    Create DataLoader objects for training, validation and testing
    """
    train_texts, train_labels = train_data
    val_texts, val_labels = val_data
    test_texts, test_labels = test_data
    
    # Create datasets
    train_dataset = DocumentDataset(train_texts, train_labels, tokenizer_name, max_length, num_classes)
    val_dataset = DocumentDataset(val_texts, val_labels, tokenizer_name, max_length, num_classes)
    test_dataset = DocumentDataset(test_texts, test_labels, tokenizer_name, max_length, num_classes)
    
    # Create data loaders
    if len(train_dataset.texts) == 0:
        logger.warning("Training dataset is empty. Check your data loading and splitting.")
        train_loader = None
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if len(val_dataset.texts) == 0:
        logger.warning("Validation dataset is empty. Check your data loading and splitting.")
        val_loader = None
    else:
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    if len(test_dataset.texts) == 0:
        logger.warning("Test dataset is empty. Check your data loading and splitting.")
        test_loader = None
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader