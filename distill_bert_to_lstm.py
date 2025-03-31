import argparse
import os
import logging
import torch
import random
import json
import numpy as np
from model import DocBERT
from models.lstm_model import DocumentBiLSTM
from dataset import load_data, create_data_loaders
from knowledge_distillation import DistillationTrainer
from transformers import BertTokenizer

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set all seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def tokenize_for_lstm(texts, bert_tokenizer, max_seq_length=512):
    """
    Convert BERT tokenization format to format suitable for LSTM
    This is a simple approach that just takes whole words from BERT tokenization
    """
    from collections import Counter
    
    # Create vocabulary from all texts
    word_counts = Counter()
    all_words = []
    
    for text in texts:
        # Simple tokenization by splitting on whitespace
        words = text.lower().split()
        word_counts.update(words)
        all_words.extend(words)
    
    # Create word->index mapping
    word2idx = {'<pad>': 0, '<unk>': 1}
    for idx, (word, _) in enumerate(word_counts.most_common(30000 - 2), 2):
        word2idx[word] = idx
    
    vocab_size = len(word2idx)
    logger.info(f"Created vocabulary with {vocab_size} tokens")
    
    return word2idx, vocab_size

def main():
    parser = argparse.ArgumentParser(description="Distill knowledge from BERT to LSTM for document classification")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file (CSV or TSV)")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text column")
    parser.add_argument("--label_column", type=str, default="label", help="Name of the label column")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation set split ratio")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test set split ratio")
    
    # BERT model arguments
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", help="BERT model to use")
    parser.add_argument("--bert_model_path", type=str, required=True, help="Path to saved BERT model weights")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    
    # LSTM model arguments
    parser.add_argument("--embedding_dim", type=int, default=300, help="Dimension of word embeddings in LSTM")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of LSTM")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")
    
    # Distillation arguments
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperature for softening probability distributions")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for distillation loss vs. regular loss")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes to predict")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for LSTM")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save models")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load and prepare data for both BERT and LSTM
    logger.info("Loading and preparing data...")
    
    # Load data first
    train_data, val_data, test_data = load_data(
        args.data_path,
        text_col=args.text_column,
        label_col=args.label_column,
        validation_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )
    
    # Create BERT data loaders
    logger.info("Creating BERT data loaders...")
    bert_train_loader, bert_val_loader, bert_test_loader = create_data_loaders(
        train_data, 
        val_data, 
        test_data,
        tokenizer_name=args.bert_model,
        max_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_classes=args.num_classes
    )
    
    vocab_size = bert_train_loader.tokenizer.vocab_size
    
    logger.info(f"LSTM Vocabulary size: {vocab_size}")
    
    # Load pre-trained BERT model (teacher)
    logger.info("Loading pre-trained BERT model (teacher)...")
    bert_model = DocBERT(
        num_classes=args.num_classes,
        bert_model_name=args.bert_model,
        dropout_prob=0.1
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load saved BERT weights
    bert_model.load_state_dict(torch.load(args.bert_model_path, map_location=device))
    logger.info(f"Loaded teacher model from {args.bert_model_path}")
    
    # Initialize LSTM model (student)
    logger.info("Initializing LSTM model (student)...")
    lstm_model = DocumentBiLSTM(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.num_classes,
        n_layers=args.num_layers,
        dropout=args.dropout
    )
    
    # Print model sizes for comparison
    bert_params = sum(p.numel() for p in bert_model.parameters())
    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    logger.info(f"BERT model size: {bert_params:,} parameters")
    logger.info(f"LSTM model size: {lstm_params:,} parameters")
    logger.info(f"Size reduction: {bert_params / lstm_params:.1f}x")
    
    # Initialize distillation trainer
    trainer = DistillationTrainer(
        teacher_model=bert_model,
        student_model=lstm_model,
        train_loader=bert_train_loader,  # Using BERT loader to match tokenization
        val_loader=bert_val_loader,
        test_loader=bert_test_loader,
        temperature=args.temperature,
        alpha=args.alpha,
        lr=args.learning_rate,
        weight_decay=1e-5
    )
    
    # Train with knowledge distillation
    logger.info("Starting knowledge distillation...")
    save_path = os.path.join(args.output_dir, "distilled_lstm_model.pth")
    trainer.train(epochs=args.epochs, save_path=save_path)

    logger.info("Knowledge distillation completed!")

if __name__ == "__main__":
    main()