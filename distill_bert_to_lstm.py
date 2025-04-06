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



def main():
    parser = argparse.ArgumentParser(description="Distill knowledge from BERT to LSTM for document classification")
    
    # Data arguments
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the dataset file (CSV or TSV)")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to the validation dataset file (CSV or TSV)")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test dataset file (CSV or TSV)")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text column")
    parser.add_argument("--label_column", type=str, nargs="+", help="Name of the label column")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation set split ratio")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test set split ratio")
    
    # BERT model arguments
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", help="BERT model to use")
    parser.add_argument("--bert_model_path", type=str, required=True, help="Path to saved BERT model weights")
    parser.add_argument("--max_seq_length", type=int, default=250, help="Maximum sequence length (e.g., 250 for PhoBERT as PhoBERT allows max_position_embeddings=258)")
    
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
    label_column = args.label_column[0] if isinstance(args.label_column, list) and len(args.label_column) == 1 else args.label_column
    num_categories = len(args.label_column) if isinstance(args.label_column, list) else 1
    
    train_data, _, _ = load_data(
        args.data_path,
        text_col=args.text_column,
        label_col=label_column,
        validation_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )

    _, val_data, _ = load_data(
        args.val_data_path,
        text_col=args.text_column,
        label_col=label_column,
        validation_split=1.0,
        test_split=0.0,
        seed=args.seed
    )
    _, _, test_data = load_data(
        args.test_data_path,
        text_col=args.text_column,
        label_col=label_column,
        validation_split=0.0,
        test_split=1.0,
        seed=args.seed
    )
    
    # Create BERT data loaders
    logger.info("Creating BERT data loaders...")
    bert_train_dataset, bert_val_dataset, bert_test_dataset = create_data_loaders(
        train_data, 
        val_data, 
        test_data,
        tokenizer_name=args.bert_model,
        max_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        return_datasets=True
    )

    # Create dataloaders 
    bert_train_loader = torch.utils.data.DataLoader(bert_train_dataset, batch_size=args.batch_size, shuffle=True)
    bert_val_loader = torch.utils.data.DataLoader(bert_val_dataset, batch_size=args.batch_size)
    bert_test_loader = torch.utils.data.DataLoader(bert_test_dataset, batch_size=args.batch_size)
    
    vocab_size = bert_train_dataset.tokenizer.vocab_size
    
    logger.info(f"LSTM Vocabulary size: {vocab_size}")
    
    # Load pre-trained BERT model (teacher)
    logger.info("Loading pre-trained BERT model (teacher)...")
    bert_model = DocBERT(
        num_classes=args.num_classes,
        bert_model_name=args.bert_model,
        dropout_prob=0.1,
        num_categories=num_categories
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
        output_dim=args.num_classes * num_categories,
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
        num_categories=num_categories,
        num_classes=args.num_classes,
        weight_decay=1e-5
    )
    
    # Train with knowledge distillation
    logger.info("Starting knowledge distillation...")
    save_path = os.path.join(args.output_dir, "distilled_lstm_model.pth")
    trainer.train(epochs=args.epochs, save_path=save_path)

    logger.info("Knowledge distillation completed!")

if __name__ == "__main__":
    main()