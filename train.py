import argparse
import os
import logging
import torch
import random
import numpy as np
from model import DocBERT
from dataset import load_data, create_data_loaders
from trainer import Trainer

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
    parser = argparse.ArgumentParser(description="Train a document classification model with BERT")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file (CSV or TSV)")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text column")
    parser.add_argument("--label_column", type=str, default="label", help="Name of the label column")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation set split ratio")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test set split ratio")
    
    # Model arguments
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", 
                        help="BERT model to use (e.g., bert-base-uncased, bert-large-uncased)")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes to predict")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Proportion of training for LR warmup")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the model and logs")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Log args for debugging
    logger.info(f"Running with arguments: {args}")
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    train_data, val_data, test_data = load_data(
        args.data_path,
        text_col=args.text_column,
        label_col=args.label_column,
        validation_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, 
        val_data, 
        test_data,
        tokenizer_name=args.bert_model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_classes=args.num_classes
    )
    
    logger.info(f"Train samples: {len(train_data[0])}, "
               f"Validation samples: {len(val_data[0])}, "
               f"Test samples: {len(test_data[0])}")
    
    # Initialize model
    logger.info(f"Initializing DocBERT model with {args.bert_model}...")
    model = DocBERT(
        num_classes=args.num_classes,
        bert_model_name=args.bert_model,
        dropout_prob=args.dropout
    )
    
    # Count and log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_proportion=args.warmup_proportion,
        gradient_accumulation_steps=args.grad_accum_steps
    )
    
    # Train the model
    logger.info("Starting training...")
    save_path = os.path.join(args.output_dir, "best_model.pth")
    trainer.train(epochs=args.epochs, save_path=save_path)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()