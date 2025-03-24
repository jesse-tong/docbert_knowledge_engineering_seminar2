"""
Simple script to run the DocBERT model with predefined config presets
"""
import argparse
import logging
import os
from config import get_config
from model import DocBERT
from dataset import load_data, create_data_loaders
from trainer import Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run DocBERT with a predefined config")
    
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file (CSV or TSV)")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text column")
    parser.add_argument("--label_column", type=str, default="label", help="Name of the label column")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes to predict")
    parser.add_argument("--config", type=str, default="default", 
                        choices=["default", "short_text", "long_document", "fine_tuning"],
                        help="Configuration preset to use")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save outputs")
    
    args = parser.parse_args()
    
    # Get config
    config_class = get_config(args.config)
    config = config_class()
    
    logger.info(f"Using '{args.config}' config preset")
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load and prepare data
    logger.info("Loading data...")
    train_data, val_data, test_data = load_data(
        args.data_path,
        text_col=args.text_column,
        label_col=args.label_column,
        validation_split=config.val_split,
        test_split=config.test_split,
        seed=config.seed
    )
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, 
        val_data, 
        test_data,
        tokenizer_name=config.bert_model,
        max_length=config.max_seq_length,
        batch_size=config.batch_size
    )
    
    # Initialize model
    logger.info(f"Initializing model with {config.bert_model}...")
    model = DocBERT(
        num_classes=args.num_classes,
        bert_model_name=config.bert_model,
        dropout_prob=config.dropout
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_accumulation_steps=config.grad_accum_steps
    )
    
    # Train model
    logger.info("Starting training...")
    save_path = os.path.join(args.output_dir, "best_model.pth")
    trainer.train(epochs=config.epochs, save_path=save_path)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()