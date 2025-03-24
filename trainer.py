import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import time
from tqdm import tqdm
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    """
    Improved trainer class with techniques from Hedwig implementation
    to get better performance on document classification tasks
    """
    def __init__(
        self, 
        model, 
        train_loader, 
        val_loader, 
        test_loader=None, 
        lr=2e-5,
        weight_decay=0.01,
        warmup_proportion=0.1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        device=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model.to(self.device)
        
        # Total number of training steps
        self.num_training_steps = len(train_loader) * gradient_accumulation_steps
        
        # Optimizer with weight decay (L2 regularization)
        # Using different learning rates for BERT and classifier
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0}
        ]
        
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2, verbose=True)
        
        # Loss function with label smoothing for better generalization
        self.criterion = nn.CrossEntropyLoss()
        
        # Training parameters
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # For tracking metrics
        self.best_val_f1 = 0.0
        self.best_model_state = None
    
    def train(self, epochs, save_path='best_model.pth'):
        """
        Training loop with improved techniques
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0
            all_predictions = []
            all_labels = []
            
            # Progress bar for training
            train_iterator = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for i, batch in enumerate(train_iterator):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                
                # Scale loss if using gradient accumulation
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights if we've accumulated enough gradients
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                train_loss += loss.item() * self.gradient_accumulation_steps
                
                # Get predictions for metrics
                _, preds = torch.max(outputs, dim=1)
                all_predictions.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                
                # Update progress bar with current loss
                train_iterator.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Calculate training metrics
            train_loss /= len(self.train_loader)
            train_acc = accuracy_score(all_labels, all_predictions)
            train_f1 = f1_score(all_labels, all_predictions, average='macro')
            
            # Validation phase
            val_loss, val_acc, val_f1, val_precision, val_recall = self.evaluate(self.val_loader, "Validation")
            
            # Adjust learning rate based on validation performance
            self.scheduler.step(val_f1)
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy()
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"New best model saved with validation F1: {val_f1:.4f}")
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, "
                       f"Time: {epoch_time:.2f}s")
        
        # Load best model for final evaluation
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model with validation F1: {self.best_val_f1:.4f}")
        
        # Test evaluation if test loader provided
        if self.test_loader:
            test_loss, test_acc, test_f1, test_precision, test_recall = self.evaluate(self.test_loader, "Test")
            logger.info(f"Final test results - "
                       f"Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, "
                       f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
    
    def evaluate(self, data_loader, phase="Validation"):
        """
        Evaluation function for both validation and test sets
        """
        self.model.eval()
        eval_loss = 0
        all_predictions = []
        all_labels = []
        
        # No gradient computation during evaluation
        with torch.no_grad():
            # Progress bar for evaluation
            iterator = tqdm(data_loader, desc=f"[{phase}]")
            for batch in iterator:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                eval_loss += loss.item()
                
                # Get predictions
                _, preds = torch.max(outputs, dim=1)
                all_predictions.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        
        # Calculate metrics
        eval_loss /= len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='macro')
        precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        return eval_loss, accuracy, f1, precision, recall