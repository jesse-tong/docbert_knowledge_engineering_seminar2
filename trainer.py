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
        num_classes=2,
        num_categories=1,
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
        # Exclude layer normalization and bias from weight decay
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
        
        # As this is a multi-class classification task, we use CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss()
        
        # Training parameters
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # For tracking metrics
        self.best_val_f1 = 0.0
        self.best_model_state = None

        self.num_classes = num_classes  # Number of classes for classification
        # For training if using multiple categories (e.g., multiple sentiment classes, there can be multiple sentiment in one document)
        self.num_categories = num_categories
    
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
                
                if self.num_categories > 1:
                    total_loss = 0
                    for i in range(self.num_categories):
                        start_idx = i * self.num_classes
                        end_idx = (i + 1) * self.num_classes
                        category_outputs = outputs[:, start_idx:end_idx] # Shape (batch, num_classes)
                        category_labels = labels[:, i] # Shape (batch)
                        # Ensure category_labels are in [0, self.num_classes - 1]
                        if category_labels.max() >= self.num_classes or category_labels.min() < 0:
                            print(f"ERROR: Category {i} labels out of range [0, {self.num_classes - 1}]: min={category_labels.min()}, max={category_labels.max()}")
                            
                        total_loss += self.criterion(category_outputs, category_labels)

                    loss = total_loss / self.num_categories # Average loss
                else:
                    loss = self.criterion(outputs, labels)
                
                # Scale loss if using gradient accumulation
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                train_loss += loss.item() * self.gradient_accumulation_steps
                
                # Get predictions for metrics
                if self.num_categories > 1:
                    batch_size, total_classes = outputs.shape
                    if total_classes % self.num_categories != 0:
                        raise ValueError(f"Error: Number of total classes in the batch must of divisible by {self.num_categories}")

                    classes_per_group = total_classes // self.num_categories
                    # Group every classes_per_group values along dim=1
                    reshaped = outputs.view(outputs.size(0), -1, classes_per_group)  # shape: (batch, self., classes_per_group)

                    # Argmax over each group of classes_per_group
                    preds = reshaped.argmax(dim=-1)
                else:
                    _, preds = torch.max(outputs, dim=1)

                all_predictions.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                
                train_iterator.set_postfix({'loss': f"{loss.item():.4f}"})
            
            train_loss /= len(self.train_loader)
            if self.num_categories > 1:

                all_predictions = np.concatenate(all_predictions)
                all_labels = np.concatenate(all_labels)
                
                train_acc = accuracy_score(all_labels, all_predictions)
                train_f1 = f1_score(all_labels, all_predictions, average='macro')
            else:
                train_acc = accuracy_score(all_labels, all_predictions)
                train_f1 = f1_score(all_labels, all_predictions, average='macro')
            
            # Validation phase
            val_loss, val_acc, val_f1, val_precision, val_recall = self.evaluate(self.val_loader, "Validation")

            logger.info(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, "
                        f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
            
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
            print(f"Epoch {epoch+1}/{epochs} - ",
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, ",
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, ",
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
            print(f"Final test results - ",
                       f"Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, ",
                       f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
    
    def evaluate(self, data_loader, phase="Validation", threshold=0.55):
        """
        Evaluation function for both validation and test sets
        """
        self.model.eval()
        eval_loss = 0
        all_predictions = np.array([], dtype=int)
        all_labels = np.array([], dtype=int)
        
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
                if self.num_categories > 1:
                    total_loss = 0
                    for i in range(self.num_categories):
                        start_idx = i * self.num_classes
                        end_idx = (i + 1) * self.num_classes
                        category_outputs = outputs[:, start_idx:end_idx] # Shape (batch, num_classes)
                        category_labels = labels[:, i] # Shape (batch)

                        # Ensure category_labels are in [0, self.num_classes - 1]
                        if category_labels.max() >= self.num_classes or category_labels.min() < 0:
                            print(f"ERROR: Category {i} labels out of range [0, {self.num_classes - 1}]: min={category_labels.min()}, max={category_labels.max()}")
                            
                        total_loss += self.criterion(category_outputs, category_labels)

                    loss = total_loss / self.num_categories # Average loss
                else:
                    loss = self.criterion(outputs, labels)

                eval_loss += loss.item()
                
                # Get predictions
                # Get predictions for metrics
                if self.num_categories > 1:
                    batch_size, total_classes = outputs.shape
                    if total_classes % self.num_categories != 0:
                        raise ValueError(f"Error: Number of total classes in the batch must of divisible by {self.num_categories}")

                    classes_per_group = total_classes // self.num_categories
                    # Group every classes_per_group values along dim=1
                    reshaped = outputs.view(outputs.size(0), -1, classes_per_group)  # shape: (batch, self., classes_per_group)

                    # Softmax and apply threshold
                    probs = torch.softmax(reshaped, dim=1)
                    probs = torch.where(probs > threshold, probs, 0.0)
                    # Argmax over each group of classes_per_group
                    preds = probs.argmax(dim=-1)
                else:
                    _, preds = torch.max(outputs, dim=1)

                all_predictions = np.append(all_predictions, preds.cpu().tolist())
                all_labels = np.append(all_labels, labels.cpu().tolist())
        

        # Calculate metrics
        eval_loss /= len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        return eval_loss, accuracy, f1, precision, recall