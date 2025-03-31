import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging
import os

logger = logging.getLogger(__name__)

class DistillationTrainer:
    """
    Trainer for knowledge distillation from teacher model (BERT) to student model (LSTM)
    """
    def __init__(
        self, 
        teacher_model, 
        student_model,
        train_loader, 
        val_loader, 
        test_loader=None,
        temperature=2.0,
        alpha=0.5,  # Weight for distillation loss vs. regular loss
        lr=0.001,
        weight_decay=1e-5,
        max_grad_norm=1.0,
        label_mapping=None,
        device=None
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.temperature = temperature
        self.alpha = alpha
        self.max_grad_norm = max_grad_norm
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Move models to device
        self.teacher_model.to(self.device)
        self.student_model.to(self.device)
        
        # Set teacher model to evaluation mode
        self.teacher_model.eval()
        
        # Optimizer for student model
        self.optimizer = torch.optim.Adam(
            self.student_model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()  # For hard targets
        
        # Tracking metrics
        self.best_val_f1 = 0.0
        self.best_model_state = None
        self.label_mapping = label_mapping
    
    def distillation_loss(self, student_logits, teacher_logits, labels, temperature, alpha):
        """
        Compute the knowledge distillation loss
        
        Args:
            student_logits: Output from student model
            teacher_logits: Output from teacher model
            labels: Ground truth labels
            temperature: Temperature for softening probability distributions
            alpha: Weight for distillation loss vs. cross-entropy loss
            
        Returns:
            Combined loss
        """
        # Softmax with temperature for soft targets
        soft_targets = F.softmax(teacher_logits / temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / temperature, dim=1)
        
        # Distillation loss (KL divergence)
        distill_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
        
        # Standard cross entropy with hard targets
        ce_loss = self.ce_loss(student_logits, labels)
        
        # Weighted combination of the two losses
        loss = alpha * distill_loss + (1 - alpha) * ce_loss
        
        return loss
    
    def train(self, epochs, save_path='best_distilled_model.pth'):
        """
        Train student model with knowledge distillation
        """
        logger.info(f"Starting distillation training for {epochs} epochs")
        logger.info(f"Temperature: {self.temperature}, Alpha: {self.alpha}")
        
        for epoch in range(epochs):
            self.student_model.train()
            train_loss = 0.0
            all_preds = []
            all_labels = []
            
            # Training loop
            train_iterator = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch in train_iterator:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get teacher predictions (no grad needed for teacher)
                with torch.no_grad():
                    teacher_logits = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                # Forward pass through student model
                student_logits = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Calculate distillation loss
                loss = self.distillation_loss(
                    student_logits, 
                    teacher_logits, 
                    labels, 
                    self.temperature, 
                    self.alpha
                )
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy for progress tracking
                _, preds = torch.max(student_logits, 1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                
                # Update progress bar
                train_iterator.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Calculate training metrics
            train_loss = train_loss / len(self.train_loader)
            train_acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_preds)
            
            # Evaluate on validation set
            val_loss, val_acc, val_f1 = self.evaluate()
            
            # Update learning rate based on validation performance
            self.scheduler.step(val_f1)
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_state = self.student_model.state_dict().copy()
                torch.save({
                    'model_state_dict': self.student_model.state_dict(),
                    'label_mapping': self.label_mapping,
                }, save_path)
                logger.info(f"New best model saved with validation F1: {val_f1:.4f}")
            
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Load best model for final evaluation
        if self.best_model_state is not None:
            self.student_model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model with validation F1: {self.best_val_f1:.4f}")
        
        # Final evaluation on test set if provided
        if self.test_loader:
            test_loss, test_acc, test_f1 = self.evaluate(self.test_loader, "Test")
            logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    
    def evaluate(self, data_loader=None, phase="Validation"):
        """
        Evaluate the student model
        """
        if data_loader is None:
            data_loader = self.val_loader
        
        self.student_model.eval()
        eval_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"[{phase}]"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass through student
                student_logits = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Calculate regular CE loss (no distillation during evaluation)
                loss = self.ce_loss(student_logits, labels)
                eval_loss += loss.item()
                
                # Get predictions
                _, preds = torch.max(student_logits, 1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        
        # Calculate metrics
        eval_loss = eval_loss / len(data_loader)
        
        # Accuracy
        accuracy = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_preds)
        
        # F1 score (macro-averaged)
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return eval_loss, accuracy, f1