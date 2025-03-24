import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class DocBERT(nn.Module):
    """
    Document classification using BERT with improved architecture
    based on Hedwig implementation patterns.
    """
    def __init__(self, num_classes, bert_model_name='bert-base-uncased', dropout_prob=0.1):
        super(DocBERT, self).__init__()
        
        # Load pre-trained BERT model or config
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.config = self.bert.config
        
        # Dropout layer for regularization (helps prevent overfitting)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Multiple classification heads approach (inspired by Hedwig)
        self.hidden_size = self.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # Layer normalization before classification (helps stabilize training)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass through the model
        """
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, 
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        
        # Get the [CLS] token representation (first token)
        pooled_output = outputs.pooler_output
        
        # Apply layer normalization
        normalized_output = self.layer_norm(pooled_output)
        
        # Apply dropout for regularization
        dropped_output = self.dropout(normalized_output)
        
        # Pass through the classifier
        logits = self.classifier(dropped_output)
        
        return logits