import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import GloVe  # For loading pre-trained word embeddings

class DocumentLSTM(nn.Module):
    """
    LSTM model for document classification using GloVe embeddings
    """
    def __init__(self, num_classes, vocab_size=30000, embedding_dim=300, 
                 hidden_dim=256, num_layers=2, bidirectional=True, 
                 dropout_rate=0.3, use_pretrained=True, padding_idx=0):
        super(DocumentLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer (with option to use pre-trained GloVe)
        if use_pretrained:
            # Initialize with GloVe embeddings
            try:
                glove = GloVe(name='6B', dim=embedding_dim)
                # You'd need to map your vocabulary to GloVe indices
                # This is a simplified placeholder
                self.embedding = nn.Embedding.from_pretrained(
                    glove.vectors[:vocab_size], 
                    padding_idx=padding_idx,
                    freeze=False
                )
            except Exception as e:
                print(f"Could not load pretrained embeddings: {e}")
                # Fall back to random initialization
                self.embedding = nn.Embedding(
                    vocab_size, embedding_dim, padding_idx=padding_idx
                )
        else:
            # Random initialization
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=padding_idx
            )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * self.num_directions, 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * self.num_directions)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification layer
        self.classifier = nn.Linear(hidden_dim * self.num_directions, num_classes)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Forward pass through LSTM model
        
        Args:
            input_ids: Tensor of token ids [batch_size, seq_len]
            attention_mask: Tensor indicating which tokens to attend to [batch_size, seq_len]
        """
        # Word embeddings
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: [batch_size, seq_len, hidden_dim * num_directions]
        
        # Apply attention
        if attention_mask is not None:
            # Apply attention mask (1 for tokens to attend to, 0 for padding)
            attention_mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            attention_scores = self.attention(lstm_out)  # [batch_size, seq_len, 1]
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e10)
            attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len, 1]
            
            # Weighted sum
            context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, hidden_dim * num_directions]
        else:
            # If no attention mask, use the last hidden state
            if self.bidirectional:
                # For bidirectional LSTM, concatenate last hidden states from both directions
                last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch_size, hidden_dim * 2]
            else:
                last_hidden = hidden[-1]  # [batch_size, hidden_dim]
            
            context_vector = last_hidden
        
        # Layer normalization
        normalized = self.layer_norm(context_vector)
        
        # Dropout
        dropped = self.dropout(normalized)
        
        # Classification
        logits = self.classifier(dropped)
        
        return logits

class DocumentBiLSTM(nn.Module):
    """
    A simpler BiLSTM implementation that doesn't require pre-loaded embeddings
    Good for getting started quickly
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers=2, dropout=0.5, pad_idx=0):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=True, 
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # input_ids = [batch size, seq len]
        
        # embedded = [batch size, seq len, emb dim]
        embedded = self.embedding(input_ids)
        
        # Apply dropout to embeddings
        embedded = self.dropout(embedded)
        
        if attention_mask is not None:
            # Create packed sequence for variable length sequences
            # This is a simplified version - in practice you'd use pack_padded_sequence
            # but that requires knowing the actual sequence lengths
            pass
            
        # output = [batch size, seq len, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]
        # cell = [n layers * num directions, batch size, hid dim]
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Apply dropout to hidden state
        hidden = self.dropout(hidden)
            
        # prediction = [batch size, output dim]
        prediction = self.fc(hidden)
        
        return prediction