import torch
import torch.nn as nn
import torch.nn.functional as F

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