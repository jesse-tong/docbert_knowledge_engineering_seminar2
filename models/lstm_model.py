import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class DocumentBiLSTM(nn.Module):
    """
    BiLSTM implementation with stability improvements inspired by DocBERT
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
        
        # Add layer normalization for stability (like in DocBERT)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # input_ids = [batch size, seq len]
        
        # embedded = [batch size, seq len, emb dim]
        embedded = self.embedding(input_ids)
        
        # Apply dropout to embeddings
        embedded = self.dropout(embedded)
        
        # Initialize hidden and cell variables
        hidden = None
        cell = None
        
        if attention_mask is not None:
            # Convert attention mask to sequence lengths
            seq_lengths = attention_mask.sum(dim=1).to(torch.int64).cpu()
            
            # Sort sequences by decreasing length
            seq_lengths, indices = torch.sort(seq_lengths, descending=True)
            sorted_embedded = embedded[indices]
            
            # Pack the embedded sequences
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                sorted_embedded, seq_lengths, batch_first=True, enforce_sorted=True
            )
            
            # Pass through LSTM
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            
            # Unpack the sequence
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            
            # Get the hidden states in correct order
            _, restore_indices = torch.sort(indices)
            hidden = hidden[:, restore_indices]
        else:
            # Standard processing without masking
            _, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate the final forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Apply layer normalization (improves stability)
        normalized = self.layer_norm(hidden_cat)
        
        # Apply dropout to hidden state
        dropped = self.dropout(normalized)
            
        # prediction = [batch size, output dim]
        prediction = self.fc(dropped)
        
        return prediction