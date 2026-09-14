#model
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, pad_idx, cell_type='lstm'):
        super(SentimentRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower() # lstm or rnn

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)

        if self.cell_type == 'lstm':
            # dropout when len(layers) > 1
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)
        else:
            self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)

        self.dropout = nn.Dropout(dropout)
        # score of pos or neg
        self.fc = nn.Linear(hidden_size, 2)
    
    def forward(self, x, lenghts):
        embedded = self.embedding(x)
        packed_embedded = pack_padded_sequence(embedded, lenghts.cpu(), batch_first=True, enforce_sorted=True)

        if self.cell_type == 'lstm':
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        else:
            packed_output, hidden = self.rnn(packed_embedded)
        
        final_hidden = hidden[-1,:,:]
        out = self.dropout(final_hidden)
        logits = self.fc(out)

        return logits

