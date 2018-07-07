import torch
import torch.nn as nn


class RNNModule(nn.Module):
    def __init__(self, input_size, num_layers, hidden_units, vocab_size, dropout=0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.lstm = nn.LSTM(input_size,
                            hidden_units,
                            num_layers,
                            bidirectional=True,
                            dropout=dropout)
        self.linear = nn.Linear(hidden_units * 2, vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input_tensor, device):
        h0, c0 = self.init_states(input_tensor.shape[1], device)
        lstm_out, _ = self.lstm(input_tensor, (h0, c0))
        linear_out = self.linear(lstm_out)
        probs = self.softmax(linear_out)
        return probs

    def init_states(self, batch_size, device):
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_units, device=device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_units, device=device)

        return h0, c0
