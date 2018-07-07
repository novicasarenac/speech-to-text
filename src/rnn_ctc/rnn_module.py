import torch
import torch.nn as nn


class RNNModule(nn.Module):
    def __init__(self, input_size, num_layers, hidden_units, dropout=0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.lstm = nn.LSTM(input_size,
                            hidden_units,
                            num_layers,
                            bidirectional=True,
                            dropout=dropout)

    def forward(self, input_tensor, device):
        h0, c0 = self.init_states(input_tensor.shape[1], device)
        out, _ = self.lstm(input_tensor, (h0, c0))
        return out

    def init_states(self, batch_size, device):
        h0 = torch.zeros(self.num_layers * 2, BATCH_SIZE, self.hidden_units, device=device)
        c0 = torch.zeros(self.num_layers * 2, BATCH_SIZE, self.hidden_units, device=device)

        return h0, c0
