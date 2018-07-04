import torch
import torch.nn as nn

BATCH_SIZE = 1
HIDDEN_UNITS = 512
NUM_LAYERS = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    def forward(self, input_tensor):
        h0, c0 = self.init_states()
        out, _ = self.lstm(input_tensor, (h0, c0))
        return out

    def init_states(self):
        h0 = torch.zeros(self.num_layers * 2, BATCH_SIZE, self.hidden_units, device=device)
        c0 = torch.zeros(self.num_layers * 2, BATCH_SIZE, self.hidden_units, device=device)

        return h0, c0
