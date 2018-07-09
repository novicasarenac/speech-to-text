import torch
import torch.nn as nn

from torch.autograd import Variable


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
        self.batch_norm = nn.BatchNorm1d(hidden_units * 2)
        self.linear = nn.Linear(hidden_units * 2, vocab_size)

        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input_tensor, device):
        h0, c0 = self.init_states(input_tensor.shape[1], device)
        lstm_out, _ = self.lstm(input_tensor)
        lstm_out = lstm_out.squeeze().unsqueeze(2)
        norm = self.batch_norm(lstm_out)
        norm = norm.squeeze().unsqueeze(1)
        linear_out = self.linear(norm)
        return linear_out

    def init_states(self, batch_size, device):
        h0 = nn.init.xavier_uniform_(torch.zeros(self.num_layers * 2,
                                                 batch_size,
                                                 self.hidden_units,
                                                 device=device))
        c0 = nn.init.xavier_uniform_(torch.zeros(self.num_layers * 2,
                                                 batch_size,
                                                 self.hidden_units,
                                                 device=device))

        return Variable(h0), Variable(c0)
