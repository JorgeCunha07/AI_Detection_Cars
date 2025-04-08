import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.permute(1, 0, 2)  # (batch, 1, hidden_dim)
        hidden = hidden.repeat(1, encoder_outputs.size(1), 1)  # (batch, seq_len, hidden_dim)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)
