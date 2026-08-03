"""Inference-only TrpB oracle architecture."""

import torch.nn as nn


HIDDEN_DIM = 400
DROPOUT = 0.1


class OracleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, dropout_rate=DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        return self.fc2(self.dropout(self.relu(self.fc1(inputs))))
