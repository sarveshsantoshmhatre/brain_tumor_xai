"""Clinical feature encoder for standardized age and encoded sex."""

import torch.nn as nn


class ClinicalBranch(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=64, dropout=0.20):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, clinical):
        return self.network(clinical)

