"""Four-class tumor classification head."""

import torch.nn as nn


class TumorClassifier(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_classes=4, dropout=0.30):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        return self.classifier(features)

