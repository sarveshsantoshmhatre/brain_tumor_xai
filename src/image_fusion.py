"""Fusion block for the three 1280-dimensional image embeddings."""

import torch
import torch.nn as nn


class ImageFusionBlock(nn.Module):
    def __init__(self, input_dim=3840, hidden_dim=1024, output_dim=512, dropout=0.30):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, axial, coronal, sagittal):
        return self.fusion(torch.cat((axial, coronal, sagittal), dim=1))

