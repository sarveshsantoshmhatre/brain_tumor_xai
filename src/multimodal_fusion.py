"""Fusion block for image and clinical embeddings."""

import torch
import torch.nn as nn


class MultimodalFusionBlock(nn.Module):
    def __init__(self, image_dim=512, clinical_dim=64, output_dim=256, dropout=0.30):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(image_dim + clinical_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, image_features, clinical_features):
        return self.fusion(torch.cat((image_features, clinical_features), dim=1))

