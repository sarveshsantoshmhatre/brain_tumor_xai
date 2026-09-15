"""Three independent grayscale EfficientNet-B0 feature extractors."""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


def _make_branch():
    model = efficientnet_b0(weights=None)

    # MRI slices are single-channel. Keep the original module layout so the
    # trained checkpoint keys remain compatible with torchvision EfficientNet.
    original = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        1,
        original.out_channels,
        kernel_size=original.kernel_size,
        stride=original.stride,
        padding=original.padding,
        bias=False,
    )
    model.classifier = nn.Identity()
    return model


class ThreeViewEfficientNetBackbones(nn.Module):
    def __init__(self):
        super().__init__()
        self.axial_branch = _make_branch()
        self.coronal_branch = _make_branch()
        self.sagittal_branch = _make_branch()

    def forward(self, axial, coronal, sagittal):
        return (
            self.axial_branch(axial),
            self.coronal_branch(coronal),
            self.sagittal_branch(sagittal),
        )

