"""
Phase 4.12 — Complete Multi-Angle Multimodal Brain Tumor Model

Frozen architecture:

Axial MRI    -> EfficientNet-B0 -> 1280
Coronal MRI  -> EfficientNet-B0 -> 1280
Sagittal MRI -> EfficientNet-B0 -> 1280

3-view concat:
    3840 -> 1024 -> 512

Clinical:
    Age_scaled + Sex_encoded
    2 -> 32 -> 64

Multimodal:
    512 + 64 = 576
    576 -> 256

Classifier:
    256 -> 128 -> 4 raw logits
"""

import torch
import torch.nn as nn

from three_view_backbone import *
from image_fusion import ImageFusionBlock
from clinical_branch import ClinicalBranch
from multimodal_fusion import MultimodalFusionBlock
from tumor_classifier import TumorClassifier


THREE_VIEW_BACKBONE_CLASS_NAME = "ThreeViewEfficientNetBackbones"


def _flatten_feature(tensor):

    if tensor.ndim > 2:
        tensor = torch.flatten(tensor, 1)

    return tensor


class MultiAngleFusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = ThreeViewEfficientNetBackbones()

        self.image_fusion = ImageFusionBlock(
            input_dim=3840,
            hidden_dim=1024,
            output_dim=512,
            dropout=0.30
        )

        self.clinical_branch = ClinicalBranch(
            input_dim=2,
            hidden_dim=32,
            output_dim=64,
            dropout=0.20
        )

        self.multimodal_fusion = MultimodalFusionBlock(
            image_dim=512,
            clinical_dim=64,
            output_dim=256,
            dropout=0.30
        )

        self.classifier = TumorClassifier(
            input_dim=256,
            hidden_dim=128,
            num_classes=4,
            dropout=0.30
        )


    def _extract_backbone_features(
        self,
        axial,
        coronal,
        sagittal
    ):

        try:

            output = self.backbone(
                axial,
                coronal,
                sagittal
            )

            if isinstance(output, dict):

                lower = {
                    str(k).lower(): v
                    for k, v in output.items()
                }

                return (
                    _flatten_feature(lower["axial"]),
                    _flatten_feature(lower["coronal"]),
                    _flatten_feature(lower["sagittal"])
                )

            if isinstance(output, (tuple, list)) and len(output) == 3:

                return tuple(
                    _flatten_feature(x)
                    for x in output
                )

        except TypeError:
            pass


        branch_names = dir(self.backbone)


        def find_branch(keyword):

            matches = [
                name
                for name in branch_names
                if keyword in name.lower()
                and isinstance(
                    getattr(self.backbone, name),
                    nn.Module
                )
            ]

            if len(matches) != 1:
                raise RuntimeError(
                    f"Could not identify {keyword} branch: {matches}"
                )

            return getattr(
                self.backbone,
                matches[0]
            )


        return (
            _flatten_feature(
                find_branch("axial")(axial)
            ),

            _flatten_feature(
                find_branch("coronal")(coronal)
            ),

            _flatten_feature(
                find_branch("sagittal")(sagittal)
            )
        )


    def forward(
        self,
        axial,
        coronal,
        sagittal,
        clinical,
        return_features=False
    ):

        axial_f, coronal_f, sagittal_f = (
            self._extract_backbone_features(
                axial,
                coronal,
                sagittal
            )
        )

        image_f = self.image_fusion(
            axial_f,
            coronal_f,
            sagittal_f
        )

        clinical_f = self.clinical_branch(
            clinical
        )

        multimodal_f = self.multimodal_fusion(
            image_f,
            clinical_f
        )

        logits = self.classifier(
            multimodal_f
        )

        if return_features:

            return {
                "axial_feature": axial_f,
                "coronal_feature": coronal_f,
                "sagittal_feature": sagittal_f,
                "image_feature": image_f,
                "clinical_feature": clinical_f,
                "multimodal_feature": multimodal_f,
                "logits": logits
            }

        return logits
