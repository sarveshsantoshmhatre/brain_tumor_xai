from pathlib import Path
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage import gaussian_filter
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multi_angle_fusion_model import MultiAngleFusionModel


CLASS_NAMES = [
    "GLIOMA",
    "MENINGIOMA",
    "METASTASIS",
    "PITUITARY",
]

TRAIN_AGE_MEAN = 57.94320357370772
TRAIN_AGE_STD = 14.663421122942468
TRAIN_AGE_MEDIAN = 59.0

BASELINE_AGE_SCALED = 9.068828790811745e-17
BASELINE_SEX = 0.4582003828972559

CHECKPOINT_PATH = (
    ROOT /
    "checkpoints/attention_guided_final_a100/best.pt"
)


# ================================================================================================================
# MODEL
# ================================================================================================================

class SpatialAttentionHead(nn.Module):

    def __init__(self, in_channels=112):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                64,
                1,
                kernel_size=1
            )
        )

    def forward(self, x):
        return self.net(x)


class FinalAttentionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.core = MultiAngleFusionModel()

        self.axial_attention_head = SpatialAttentionHead()
        self.coronal_attention_head = SpatialAttentionHead()
        self.sagittal_attention_head = SpatialAttentionHead()

        self._captured = {}

        self._capture_hooks = [
            self.core.backbone.axial_branch.features[5]
            .register_forward_hook(self._hook("axial")),

            self.core.backbone.coronal_branch.features[5]
            .register_forward_hook(self._hook("coronal")),

            self.core.backbone.sagittal_branch.features[5]
            .register_forward_hook(self._hook("sagittal")),
        ]

    def _hook(self, name):

        def capture(module, inputs, output):
            self._captured[name] = output

        return capture

    def forward(
        self,
        axial,
        coronal,
        sagittal,
        clinical
    ):

        self._captured = {}

        output = self.core(
            axial,
            coronal,
            sagittal,
            clinical
        )

        if torch.is_tensor(output):
            return output

        if isinstance(output, dict):
            return output["logits"]

        return output[0]


# ================================================================================================================
# DEVICE + MODEL LOADER
# ================================================================================================================

def resolve_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_final_model(
    checkpoint_path=CHECKPOINT_PATH,
    device=None
):

    if device is None:
        device = resolve_device()

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    model = FinalAttentionModel().to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    if parameter_count != 16_860_379:
        raise RuntimeError(
            f"Architecture mismatch: {parameter_count:,}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    return model, checkpoint, device


# ================================================================================================================
# RAW MRI PREPROCESSING
# ================================================================================================================

def load_raw_mri_image(
    image_path
):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            image_path
        )

    supported = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    if image_path.suffix.lower() not in supported:
        raise ValueError(
            f"Unsupported format: {image_path.suffix}"
        )

    image = Image.open(
        image_path
    ).convert("F")

    image = np.asarray(
        image,
        dtype=np.float32
    )

    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale MRI, got {image.shape}"
        )

    return image


def normalize_mri_slice(
    image
):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    finite = np.isfinite(
        image
    )

    if not finite.any():
        raise ValueError(
            "No finite MRI pixels."
        )

    image = image.copy()

    image[
        ~finite
    ] = np.median(
        image[finite]
    )

    p1 = float(
        np.percentile(
            image,
            1
        )
    )

    p99 = float(
        np.percentile(
            image,
            99
        )
    )

    if p99 <= p1:
        raise ValueError(
            "Insufficient MRI intensity variation."
        )

    image = np.clip(
        image,
        p1,
        p99
    )

    image = (
        image - p1
    ) / (
        p99 - p1
    )

    return np.clip(
        image,
        0,
        1
    ).astype(np.float32)


def resize_and_pad_224(
    image
):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    h, w = image.shape

    scale = min(
        224 / h,
        224 / w
    )

    new_h = max(
        1,
        int(round(h * scale))
    )

    new_w = max(
        1,
        int(round(w * scale))
    )

    tensor = torch.from_numpy(
        image
    )[None, None]

    resized = F.interpolate(
        tensor,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False
    )[0, 0]

    canvas = torch.zeros(
        (224, 224),
        dtype=torch.float32
    )

    top = (
        224 - new_h
    ) // 2

    left = (
        224 - new_w
    ) // 2

    canvas[
        top:top + new_h,
        left:left + new_w
    ] = resized

    return (
        canvas.numpy()
        .astype(np.float32)
    )


def preprocess_raw_upload(
    image_path
):

    raw = load_raw_mri_image(
        image_path
    )

    normalized = normalize_mri_slice(
        raw
    )

    processed = resize_and_pad_224(
        normalized
    )

    if processed.shape != (224, 224):
        raise RuntimeError(
            "Processed MRI must be 224x224."
        )

    if not np.isfinite(processed).all():
        raise RuntimeError(
            "Processed MRI contains non-finite pixels."
        )

    return processed


# ================================================================================================================
# CLINICAL
# ================================================================================================================

def encode_sex(sex):

    if isinstance(sex, str):

        value = sex.strip().lower()

        if value in {"male", "m", "1"}:
            return 1.0

        if value in {"female", "f", "0"}:
            return 0.0

        raise ValueError(
            "Sex must be male/female."
        )

    value = float(sex)

    if value not in {0.0, 1.0}:
        raise ValueError(
            "Sex must be 0 or 1."
        )

    return value


def preprocess_clinical(
    age,
    sex,
    device
):

    age = float(age)

    if not np.isfinite(age):
        age = TRAIN_AGE_MEDIAN

    age_scaled = (
        age -
        TRAIN_AGE_MEAN
    ) / TRAIN_AGE_STD

    sex_encoded = encode_sex(
        sex
    )

    return torch.tensor(
        [[
            age_scaled,
            sex_encoded
        ]],
        dtype=torch.float32,
        device=device
    )


# ================================================================================================================
# MRI TENSOR
# ================================================================================================================

def processed_to_tensor(
    image,
    device
):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    if image.shape != (224, 224):
        raise ValueError(
            f"Expected 224x224, got {image.shape}"
        )

    if not np.isfinite(image).all():
        raise ValueError(
            "MRI contains non-finite pixels."
        )

    if (
        float(image.min()) < 0.0
        or
        float(image.max()) > 1.0
    ):
        raise ValueError(
            "MRI must be normalized to [0,1]."
        )

    return (
        torch.from_numpy(image)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )


# ================================================================================================================
# GRAD-CAM
# ================================================================================================================

class GradCAMCapture:

    def __init__(self, layer):

        self.activation = None
        self.gradient = None

        self.forward_handle = (
            layer.register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            layer.register_full_backward_hook(
                self._backward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output
    ):
        self.activation = output

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):
        self.gradient = grad_output[0]

    def compute(self):

        weights = self.gradient.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights *
            self.activation
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bicubic",
            align_corners=False
        )

        return (
            cam[0, 0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

    def close(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


def improve_cam_quality(
    raw_cam
):

    cam = gaussian_filter(
        raw_cam.astype(np.float32),
        sigma=2.0
    )

    low = np.percentile(
        cam,
        55
    )

    high = np.percentile(
        cam,
        99.5
    )

    if high <= low:
        return np.zeros_like(
            cam,
            dtype=np.float32
        )

    cam = (
        cam - low
    ) / (
        high - low
    )

    cam = np.clip(
        cam,
        0,
        1
    )

    cam[
        cam < 0.12
    ] = 0.0

    if cam.max() > 1e-8:
        cam = cam / cam.max()

    return cam.astype(
        np.float32
    )


# ================================================================================================================
# SHAPLEY
# ================================================================================================================

@torch.no_grad()
def _forward_logits(
    model,
    axial,
    coronal,
    sagittal,
    clinical
):

    device_type = (
        "cuda"
        if axial.device.type == "cuda"
        else "cpu"
    )

    if device_type == "cuda":

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            return model(
                axial,
                coronal,
                sagittal,
                clinical
            ).float()

    return model(
        axial,
        coronal,
        sagittal,
        clinical
    ).float()


def exact_clinical_shapley(
    model,
    axial,
    coronal,
    sagittal,
    clinical,
    predicted_label
):

    device = clinical.device

    real_age = float(
        clinical[0, 0].item()
    )

    real_sex = float(
        clinical[0, 1].item()
    )

    def clinical_tensor(
        age,
        sex
    ):

        return torch.tensor(
            [[age, sex]],
            dtype=torch.float32,
            device=device
        )

    c00 = clinical_tensor(
        BASELINE_AGE_SCALED,
        BASELINE_SEX
    )

    c10 = clinical_tensor(
        real_age,
        BASELINE_SEX
    )

    c01 = clinical_tensor(
        BASELINE_AGE_SCALED,
        real_sex
    )

    c11 = clinical

    f00 = _forward_logits(
        model,
        axial,
        coronal,
        sagittal,
        c00
    )

    f10 = _forward_logits(
        model,
        axial,
        coronal,
        sagittal,
        c10
    )

    f01 = _forward_logits(
        model,
        axial,
        coronal,
        sagittal,
        c01
    )

    f11 = _forward_logits(
        model,
        axial,
        coronal,
        sagittal,
        c11
    )

    age_shap = 0.5 * (
        (
            f10[0, predicted_label]
            -
            f00[0, predicted_label]
        )
        +
        (
            f11[0, predicted_label]
            -
            f01[0, predicted_label]
        )
    )

    sex_shap = 0.5 * (
        (
            f01[0, predicted_label]
            -
            f00[0, predicted_label]
        )
        +
        (
            f11[0, predicted_label]
            -
            f10[0, predicted_label]
        )
    )

    return {
        "age_shap_logit":
            float(age_shap.item()),

        "sex_shap_logit":
            float(sex_shap.item()),
    }


# ================================================================================================================
# FINAL INFERENCE
# ================================================================================================================

def predict_patient(
    model,
    device,
    axial_image,
    coronal_image,
    sagittal_image,
    age,
    sex,
    generate_gradcam=True,
    generate_shapley=True
):

    axial = processed_to_tensor(
        axial_image,
        device
    )

    coronal = processed_to_tensor(
        coronal_image,
        device
    )

    sagittal = processed_to_tensor(
        sagittal_image,
        device
    )

    clinical = preprocess_clinical(
        age,
        sex,
        device
    )


    cam_engines = None

    if generate_gradcam:

        cam_engines = {
            "axial":
                GradCAMCapture(
                    model.core.backbone
                    .axial_branch.features[-1]
                ),

            "coronal":
                GradCAMCapture(
                    model.core.backbone
                    .coronal_branch.features[-1]
                ),

            "sagittal":
                GradCAMCapture(
                    model.core.backbone
                    .sagittal_branch.features[-1]
                ),
        }


    model.zero_grad(
        set_to_none=True
    )

    logits = model(
        axial,
        coronal,
        sagittal,
        clinical
    )

    probabilities = torch.softmax(
        logits.float(),
        dim=1
    )

    predicted_label = int(
        probabilities.argmax(
            dim=1
        ).item()
    )

    probs_np = (
        probabilities[0]
        .detach()
        .cpu()
        .numpy()
    )

    result = {
        "predicted_label":
            predicted_label,

        "predicted_class":
            CLASS_NAMES[
                predicted_label
            ],

        "confidence":
            float(
                probs_np[
                    predicted_label
                ]
            ),

        "probabilities":
            {
                CLASS_NAMES[i]:
                    float(probs_np[i])
                for i in range(4)
            },
    }


    if generate_gradcam:

        score = logits[
            0,
            predicted_label
        ]

        model.zero_grad(
            set_to_none=True
        )

        score.backward()

        result["gradcam"] = {
            view:
                improve_cam_quality(
                    engine.compute()
                )
            for view, engine in cam_engines.items()
        }

        for engine in cam_engines.values():
            engine.close()


    if generate_shapley:

        result["clinical_shapley"] = (
            exact_clinical_shapley(
                model,
                axial,
                coronal,
                sagittal,
                clinical,
                predicted_label
            )
        )


    return result




