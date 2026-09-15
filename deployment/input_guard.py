
import re
from pathlib import Path

import numpy as np
from PIL import Image


ORIENTATION_WORDS = {
    "axial": {
        "axial",
        "axi",
        "ax"
    },
    "coronal": {
        "coronal",
        "cor",
        "coron"
    },
    "sagittal": {
        "sagittal",
        "sag",
        "sagi"
    },
}


def inspect_raw_image(uploaded_file):
    """
    Basic image-quality inspection.

    Returns metadata only.
    Does not modify the image.
    """

    uploaded_file.seek(0)

    image = Image.open(uploaded_file).convert("F")

    arr = np.asarray(
        image,
        dtype=np.float32
    )

    uploaded_file.seek(0)

    if arr.ndim != 2:
        raise ValueError(
            "Expected a 2D MRI image."
        )

    finite = np.isfinite(arr)

    if not finite.any():
        raise ValueError(
            "Image contains no valid pixels."
        )

    values = arr[finite]

    p01 = float(np.percentile(values, 1))
    p99 = float(np.percentile(values, 99))

    dynamic_range = p99 - p01

    std = float(np.std(values))

    nonzero_fraction = float(
        np.mean(
            np.abs(arr) > 1e-8
        )
    )

    return {
        "width":
            int(arr.shape[1]),

        "height":
            int(arr.shape[0]),

        "std":
            std,

        "dynamic_range":
            dynamic_range,

        "nonzero_fraction":
            nonzero_fraction,
    }


def normalized_signature(image):
    """
    Normalize only for duplicate-image comparison.
    Not model preprocessing.
    """

    arr = np.asarray(
        image,
        dtype=np.float32
    )

    if arr.shape != (224, 224):
        raise ValueError(
            f"Expected processed 224x224 image, got {arr.shape}"
        )

    flat = arr.reshape(-1)

    mean = float(flat.mean())
    std = float(flat.std())

    if std < 1e-8:
        return np.zeros_like(flat)

    return (
        (flat - mean) / std
    )


def pair_similarity(a, b):

    sa = normalized_signature(a)
    sb = normalized_signature(b)

    corr = float(
        np.mean(
            sa * sb
        )
    )

    mae = float(
        np.mean(
            np.abs(
                np.asarray(a, dtype=np.float32)
                -
                np.asarray(b, dtype=np.float32)
            )
        )
    )

    return {
        "correlation":
            corr,

        "mae":
            mae,
    }


def detect_duplicate_views(
    axial,
    coronal,
    sagittal
):

    pairs = {
        "Axial vs Coronal":
            pair_similarity(
                axial,
                coronal
            ),

        "Axial vs Sagittal":
            pair_similarity(
                axial,
                sagittal
            ),

        "Coronal vs Sagittal":
            pair_similarity(
                coronal,
                sagittal
            ),
    }

    duplicates = []

    for name, metrics in pairs.items():

        if (
            metrics["correlation"] >= 0.995
            and
            metrics["mae"] <= 0.015
        ):
            duplicates.append(
                name
            )

    return duplicates, pairs


def orientation_from_filename(filename):

    stem = Path(
        filename
    ).stem.lower()

    tokens = set(
        re.findall(
            r"[a-z]+",
            stem
        )
    )

    matches = []

    for orientation, aliases in ORIENTATION_WORDS.items():

        if tokens.intersection(
            aliases
        ):
            matches.append(
                orientation
            )

    if len(matches) == 1:
        return matches[0]

    return None


def check_orientation_filename(
    filename,
    expected_orientation
):

    detected = orientation_from_filename(
        filename
    )

    if detected is None:

        return {
            "status":
                "unknown",

            "detected":
                None,

            "message":
                "Orientation could not be verified from filename."
        }

    if detected != expected_orientation:

        return {
            "status":
                "mismatch",

            "detected":
                detected,

            "message":
                (
                    f"File appears to be {detected}, "
                    f"but was uploaded to {expected_orientation}."
                )
        }

    return {
        "status":
            "pass",

        "detected":
            detected,

        "message":
            "Filename orientation matches upload slot."
    }


def validate_age(age):

    age = float(age)

    if not np.isfinite(age):

        return {
            "block":
                True,

            "warning":
                None,

            "message":
                "Age is invalid."
        }

    if age < 1 or age > 110:

        return {
            "block":
                True,

            "warning":
                None,

            "message":
                "Age must be between 1 and 110 years."
        }

    if age < 18 or age > 90:

        return {
            "block":
                False,

            "warning":
                True,

            "message":
                (
                    "Age is outside the common range represented "
                    "by much of the training cohort. Verify it carefully."
                )
        }

    return {
        "block":
            False,

        "warning":
            False,

        "message":
            "Age accepted."
    }


def severe_image_problem(info):

    if info["dynamic_range"] <= 1e-6:
        return "Image appears blank or constant."

    if info["std"] <= 1e-6:
        return "Image has almost no intensity variation."

    if info["nonzero_fraction"] < 0.01:
        return "Image contains almost no visible MRI content."

    return None


def image_quality_warning(info):

    if info["nonzero_fraction"] < 0.05:
        return "Very little non-zero image content detected."

    return None