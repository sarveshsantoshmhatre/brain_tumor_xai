from pathlib import Path
import sys
from io import BytesIO
import tempfile
import os

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    gaussian_filter,
    label
)


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT)
    )


from deployment.final_deployment import (
    load_final_model,
    preprocess_raw_upload,
    predict_patient,
    CLASS_NAMES,
)

from deployment.input_guard import (
    inspect_raw_image,
    detect_duplicate_views,
    check_orientation_filename,
    validate_age,
    severe_image_problem,
    image_quality_warning,
)

from deployment.report_generator import build_patient_report


# ================================================================================================================
# PAGE
# ================================================================================================================

st.set_page_config(
    page_title="Brain Tumor XAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown(
    """
<style>

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero-title {
    font-size: 2.55rem;
    font-weight: 760;
    letter-spacing: -0.04rem;
    margin-bottom: 0.15rem;
}

.hero-subtitle {
    opacity: 0.68;
    margin-bottom: 2rem;
}

.section {
    font-size: 1.35rem;
    font-weight: 720;
    margin-top: 1.5rem;
    margin-bottom: 0.8rem;
}

.result-card {
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 18px;
    padding: 22px;
    background: rgba(128,128,128,.045);
}

.result-label {
    opacity: .64;
    font-size: .88rem;
}

.result-value {
    font-size: 2.0rem;
    font-weight: 760;
    margin-top: .25rem;
}

.guard-pass {
    border-radius: 12px;
    padding: 10px 14px;
    background: rgba(50,180,100,.10);
    border: 1px solid rgba(50,180,100,.22);
}

.guard-info {
    border-radius: 12px;
    padding: 10px 14px;
    background: rgba(60,130,220,.09);
    border: 1px solid rgba(60,130,220,.20);
}

.footer {
    text-align:center;
    opacity:.50;
    font-size:.78rem;
    padding-top:1rem;
}

.stButton > button {
    border-radius: 12px;
    min-height: 48px;
    font-weight: 650;
}

[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.18);
    padding: 15px;
    border-radius: 14px;
}

[data-testid="stImage"] img {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

</style>
""",
    unsafe_allow_html=True
)


st.markdown(
    '<div class="hero-title">🧠 Brain Tumor XAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'Multi-angle MRI classification with image and clinical explanations'
    '</div>',
    unsafe_allow_html=True
)


# ================================================================================================================
# MODEL
# ================================================================================================================

@st.cache_resource
def get_model():

    return load_final_model()


with st.spinner(
    "Loading AI model..."
):

    model, checkpoint, device = get_model()


# ================================================================================================================
# HELPERS
# ================================================================================================================

def preprocess_uploaded(
    uploaded_file
):

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:

        temp.write(
            uploaded_file.getbuffer()
        )

        temp_path = temp.name

    try:

        processed = preprocess_raw_upload(
            temp_path
        )

    finally:

        try:
            os.remove(
                temp_path
            )

        except Exception:
            pass

    return processed


def largest_component(mask):

    labeled, count = label(
        mask
    )

    if count == 0:
        return mask

    sizes = np.bincount(
        labeled.ravel()
    )

    sizes[0] = 0

    largest = int(
        sizes.argmax()
    )

    return (
        labeled == largest
    )



def anatomy_alpha_mask(image):
    """
    DISPLAY ONLY.

    Keeps the previous smooth Grad-CAM appearance,
    while preventing heatmap from appearing over pure black padding.

    Raw Grad-CAM values and prediction are NOT changed.
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    # Anything with actual MRI intensity is considered image support.
    mask = (
        image > 1e-5
    ).astype(
        np.float32
    )

    # Soft edge so heatmap does not look sharply clipped.
    mask = gaussian_filter(
        mask,
        sigma=1.8
    )

    if mask.max() > 0:

        mask = (
            mask /
            mask.max()
        )

    return np.clip(
        mask,
        0.0,
        1.0
    )


def make_gradcam_overlay(
    image,
    raw_cam
):
    """
    Previous-style smooth Grad-CAM visualization.

    IMPORTANT:
    - raw CAM location unchanged
    - no CAM renormalization after anatomy mask
    - no tumor-region forcing
    - only black padding is made transparent
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    cam = np.asarray(
        raw_cam,
        dtype=np.float32
    )

    cam = np.clip(
        cam,
        0.0,
        1.0
    )

    anatomy = anatomy_alpha_mask(
        image
    )


    # Previous visual style:
    # broad smooth heatmap remains visible.
    visible_cam = np.ma.masked_where(
        anatomy < 0.03,
        cam
    )


    # Alpha depends mostly on anatomy,
    # NOT on CAM strength.
    #
    # This restores the full blue-green-yellow-red map
    # instead of showing only tiny red patches.
    alpha_map = (
        0.55
        *
        anatomy
    )


    fig, ax = plt.subplots(
        figsize=(
            6.4,
            6.4
        ),
        dpi=180
    )

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")


    # MRI base
    ax.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="lanczos"
    )


    # Smooth full Grad-CAM
    ax.imshow(
        visible_cam,
        cmap="jet",
        vmin=0,
        vmax=1,
        alpha=alpha_map,
        interpolation="lanczos"
    )


    # Strong-attention contour
    contour_cam = np.ma.masked_where(
        anatomy < 0.08,
        cam
    )

    if np.any(
        (
            cam >= 0.60
        )
        &
        (
            anatomy >= 0.08
        )
    ):

        try:

            ax.contour(
                contour_cam,
                levels=[
                    0.60
                ],
                linewidths=1.15
            )

        except Exception:
            pass


    ax.set_xlim(
        0,
        223
    )

    ax.set_ylim(
        223,
        0
    )

    ax.axis(
        "off"
    )

    fig.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0
    )

    return fig


def figure_to_borderless_png(fig):
    """Convert a Grad-CAM figure to a sharp PNG without a surrounding frame."""
    output = BytesIO()
    fig.savefig(
        output,
        format="png",
        dpi=180,
        facecolor="black",
        edgecolor="none",
        bbox_inches=None,
        pad_inches=0,
    )
    output.seek(0)
    return output.getvalue()


def make_mri_figure(image):
    """Render a processed MRI slice without altering inference data."""
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="bicubic")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def make_gradcam_heatmap(raw_cam):
    """Render the normalized Grad-CAM values as a standalone heatmap."""
    cam = np.clip(np.asarray(raw_cam, dtype=np.float32), 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    heatmap = ax.imshow(cam, cmap="jet", vmin=0, vmax=1, interpolation="bicubic")
    ax.axis("off")
    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    fig.subplots_adjust(left=0, right=0.9, top=1, bottom=0)
    return fig


def gradcam_summary(image, raw_cam):
    """Calculate display-space CAM statistics inside the visible MRI support."""
    cam = np.clip(np.asarray(raw_cam, dtype=np.float32), 0.0, 1.0)
    support = anatomy_alpha_mask(image) >= 0.08
    if not np.any(support):
        support = np.ones_like(cam, dtype=bool)

    supported_cam = np.where(support, cam, -1.0)
    y, x = np.unravel_index(np.argmax(supported_cam), supported_cam.shape)
    height, width = cam.shape

    horizontal = "left" if x < width / 3 else "right" if x >= 2 * width / 3 else "center"
    vertical = "upper" if y < height / 3 else "lower" if y >= 2 * height / 3 else "middle"
    values = cam[support]

    return {
        "location": f"{horizontal}-{vertical}",
        "high": float(np.mean(values >= 0.70) * 100.0),
        "moderate": float(np.mean((values >= 0.40) & (values < 0.70)) * 100.0),
        "low": float(np.mean((values > 0.0) & (values < 0.40)) * 100.0),
    }


def shapley_explanation(feature_name, value, predicted_name):
    magnitude = abs(float(value))
    strength = "very small" if magnitude <= 0.01 else "small" if magnitude < 0.05 else "noticeable"

    if value > 0.01:
        return (
            f"{feature_name} supported the {predicted_name} prediction. The positive influence "
            f"means this feature increased the model's support for {predicted_name} relative "
            f"to the clinical baseline. Its magnitude was {strength}."
        )
    if value < -0.01:
        return (
            f"{feature_name} opposed the {predicted_name} prediction. The negative influence "
            f"means this feature reduced the model's support for {predicted_name} relative "
            f"to the clinical baseline. Its magnitude was {strength}."
        )
    return (
        f"{feature_name} had a very small influence on the {predicted_name} prediction; "
        "its Shapley value was close to zero."
    )


# ================================================================================================================
# HUMAN-FRIENDLY CLINICAL INFLUENCE GRAPH
# ================================================================================================================

def make_clinical_influence_chart(
    age_value,
    sex_value
):

    """
    Visualization only.

    Positive value:
        supports predicted class

    Negative value:
        opposes predicted class

    Values are model-score contributions,
    NOT percentages.
    """

    labels = [
        "Age",
        "Sex"
    ]

    values = np.array(
        [
            age_value,
            sex_value
        ],
        dtype=np.float32
    )


    max_abs = max(
        float(
            np.max(
                np.abs(
                    values
                )
            )
        ),
        0.01
    )


    fig, ax = plt.subplots(
        figsize=(8.5, 3.2),
        dpi=160
    )


    y = np.arange(
        len(
            labels
        )
    )


    bar_colors = [
        "#2E8B57" if value > 0.01 else "#C94C4C" if value < -0.01 else "#7A8A99"
        for value in values
    ]

    bars = ax.barh(
        y,
        values,
        color=bar_colors,
        height=0.48
    )


    ax.axvline(
        0,
        linewidth=1.2
    )


    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        labels
    )


    ax.set_xlim(
        -max_abs * 1.35,
        max_abs * 1.35
    )


    ax.set_xlabel(
        "Influence on predicted result"
    )


    ax.set_title(
        "Clinical Feature Influence"
    )


    ax.grid(
        axis="x",
        alpha=0.15
    )


    for bar, value in zip(
        bars,
        values
    ):

        x = float(value)
        is_large_enough = abs(x) >= max_abs * 0.28

        if is_large_enough:
            label_x = x / 2
            label_color = "white"
            horizontal_alignment = "center"
        else:
            label_x = (
                max_abs * 0.10
                if x >= 0
                else -max_abs * 0.10
            )
            label_color = "#1C2630"
            horizontal_alignment = "left" if x >= 0 else "right"

        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.4f}",
            va="center",
            ha=horizontal_alignment,
            fontsize=10,
            fontweight="semibold",
            color=label_color
        )


    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )


    ax.text(
        0.5,
        -0.27,
        "Green = supports prediction    |    Red = opposes prediction    |    Gray = minimal influence",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color="#596773"
    )

    fig.tight_layout(pad=1.4)

    return fig


def influence_label(
    value
):

    if value > 0.01:

        return (
            "Supports the prediction"
        )

    if value < -0.01:

        return (
            "Opposes the prediction"
        )

    return (
        "Very small / nearly neutral influence"
    )




# ================================================================================================================
# INPUTS
# ================================================================================================================

st.markdown(
    '<div class="section">MRI Views</div>',
    unsafe_allow_html=True
)


u1, u2, u3 = st.columns(
    3,
    gap="medium"
)


with u1:

    axial_file = st.file_uploader(
        "Axial MRI",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "tif",
            "tiff"
        ],
        key="axial"
    )


with u2:

    coronal_file = st.file_uploader(
        "Coronal MRI",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "tif",
            "tiff"
        ],
        key="coronal"
    )


with u3:

    sagittal_file = st.file_uploader(
        "Sagittal MRI",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "tif",
            "tiff"
        ],
        key="sagittal"
    )


st.markdown(
    '<div class="section">Clinical Information</div>',
    unsafe_allow_html=True
)


c1, c2 = st.columns(
    2
)


with c1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=110,
        value=59,
        step=1
    )


with c2:

    sex = st.selectbox(
        "Sex",
        [
            "female",
            "male"
        ]
    )


# ================================================================================================================
# ORIGINAL MRI PREVIEW
# ================================================================================================================

all_files = all(
    file is not None
    for file in [
        axial_file,
        coronal_file,
        sagittal_file
    ]
)


if all_files:

    st.markdown(
        '<div class="section">Uploaded MRI Preview</div>',
        unsafe_allow_html=True
    )


    st.caption(
        "Verify the original uploaded images before analysis."
    )


    preview1, preview2, preview3 = st.columns(
        3,
        gap="medium"
    )


    with preview1:

        st.image(
            axial_file,
            caption="Axial MRI",
            width="stretch"
        )


    with preview2:

        st.image(
            coronal_file,
            caption="Coronal MRI",
            width="stretch"
        )


    with preview3:

        st.image(
            sagittal_file,
            caption="Sagittal MRI",
            width="stretch"
        )




# ================================================================================================================
# CONFIRMATION
# ================================================================================================================

st.markdown(
    '<div class="section">Input Verification</div>',
    unsafe_allow_html=True
)


same_patient_confirm = st.checkbox(
    "I confirm that axial, coronal and sagittal MRI views belong to the same patient and study."
)


orientation_confirm = st.checkbox(
    "I confirm that each MRI has been uploaded into the correct Axial / Coronal / Sagittal slot."
)


clinical_confirm = st.checkbox(
    "I confirm that Age and Sex belong to the same patient as the uploaded MRI views."
)


# ================================================================================================================
# PRE-VALIDATION
# ================================================================================================================

files = {
    "axial":
        axial_file,

    "coronal":
        coronal_file,

    "sagittal":
        sagittal_file,
}


all_files = all(
    file is not None
    for file in files.values()
)


hard_errors = []
warnings = []

processed_views = None


if all_files:

    raw_info = {}


    for view_name, file in files.items():

        try:

            info = inspect_raw_image(
                file
            )

            raw_info[
                view_name
            ] = info


            severe = severe_image_problem(
                info
            )

            if severe:

                hard_errors.append(
                    f"{view_name.title()}: {severe}"
                )


            warning = image_quality_warning(
                info
            )

            if warning:

                warnings.append(
                    f"{view_name.title()}: {warning}"
                )


        except Exception as e:

            hard_errors.append(
                f"{view_name.title()} image could not be read: {e}"
            )


        orientation = check_orientation_filename(
            file.name,
            view_name
        )


        if orientation[
            "status"
        ] == "mismatch":

            hard_errors.append(
                orientation[
                    "message"
                ]
            )


    if not hard_errors:

        try:

            processed_views = {
                view_name:
                    preprocess_uploaded(
                        file
                    )

                for view_name, file in files.items()
            }


            duplicates, duplicate_metrics = (
                detect_duplicate_views(
                    processed_views[
                        "axial"
                    ],
                    processed_views[
                        "coronal"
                    ],
                    processed_views[
                        "sagittal"
                    ]
                )
            )


            if duplicates:

                hard_errors.append(
                    "Near-duplicate MRI views detected: "
                    +
                    ", ".join(
                        duplicates
                    )
                    +
                    ". Each slot should contain its correct anatomical view."
                )


        except Exception as e:

            hard_errors.append(
                f"MRI preprocessing failed: {e}"
            )


age_check = validate_age(
    age
)


if age_check[
    "block"
]:

    hard_errors.append(
        age_check[
            "message"
        ]
    )


elif age_check[
    "warning"
]:

    warnings.append(
        age_check[
            "message"
        ]
    )


# ================================================================================================================
# INPUT STATUS
# ================================================================================================================

if all_files:

    if hard_errors:

        for message in hard_errors:

            st.error(
                message
            )


    else:

        st.markdown(
            """
<div class="guard-pass">
✅ Uploaded files passed the automatic technical checks.
</div>
""",
            unsafe_allow_html=True
        )


        st.caption(
            "Automatic checks cannot prove that three ordinary PNG/JPG images belong to the same patient. "
            "That is why patient/study confirmation is required."
        )


        if warnings:

            for message in warnings:

                st.warning(
                    message
                )


# ================================================================================================================
# READY STATE
# ================================================================================================================

ready = (
    all_files
    and
    not hard_errors
    and
    same_patient_confirm
    and
    orientation_confirm
    and
    clinical_confirm
)


if all_files and not ready and not hard_errors:

    st.info(
        "Complete the three verification confirmations to enable analysis."
    )


# ================================================================================================================
# ANALYZE BUTTON
# ================================================================================================================

analyze = st.button(
    "🔬 Analyze MRI",
    type="primary",
    width="stretch",
    disabled=not ready
)


if analyze:

    try:

        with st.status(
            "Analyzing patient...",
            expanded=True
        ) as status:


            st.write(
                "Input validation passed."
            )

            st.write(
                "Running final classification model..."
            )


            result = predict_patient(
                model=model,
                device=device,
                axial_image=processed_views[
                    "axial"
                ],
                coronal_image=processed_views[
                    "coronal"
                ],
                sagittal_image=processed_views[
                    "sagittal"
                ],
                age=float(age),
                sex=sex,
                generate_gradcam=True,
                generate_shapley=True
            )


            st.write(
                "Generating explanations..."
            )


            status.update(
                label="Analysis complete",
                state="complete",
                expanded=False
            )


        predicted_name = result["predicted_class"].title()
        confidence_percent = float(result["confidence"]) * 100.0

        # 1. PATIENT INPUT SUMMARY
        st.markdown(
            '<div class="section">Patient Input Summary</div>',
            unsafe_allow_html=True
        )
        summary_cols = st.columns(3)
        summary_cols[0].metric("Age", f"{float(age):.0f} years")
        summary_cols[1].metric("Sex", str(sex).title())
        summary_cols[2].metric("MRI views", "3 validated views")
        st.caption("Inputs used: axial, coronal and sagittal MRI views together with Age and Sex.")

        # 2. AI PREDICTION
        st.markdown(
            '<div class="section">AI Prediction</div>',
            unsafe_allow_html=True
        )


        r1, r2 = st.columns(
            2
        )


        with r1:

            st.markdown(
                f"""
<div class="result-card">
<div class="result-label">Predicted Tumor Type</div>
<div class="result-value">{predicted_name}</div>
</div>
""",
                unsafe_allow_html=True
            )


        with r2:

            st.markdown(
                f"""
<div class="result-card">
<div class="result-label">Prediction Confidence</div>
<div class="result-value">{result["confidence"] * 100:.2f}%</div>
</div>
""",
                unsafe_allow_html=True
            )


        # 3. CLASS PROBABILITIES
        st.markdown(
            '<div class="section">Class Probabilities</div>',
            unsafe_allow_html=True
        )


        prob_cols = st.columns(
            4
        )


        for index, class_name in enumerate(
            CLASS_NAMES
        ):

            probability = float(
                result[
                    "probabilities"
                ][
                    class_name
                ]
            )


            with prob_cols[
                index
            ]:

                st.metric(
                    class_name.title(),
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    probability
                )


        # 4. GRAD-CAM ANALYSIS
        st.markdown(
            '<div class="section">Grad-CAM Analysis</div>',
            unsafe_allow_html=True
        )
        st.caption(
            "Grad-CAM does not show the exact tumor boundary and is not a segmentation map. "
            "It shows which regions influenced the model's classification decision."
        )

        cam_summaries = {}
        attention_cols = st.columns(3, gap="medium")
        for column, view_name in zip(
            attention_cols,
            ("axial", "coronal", "sagittal")
        ):
            image = processed_views[view_name]
            cam = result["gradcam"][view_name]
            cam_summaries[view_name] = gradcam_summary(image, cam)

            with column:
                fig = make_gradcam_overlay(image, cam)
                overlay_png = figure_to_borderless_png(fig)
                plt.close(fig)
                st.image(overlay_png, width="stretch")
                st.caption(f"{view_name.title()} attention")

        # 6. GRAD-CAM COLOR LEGEND
        st.markdown(
            '<div class="section">Grad-CAM Color Legend</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            """
<div class="guard-info">
🔴 <b>Red / dark red:</b> highest model attention; the region contributed strongly to the predicted class.<br>
🟠 <b>Orange / yellow:</b> high model attention.<br>
🟢 <b>Green:</b> moderate model attention.<br>
🔵 <b>Blue:</b> low model attention.<br>
⚫ <b>Dark / no heat:</b> very low model attention.<br><br>
The model focused strongly on red and yellow regions. These colors do not confirm tumor tissue.
</div>
""",
            unsafe_allow_html=True
        )

        # 7. PATIENT-SPECIFIC GRAD-CAM INTERPRETATION
        st.markdown(
            '<div class="section">Patient-Specific Grad-CAM Interpretation</div>',
            unsafe_allow_html=True
        )
        for view_name, summary in cam_summaries.items():
            st.markdown(
                f"**{view_name.title()} view:** The strongest AI attention is located in the "
                f"**{summary['location']} image region**. Red and yellow regions contributed most "
                f"strongly to the predicted **{predicted_name}** class; green regions contributed "
                f"moderately and blue regions had comparatively low influence. Within the visible "
                f"MRI area, **{summary['high']:.1f}%** had high activation, "
                f"**{summary['moderate']:.1f}%** moderate activation and "
                f"**{summary['low']:.1f}%** low activation. This is an image-coordinate, "
                "model-attention description, not confirmed anatomical localization."
            )

        # CLINICAL INFLUENCE
        st.markdown(
            '<div class="section">Clinical Influence</div>',
            unsafe_allow_html=True
        )
        age_value = float(result["clinical_shapley"]["age_shap_logit"])
        sex_value = float(result["clinical_shapley"]["sex_shap_logit"])
        combined_value = age_value + sex_value

        def simple_influence_status(value):
            if value > 0.01:
                return "Supports prediction"
            if value < -0.01:
                return "Opposes prediction"
            return "Very small influence"

        influence_col1, influence_col2 = st.columns(2, gap="medium")
        with influence_col1:
            st.metric("Age Influence", simple_influence_status(age_value))
            st.caption(f"Shapley contribution: {age_value:+.4f}")
        with influence_col2:
            st.metric("Sex Influence", simple_influence_status(sex_value))
            st.caption(f"Shapley contribution: {sex_value:+.4f}")

        # 10–11. FEATURE-SPECIFIC SHAPLEY EXPLANATIONS
        st.markdown("#### Age SHAP/Shapley explanation")
        st.write(shapley_explanation("Age", age_value, predicted_name))
        st.markdown("#### Sex SHAP/Shapley explanation")
        st.write(shapley_explanation("Sex", sex_value, predicted_name))
        st.caption(
            "A positive value supports the current predicted class relative to the clinical baseline; "
            "a negative value opposes it. These values explain model behavior and do not establish a diagnosis."
        )

        # 12. COMBINED CLINICAL INTERPRETATION
        st.markdown(
            '<div class="section">Combined Clinical Interpretation</div>',
            unsafe_allow_html=True
        )
        age_direction = "supported" if age_value > 0.01 else "opposed" if age_value < -0.01 else "had minimal effect on"
        sex_direction = "supported" if sex_value > 0.01 else "opposed" if sex_value < -0.01 else "had minimal effect on"
        combined_description = (
            "provided additional support" if combined_value > 0.01
            else "slightly reduced support" if combined_value < -0.01
            else "had a relatively small net effect"
        )
        st.write(
            f"Age {age_direction} the predicted {predicted_name} class, while Sex {sex_direction} it. "
            f"Together, the clinical information {combined_description}. The final prediction remained "
            "primarily driven by MRI imaging features."
        )
        fig = make_clinical_influence_chart(age_value, sex_value)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        # 13. OVERALL AI INTERPRETATION
        st.markdown(
            '<div class="section">Overall AI Interpretation</div>',
            unsafe_allow_html=True
        )
        locations = ", ".join(
            f"{name.title()}: {summary['location']}" for name, summary in cam_summaries.items()
        )
        st.write(
            f"The AI model analyzed all three MRI views together with Age and Sex. It assigned the "
            f"highest probability to **{predicted_name}** with **{confidence_percent:.2f}% confidence**. "
            f"The strongest Grad-CAM attention occurred at these image-coordinate locations: {locations}. "
            f"Age {age_direction} the predicted class and Sex {sex_direction} it. Overall, the model's "
            "decision was primarily driven by MRI features, with clinical information providing a secondary influence."
        )

        # 14. PATIENT-FRIENDLY EXPLANATION
        st.markdown(
            '<div class="section">What This Result Means</div>',
            unsafe_allow_html=True
        )
        st.write(
            f"Your three MRI views were analyzed together with your Age and Sex information. The AI model "
            f"found patterns most similar to the **{predicted_name}** class. Red and yellow areas show where "
            "the AI focused most strongly. These highlighted areas are not confirmed tumor boundaries and "
            "should be reviewed by a qualified medical specialist."
        )

        # 15. SUGGESTED NEXT STEPS
        st.markdown(
            '<div class="section">Suggested Next Steps</div>',
            unsafe_allow_html=True
        )
        st.warning("This AI output is an assistive screening/research result and not a final diagnosis.")
        st.markdown(
            """
- Show the complete original MRI study to a qualified radiologist; prefer original DICOM/NIfTI scans over screenshots.
- Compare the study with previous MRI examinations when available.
- Ask the clinician or radiologist to evaluate lesion location, size, enhancement pattern, edema and surrounding structures.
- Additional contrast imaging or follow-up should be decided only by the treating clinician.
- Histopathology, where clinically required, remains the definitive method for many tumor diagnoses.
- Do not start, stop or change treatment based only on this AI dashboard.
"""
        )

        # 16. MEDICAL DISCLAIMER
        st.markdown(
            '<div class="section">Medical Disclaimer</div>',
            unsafe_allow_html=True
        )
        st.error(
            "This system is developed for research and AI-assisted decision support. It is not a standalone "
            "diagnostic tool and does not replace evaluation by a qualified radiologist, neurologist, "
            "neurosurgeon, oncologist or other medical professional."
        )

        # 17. PRINTABLE REPORT
        st.markdown(
            '<div class="section">Printable Patient Report</div>',
            unsafe_allow_html=True
        )

        report_pdf = build_patient_report(
            age=age,
            sex=sex,
            predicted_name=predicted_name,
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            processed_views=processed_views,
            gradcam=result["gradcam"],
            cam_summaries=cam_summaries,
            age_shapley=age_value,
            sex_shapley=sex_value,
        )

        st.download_button(
            "Download / Print PDF Report",
            data=report_pdf,
            file_name="brain_tumor_ai_assessment_report.pdf",
            mime="application/pdf",
            type="primary",
            width="stretch",
        )

        st.caption(
            "Open the downloaded PDF and use your device's Print command. "
            "The report includes natural-language explanations for patients and clinicians."
        )



    except Exception as error:

        st.error(
            "Analysis could not be completed."
        )

        with st.expander(
            "Technical details"
        ):

            st.exception(
                error
            )


st.divider()

st.markdown(
    """
<div class="footer">
Brain Tumor XAI • Research prototype — not intended for clinical diagnosis
</div>
""",
    unsafe_allow_html=True
)




