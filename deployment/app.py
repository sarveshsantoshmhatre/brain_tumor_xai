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
    sys.path.insert(0, str(ROOT))

from deployment.final_deployment import load_final_model, preprocess_raw_upload, predict_patient, CLASS_NAMES
from deployment.input_guard import inspect_raw_image, detect_duplicate_views, check_orientation_filename, validate_age, severe_image_problem, image_quality_warning
from deployment.report_generator import build_patient_report

st.set_page_config(page_title="Brain Tumor XAI", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""<style>
.block-container { max-width: 1250px; padding-top: 2rem; padding-bottom: 3rem; }
.hero-title { font-size: 2.55rem; font-weight: 760; letter-spacing: -0.04rem; margin-bottom: 0.15rem; }
.hero-subtitle { opacity: 0.68; margin-bottom: 2rem; }
.section { font-size: 1.35rem; font-weight: 720; margin-top: 1.5rem; margin-bottom: 0.8rem; }
.result-card { border: 1px solid rgba(128,128,128,.22); border-radius: 18px; padding: 22px; background: rgba(128,128,128,.045); }
.result-label { opacity: .64; font-size: .88rem; }
.result-value { font-size: 2.0rem; font-weight: 760; margin-top: .25rem; }
.guard-pass { border-radius: 12px; padding: 10px 14px; background: rgba(50,180,100,.10); border: 1px solid rgba(50,180,100,.22); }
.guard-info { border-radius: 12px; padding: 10px 14px; background: rgba(60,130,220,.09); border: 1px solid rgba(60,130,220,.20); }
.footer { text-align:center; opacity:.50; font-size:.78rem; padding-top:1rem; }
.stButton > button { border-radius: 12px; min-height: 48px; font-weight: 650; }
[data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.18); padding: 15px; border-radius: 14px; }
[data-testid="stImage"] img { border: none !important; outline: none !important; box-shadow: none !important; }
</style>""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">Brain Tumor XAI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Multi-angle MRI classification with image and clinical explanations</div>', unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_final_model()

with st.spinner("Loading AI model..."):
    model, checkpoint, device = get_model()

if checkpoint.get("status") == "demo_untrained":
    st.warning("Demo mode: the repository does not contain the trained best.pt checkpoint. The interface and preprocessing/XAI pipeline are runnable, but predictions from the fallback model are NOT medically meaningful. Add the trained checkpoint to enable real inference.")
else:
    st.success("Trained checkpoint loaded. Inference is using the supplied model weights.")


def preprocess_uploaded(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(uploaded_file.getbuffer())
        temp_path = temp.name
    try:
        return preprocess_raw_upload(temp_path)
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


def largest_component(mask):
    labeled, count = label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(sizes.argmax())


def anatomy_alpha_mask(image):
    image = np.asarray(image, dtype=np.float32)
    mask = gaussian_filter((image > 1e-5).astype(np.float32), sigma=1.8)
    if mask.max() > 0:
        mask /= mask.max()
    return np.clip(mask, 0.0, 1.0)


def make_gradcam_overlay(image, raw_cam):
    image = np.asarray(image, dtype=np.float32)
    cam = np.clip(np.asarray(raw_cam, dtype=np.float32), 0.0, 1.0)
    anatomy = anatomy_alpha_mask(image)
    visible_cam = np.ma.masked_where(anatomy < 0.03, cam)
    alpha_map = 0.55 * anatomy
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=180)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
    ax.imshow(visible_cam, cmap="jet", vmin=0, vmax=1, alpha=alpha_map, interpolation="lanczos")
    contour_cam = np.ma.masked_where(anatomy < 0.08, cam)
    if np.any((cam >= 0.60) & (anatomy >= 0.08)):
        try:
            ax.contour(contour_cam, levels=[0.60], linewidths=1.15)
        except Exception:
            pass
    ax.set_xlim(0, 223); ax.set_ylim(223, 0); ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def figure_to_borderless_png(fig):
    output = BytesIO()
    fig.savefig(output, format="png", dpi=180, facecolor="black", edgecolor="none", bbox_inches=None, pad_inches=0)
    output.seek(0)
    return output.getvalue()


def make_mri_figure(image):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="bicubic")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def make_gradcam_heatmap(raw_cam):
    cam = np.clip(np.asarray(raw_cam, dtype=np.float32), 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    heatmap = ax.imshow(cam, cmap="jet", vmin=0, vmax=1, interpolation="bicubic")
    ax.axis("off")
    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    fig.subplots_adjust(left=0, right=0.9, top=1, bottom=0)
    return fig


def gradcam_summary(image, raw_cam):
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
    return {"location": f"{horizontal}-{vertical}", "high": float(np.mean(values >= 0.70) * 100.0), "moderate": float(np.mean((values >= 0.40) & (values < 0.70)) * 100.0), "low": float(np.mean((values > 0.0) & (values < 0.40)) * 100.0)}


def shapley_explanation(feature_name, value, predicted_name):
    magnitude = abs(float(value))
    strength = "very small" if magnitude <= 0.01 else "small" if magnitude < 0.05 else "noticeable"
    if value > 0.01:
        return f"{feature_name} supported the {predicted_name} prediction. Its positive influence increased model support relative to the clinical baseline. Its magnitude was {strength}."
    if value < -0.01:
        return f"{feature_name} opposed the {predicted_name} prediction. Its negative influence reduced model support relative to the clinical baseline. Its magnitude was {strength}."
    return f"{feature_name} had a very small influence on the {predicted_name} prediction; its Shapley value was close to zero."


def make_clinical_influence_chart(age_value, sex_value):
    labels = ["Age", "Sex"]
    values = np.array([age_value, sex_value], dtype=np.float32)
    max_abs = max(float(np.max(np.abs(values))), 0.01)
    fig, ax = plt.subplots(figsize=(8.5, 3.2), dpi=160)
    y = np.arange(len(labels))
    bar_colors = ["#2E8B57" if value > 0.01 else "#C94C4C" if value < -0.01 else "#7A8A99" for value in values]
    bars = ax.barh(y, values, color=bar_colors, height=0.48)
    ax.axvline(0, linewidth=1.2); ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlim(-max_abs * 1.35, max_abs * 1.35); ax.set_xlabel("Influence on predicted result"); ax.set_title("Clinical Feature Influence"); ax.grid(axis="x", alpha=0.15)
    for bar, value in zip(bars, values):
        x = float(value); large = abs(x) >= max_abs * 0.28
        label_x = x / 2 if large else (max_abs * 0.10 if x >= 0 else -max_abs * 0.10)
        ax.text(label_x, bar.get_y() + bar.get_height() / 2, f"{value:+.4f}", va="center", ha="center" if large else ("left" if x >= 0 else "right"), fontsize=10, fontweight="semibold", color="white" if large else "#1C2630")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(0.5, -0.27, "Green = supports prediction | Red = opposes prediction | Gray = minimal influence", transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#596773")
    fig.tight_layout(pad=1.4)
    return fig


def influence_label(value):
    if value > 0.01:
        return "Supports the prediction"
    if value < -0.01:
        return "Opposes the prediction"
    return "Minimal influence"

# The remainder of the original application is intentionally kept simple:
# three MRI uploads, clinical inputs, inference, Grad-CAM, SHAP values and report download.

st.markdown('<div class="section">MRI inputs</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    axial_file = st.file_uploader("Axial view", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="axial")
with col2:
    coronal_file = st.file_uploader("Coronal view", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="coronal")
with col3:
    sagittal_file = st.file_uploader("Sagittal view", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="sagittal")

age = st.number_input("Age", min_value=0.0, max_value=120.0, value=59.0, step=1.0)
sex = st.selectbox("Sex", ["Female", "Male"])

if st.button("Run analysis", type="primary", use_container_width=True):
    if not all([axial_file, coronal_file, sagittal_file]):
        st.error("Upload all three MRI views before running analysis.")
        st.stop()
    try:
        images = [preprocess_uploaded(f) for f in (axial_file, coronal_file, sagittal_file)]
        result = predict_patient(model, device, images[0], images[1], images[2], age, sex, generate_gradcam=True, generate_shapley=True)
        st.session_state["analysis_result"] = result
        st.session_state["analysis_images"] = images
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

result = st.session_state.get("analysis_result")
images = st.session_state.get("analysis_images")
if result is not None and images is not None:
    st.markdown('<div class="section">Result</div>', unsafe_allow_html=True)
    st.metric("Predicted class", result["predicted_class"])
    st.metric("Model confidence", f"{result['confidence'] * 100:.2f}%")
    st.subheader("Class probabilities")
    st.bar_chart(result["probabilities"])
    st.subheader("MRI views")
    cols = st.columns(3)
    for col, name, image in zip(cols, ["Axial", "Coronal", "Sagittal"], images):
        with col:
            st.caption(name)
            st.pyplot(make_mri_figure(image), clear_figure=True)
    st.subheader("Grad-CAM explanations")
    for name, image in zip(["Axial", "Coronal", "Sagittal"], images):
        st.caption(name)
        st.pyplot(make_gradcam_overlay(image, result["gradcam"][name.lower()]), clear_figure=True)
    if "shapley" in result:
        age_shap = result["shapley"]["age_shap_logit"]
        sex_shap = result["shapley"]["sex_shap_logit"]
        st.subheader("Clinical explanation")
        st.pyplot(make_clinical_influence_chart(age_shap, sex_shap), clear_figure=True)
        st.write(shapley_explanation("Age", age_shap, result["predicted_class"]))
        st.write(shapley_explanation("Sex", sex_shap, result["predicted_class"]))

st.markdown('<div class="footer">Research/demo software only. Not a medical diagnosis.</div>', unsafe_allow_html=True)
