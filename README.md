# Brain Tumor XAI

A research/demo Streamlit application for four-class brain-tumor MRI classification using three MRI views plus clinical features, with Grad-CAM and clinical Shapley explanations.

## Current architecture

- Axial MRI -> EfficientNet-B0 -> 1280 features
- Coronal MRI -> EfficientNet-B0 -> 1280 features
- Sagittal MRI -> EfficientNet-B0 -> 1280 features
- Three-view image fusion -> 512 features
- Clinical branch: age + sex -> 64 features
- Multimodal fusion -> 256 features
- Classifier -> 4 logits: Glioma, Meningioma, Metastasis, Pituitary

## Run locally

```bash
git clone https://github.com/sarveshsantoshmhatre/brain_tumor_xai.git
cd brain_tumor_xai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run deployment/app.py
```

Then open the local Streamlit URL shown in the terminal.

## Model checkpoint

The repository currently does **not** contain `checkpoints/attention_guided_final_a100/best.pt`. The application therefore starts in **Demo mode** with randomly initialized model weights so that the UI, preprocessing, inference pipeline, Grad-CAM pipeline and clinical explanation pipeline can be tested.

Demo predictions are not medically meaningful.

To enable real inference, place the trained checkpoint at:

```text
checkpoints/attention_guided_final_a100/best.pt
```

The checkpoint must contain a compatible `model_state_dict` for the architecture in `src/`.

## Expected MRI input

Upload one 2-D grayscale image for each view:

- Axial
- Coronal
- Sagittal

Supported formats: PNG, JPG/JPEG, BMP, TIFF.

Images are normalized and resized/padded to 224 x 224 before inference.

## Important

This software is for research and demonstration purposes only. It is not a medical device and must not be used to diagnose or treat a patient.
