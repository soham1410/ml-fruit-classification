"""
src/inference.py
~~~~~~~~~~~~~~~~

Light-weight inference helpers used by the Streamlit front-end (app.py).

Functions
---------
predict_all(img_pil) -> dict, str
    Returns confidence scores for every pipeline and the final predicted class.
"""
# -------------------- imports --------------------
import os
import json
import joblib
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
# ------------------------------------------------------------------
# 1.  Global model-loading (executed once at import time)
# ------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---- CNN ----
from src.cnn import CNN
CNN_MODEL = CNN(num_classes=219).to(DEVICE)
CNN_MODEL.load_state_dict(torch.load("models/cnn.pth", map_location=DEVICE))
CNN_MODEL.eval()
# class names
with open("models/cnn_classes.json", "r") as fp:
    CLASS_NAMES = json.load(fp)
# ---- Ensemble ----
from src.ensemble import EnsembleModel
ENSEMBLE = joblib.load("models/ensemble.pkl")
# ---- PCA + SVM (k=2,5,8) ----
PCA_DICT = {}  # k -> PCA object
SVM_DICT = {}  # k -> SVM object
for k in [2, 5, 8]:
    PCA_DICT[k] = joblib.load(f"models/pca_k{k}.pkl")
    SVM_DICT[k] = joblib.load(f"models/svm_k{k}_fold15.pkl")
# ------------------------------------------------------------------
# 2.  Helper transformers
# ------------------------------------------------------------------
def _flatten_transform(img: Image.Image) -> np.ndarray:
    """
    Resize to 100×100, convert to tensor, flatten to 30 000-D numpy vector.
    """
    t = transforms.ToTensor()(img.resize((100, 100)))
    return t.view(-1).numpy()
def _cnn_transform(img: Image.Image) -> torch.Tensor:
    """
    Resize to 100×100 and convert to tensor (keeps 3×100×100).
    """
    return transforms.ToTensor()(img.resize((100, 100)))
# ------------------------------------------------------------------
# 3.  Core API
# ------------------------------------------------------------------
def predict_all(img_pil: Image.Image):
    """
    Parameters
    ----------
    img_pil : PIL.Image
        RGB fruit image of any size.

    Returns
    -------
    probs : dict
        Keys match the UI labels:
        'SVM+PCA+KF (k=2)', 'SVM+PCA+KF (k=5)', 'SVM+PCA+KF (k=8)',
        'CNN', 'Ensemble'
        Values are float confidence scores (0–1) for the *predicted* class.
    label : str
        Human-readable class name (from ensemble prediction).
    """
    # ---------------- CNN branch ----------------
    x_cnn = _cnn_transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = CNN_MODEL(x_cnn)
        prob_cnn = torch.softmax(logits, dim=1).cpu().numpy()[0]
    # ---------------- Ensemble branch ----------------
    prob_ens = ENSEMBLE.predict(img_pil)  # soft-vote vector
    # ---------------- SVM+PCA branches ----------------
    x_flat = _flatten_transform(img_pil)
    prob_svm = {}
    for k in [2, 5, 8]:
        x_pca = PCA_DICT[k].transform([x_flat])
        prob_svm[k] = SVM_DICT[k].predict_proba(x_pca)[0]
    # ---------------- build return dict ----------------
    probs = {
        "SVM+PCA+KF (k=2)": float(prob_svm[2].max()),
        "SVM+PCA+KF (k=5)": float(prob_svm[5].max()),
        "SVM+PCA+KF (k=8)": float(prob_svm[8].max()),
        "CNN": float(prob_cnn.max()),
        "Ensemble": float(prob_ens.max())
    }
    # predicted label comes from ensemble (best single model)
    pred_idx = int(prob_ens.argmax())
    label = CLASS_NAMES[pred_idx]
    return probs, label