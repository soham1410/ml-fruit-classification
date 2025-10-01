"""
app.py
~~~~~~
Streamlit UI – upload an image and get predictions from
 1. SVM+PCA+KF (k=2)
 2. SVM+PCA+KF (k=5)
 3. SVM+PCA+KF (k=8)
 4. CNN
 5. Ensemble
Also shows the paper tables and PCA plots.
"""
import streamlit as st, PIL, joblib, torch, json, pandas as pd
from src.inference import predict_all  # tiny helper that calls every model
st.set_page_config(page_title="Fruit-360 ML Demo", layout="wide")
st.title("🍒 Fruit Classification – Reproducing the DAML 2023 Paper (+ Ensemble)")
st.markdown("Upload a fruit image and compare 6 pipelines.")
# -------------------------------------------------
# side-bar: show paper tables
# -------------------------------------------------
with st.sidebar:
    st.header("Paper Tables")
    df = pd.read_csv("models/paper_tables.csv")
    st.dataframe(df, use_container_width=True)
    st.image("figures/pca2d.png", caption="PCA=2 (Fig-9)")
    st.image("figures/pca3d.png", caption="PCA=3 (Fig-10)")
# -------------------------------------------------
# upload
# -------------------------------------------------
file = st.file_uploader("Choose a fruit photo", type=["jpg","jpeg","png"])
if file:
    img = PIL.Image.open(file).convert("RGB")
    st.image(img, caption="Input", width=200)
    with st.spinner("Predicting …"):
        probs, label = predict_all(img)   # returns dict
    st.success("Done")
    cols = st.columns(3)
    for i, (model, prob) in enumerate(probs.items()):
        cols[i%3].metric(model, f"{prob*100:5.2f} %", label)