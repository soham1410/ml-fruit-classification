"""
app.py
~~~~~~

Streamlit front-end for the Fruit-360 ML mini-project.

Features
--------
1. Sidebar displays the reproduced paper tables (CSV) + PCA scatter plots (Fig 9 & 10).
2. Main area allows drag-and-drop image upload.
3. Instantly predicts with **six** pipelines:
   - SVM+PCA+K-Fold (k = 2, 5, 8)
   - CNN
   - Ensemble (soft-vote CNN + best SVM+PCA)
4. Shows confidence score and predicted class name.

How to run
----------
>>> streamlit run app.py
"""
# -------------------- imports --------------------
import streamlit as st
import pandas as pd
from PIL import Image
from src.inference import predict_all
# -------------------- page config --------------------
st.set_page_config(
    page_title="Fruit-360 ML",
    page_icon="🍒",
    layout="wide"
)
# -------------------- title --------------------
st.title("🍒 Fruit Classification")
st.markdown(
    "Upload a fruit image and compare **six** pipelines:  \n"
    "SVM+PCA+K-Fold (k=2, 5, 8)  •  CNN  •  Ensemble"
)
# -------------------- sidebar --------------------
with st.sidebar:
    st.header("📊 Reproduced Paper Tables")
    try:
        df = pd.read_csv("models/paper_tables.csv")
        st.dataframe(df, use_container_width=True)
    except FileNotFoundError:
        st.warning("Run `python train.py` first to generate tables.")

    st.divider()
    st.header("📈 PCA Visualisations")
    try:
        st.image("figures/pca2d.png", caption="PCA=2 (Fig-9)")
        st.image("figures/pca3d.png", caption="PCA=3 (Fig-10)")
    except FileNotFoundError:
        st.warning("PCA plots not found – run training.")
# -------------------- main area : upload --------------------
uploaded_file = st.file_uploader(
    "Drag & drop a fruit image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
if uploaded_file is not None:
    # display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Input image", width=250)

    # predict
    with st.spinner("Classifying …"):
        probs, label = predict_all(img)

    # show results
    st.success("Done")
    cols = st.columns(3)
    for idx, (model, confidence) in enumerate(probs.items()):
        cols[idx % 3].metric(
            label=model,
            value=f"{confidence*100:.2f} %",
            delta=f"predicts:  **{label}**"
        )
else:
    st.info("👈 Upload an image to start classification.")