# app.py
import streamlit as st
import plotly.express as px
import pandas as pd
import os
from pathlib import Path
from PIL import Image

from src.prediction import ImagePredictor

# Page config
st.set_page_config(page_title="Intel Image Classifier", layout="wide")
st.title("🌄 Intel Image Classification Pipeline")


# Initialize predictor
@st.cache_resource
def get_predictor():
    BASE_DIR = Path(__file__).parent.resolve()
    model_path = BASE_DIR / "models" / "intel_image_model.keras"
    return ImagePredictor(model_path=str(model_path))


predictor = get_predictor()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Single Prediction", "Visualizations", "Bulk Retraining"])

# ==================== SINGLE PREDICTION PAGE ====================
if page == "Single Prediction":
    st.header("Single Image Prediction")

    uploaded_file = st.file_uploader("Upload an image (buildings, forest, glacier, mountain, sea, street)",
                                     type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            # Save temporarily and predict
            temp_path = Path("temp") / uploaded_file.name
            temp_path.parent.mkdir(exist_ok=True)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            result = predictor.predict_image(str(temp_path))

            st.success(f"**Predicted: {result['class'].upper()}**")
            st.metric("Confidence", f"{result['confidence']}%")

            # Show probabilities
            prob_df = pd.DataFrame({
                "Class": list(result['probabilities'].keys()),
                "Probability (%)": list(result['probabilities'].values())
            })
            fig = px.bar(prob_df, x="Class", y="Probability (%)",
                         title="Prediction Probabilities")
            st.plotly_chart(fig, use_container_width=True)

            # Clean up
            if temp_path.exists():
                os.unlink(temp_path)

# ==================== VISUALIZATIONS PAGE ====================
elif page == "Visualizations":
    st.header("Dataset Visualizations & Insights")

    st.subheader("1. Class Distribution")
    st.write(
        "**Interpretation:** The dataset is relatively balanced across the 6 landscape categories, with 'glacier' and 'mountain' having slightly more samples. This helps the model learn these classes well.")

    # You can add actual plots here later from your notebook if you want
    st.info("Class distribution plot would go here (from notebook)")

    st.subheader("2. Color Distribution per Class")
    st.write(
        "**Interpretation:** 'Forest' shows high green channel values due to vegetation. 'Sea' has dominant blue tones. 'Street' and 'Buildings' show more gray/red urban colors. Color is a strong feature for distinguishing scenes.")

    st.subheader("3. Brightness Analysis")
    st.write(
        "**Interpretation:** 'Glacier' and 'Mountain' tend to be much brighter due to snow and ice reflection, while 'Forest' is darker due to dense tree cover. Brightness helps the model separate snowy vs vegetated landscapes.")

# ==================== BULK RETRAINING PAGE ====================
elif page == "Bulk Retraining":
    st.header("Bulk Data Upload & Retraining")
    st.write("Upload a zip file or multiple images to retrain the model.")

    uploaded_bulk = st.file_uploader("Upload new training data (zip or folder)",
                                     type=["zip"], accept_multiple_files=False)

    if st.button("🚀 Trigger Retraining", type="primary"):
        if uploaded_bulk is not None:
            st.warning("Retraining functionality will be implemented in the next step.")
            st.info("This button will later call the retraining pipeline.")
        else:
            st.error("Please upload data first.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("MLOps Pipeline - Intel Image Classification")