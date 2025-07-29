import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
@st.cache_resource(allow_output_mutation=True)
def load_mri_model():
    model = load_model("b29088e9-c1fe-47c6-a1c9-a12ff2810fce.h5")
    return model

model = load_mri_model()

# Class labels (confirm if this is correct order)
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# Streamlit UI
st.title("🧠 Alzheimer's MRI Classifier")
st.write("Upload an MRI image to predict Alzheimer's stage.")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)

    # Preprocess
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 128, 128, 3)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown("---")
    st.subheader("🧪 Prediction Result")
    st.success(f"Predicted Class: **{predicted_class}**")

    st.subheader("📊 Confidence Scores")
    for i, prob in enumerate(prediction):
        st.write(f"{class_names[i]}: {prob*100:.2f}%")
