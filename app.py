import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model.h5")

# Define class labels (modify if different)
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# Prediction function
def predict_mri(image):
    image = image.convert("RGB").resize((128, 128))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence_scores = {class_names[i]: float(f"{prediction[i]*100:.2f}") for i in range(len(class_names))}
    return predicted_class, confidence_scores

# Create Gradio interface
iface = gr.Interface(
    fn=predict_mri,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(label="Predicted Class"),
        gr.JSON(label="Confidence Scores")
    ],
    title="Alzheimer's MRI Classifier",
    description="Upload an MRI scan to classify Alzheimer's stage."
)

# Launch locally or on HF Spaces
if __name__ == "__main__":
    iface.launch()
