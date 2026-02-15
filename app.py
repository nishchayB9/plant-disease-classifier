import gradio as gr
from fastai.vision.all import *
from fastai.vision.core import PILImage as FAImage
import numpy as np
# Load the model
try:
    learner = load_learner("plant_diseases_model_efficientnetv2_b3_add_augmented.pkl")
except FileNotFoundError:
    print("Model file not found. Please ensure 'plant_diseases_model_efficientnetv2_b3_add_augmented.pkl' is in the correct directory.")
    exit()

# Define the prediction function

def classify_image(img):
    img = FAImage(img)
    preds, idx, probs = learner.predict(img)
    return {learner.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

# Create the Gradio interface
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload Plant leaf Image"),
    outputs=gr.Label(num_top_classes=3, label="Predicted health status"),
    title="Plant Health Predictor",
    description="Upload an image of a plant leaf to identify whether its healthy or not."
)

# Launch the interface
interface.launch()