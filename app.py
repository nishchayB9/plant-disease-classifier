import gradio as gr
from fastai.vision.all import *
from fastai.vision.core import PILImage as FAImage
from huggingface_hub import hf_hub_download
import numpy as np

MODEL_REPO = "Nishchayb9/plant-disease-efficientnetv2-b3"
MODEL_FILENAME = "plant_diseases_model_efficientnetv2_b3_add_augmented.pkl"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILENAME
)

learner = load_learner(model_path, cpu=True)
learner.dls.after_item = Pipeline([])
learner.dls.after_batch = Pipeline([])

def classify_image(img):
    # FastAI's predict handles PIL images and single-item transforms correctly
    pred, pred_idx, probs = learner.predict(img)
    return dict(zip(learner.dls.vocab, map(float, probs)))

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Plant Health Predictor",
    description="Upload an image of a plant leaf to identify disease."
)

interface.launch()