import gradio as gr
import torch
import timm
import json
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

MODEL_REPO = "Nishchayb9/plant-disease-efficientnetv2-b3"

MODEL_FILE = "plant_diseases_efficientnetv2_b3_add_augmented-1-weights.pth"
CLASSES_FILE = "classes.json"

device = torch.device("cpu")

# Download model weights
model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE
)

# Download classes
classes_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=CLASSES_FILE
)

# Load class names
with open(classes_path) as f:
    classes = json.load(f)

# Create model architecture
model = timm.create_model(
    "tf_efficientnetv2_b3.in21k",
    pretrained=False,
    num_classes=len(classes)
)

# Load weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def classify_image(image):

    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]

    return {classes[i]: float(probs[i]) for i in range(len(classes))}

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Plant Health Predictor",
    description="Upload a plant leaf image to detect disease."
)

interface.launch()