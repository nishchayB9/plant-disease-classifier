import gradio as gr
import torch
import timm
import json
from huggingface_hub import hf_hub_download

MODEL_REPO = "Nishchayb9/plant-disease-efficientnetv2-b3"
MODEL_FILE = "plant_diseases_efficientnetv2_b3_add_augmented-1-weights.pth"
CLASSES_FILE = "classes.json"

device = torch.device("cpu")

# Download model weights and classes
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
classes_path = hf_hub_download(repo_id=MODEL_REPO, filename=CLASSES_FILE)

with open(classes_path) as f:
    classes = json.load(f)

# Create model architecture
model = timm.create_model(
    "tf_efficientnetv2_b3.in21k",
    pretrained=False,
    num_classes=len(classes)
)

# Load and remap FastAI state dict keys
# FastAI wraps the model as Sequential, so keys are saved as:
#   "0.model.conv_stem.weight"  →  we need  "conv_stem.weight"
raw_state_dict = torch.load(model_path, map_location=device)

remapped = {}
for k, v in raw_state_dict.items():
    if k.startswith("0.model."):
        remapped[k[len("0.model."):]] = v
    # Keys starting with "1." are FastAI's custom head — skip them,
    # timm has its own classifier layer which we load separately below

# Load backbone weights (strict=False lets the classifier layer be skipped)
model.load_state_dict(remapped, strict=False)

# Now load the classifier head weights from the FastAI head
# FastAI's final linear is stored at key "1.8.weight" / "1.8.bias"
classifier_weight = raw_state_dict.get("1.8.weight")
classifier_bias   = raw_state_dict.get("1.8.bias")

if classifier_weight is not None:
    # Cast to float32 (model may have been saved in fp16)
    model.classifier.weight = torch.nn.Parameter(classifier_weight.float())
    model.classifier.bias   = torch.nn.Parameter(classifier_bias.float())

model.eval()

# ── Image preprocessing ────────────────────────────────────────────────────────
# timm's preferred way — matches exactly what was used during training
data_config = timm.data.resolve_model_data_config(model)
transform   = timm.data.create_transform(**data_config, is_training=False)

def classify_image(image):
    image  = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    return {classes[i]: float(probs[i]) for i in range(len(classes))}

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Plant Health Predictor",
    description="Upload a plant leaf image to detect disease."
)

interface.launch()