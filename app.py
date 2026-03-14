import gradio as gr
import torch
import torch.nn as nn
import timm
import json
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

MODEL_REPO   = "Nishchayb9/plant-disease-efficientnetv2-b3"
MODEL_FILE   = "plant_diseases_efficientnetv2_b3_add_augmented-1-weights.pth"
CLASSES_FILE = "classes.json"

device = torch.device("cpu")

# ── Download files ─────────────────────────────────────────────────────────────
model_path   = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
classes_path = hf_hub_download(repo_id=MODEL_REPO, filename=CLASSES_FILE)

with open(classes_path) as f:
    classes = json.load(f)

NUM_CLASSES = len(classes)

# ── Load raw state dict ────────────────────────────────────────────────────────
raw_sd = torch.load(model_path, map_location=device)

# ── Remap backbone keys ────────────────────────────────────────────────────────
backbone_sd = {
    k[len("0.model."):]: v
    for k, v in raw_sd.items()
    if k.startswith("0.model.")
}

# ── Build timm backbone ────────────────────────────────────────────────────────
backbone = timm.create_model(
    "tf_efficientnetv2_b3.in21k",
    pretrained=False,
    num_classes=0,
    global_pool=""
)
# Weights were saved in fp16 due to .to_fp16() during training — convert to fp32
backbone_sd_fp32 = {k: v.float() for k, v in backbone_sd.items()}
backbone.load_state_dict(backbone_sd_fp32, strict=False)

# ── FastAI head ────────────────────────────────────────────────────────────────
class FastAIHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap    = nn.AdaptiveAvgPool2d(1)
        self.mp    = nn.AdaptiveMaxPool2d(1)
        self.bn1   = nn.BatchNorm1d(3072, eps=1e-05, momentum=0.1)
        self.drop1 = nn.Dropout(p=0.25)
        self.lin1  = nn.Linear(3072, 512, bias=False)
        self.relu  = nn.ReLU(inplace=True)
        self.bn2   = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1)
        self.drop2 = nn.Dropout(p=0.5)
        self.lin2  = nn.Linear(512, NUM_CLASSES, bias=False)

    def forward(self, x):
        avg = self.ap(x).flatten(1)
        mx  = self.mp(x).flatten(1)
        x   = torch.cat([avg, mx], dim=1)
        x = self.bn1(x)
        x = self.drop1(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.drop2(x)
        return self.lin2(x)

head = FastAIHead()
head.load_state_dict({
    "bn1.weight":              raw_sd["1.2.weight"].float(),
    "bn1.bias":                raw_sd["1.2.bias"].float(),
    "bn1.running_mean":        raw_sd["1.2.running_mean"].float(),
    "bn1.running_var":         raw_sd["1.2.running_var"].float(),
    "bn1.num_batches_tracked": raw_sd["1.2.num_batches_tracked"],
    "lin1.weight":             raw_sd["1.4.weight"].float(),
    "bn2.weight":              raw_sd["1.6.weight"].float(),
    "bn2.bias":                raw_sd["1.6.bias"].float(),
    "bn2.running_mean":        raw_sd["1.6.running_mean"].float(),
    "bn2.running_var":         raw_sd["1.6.running_var"].float(),
    "bn2.num_batches_tracked": raw_sd["1.6.num_batches_tracked"],
    "lin2.weight":             raw_sd["1.8.weight"].float(),
})

# ── Full model ─────────────────────────────────────────────────────────────────
class PlantDiseaseModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, x):
        x = self.backbone(x)   # (B, 1536, 12, 12)
        return self.head(x)

model = PlantDiseaseModel(backbone, head)
model.eval()

# ── Image preprocessing — matches training EXACTLY ────────────────────────────
# Training pipeline:
#   item_tfms = Resize(460)          → squish to 460x460 (no crop)
#   batch_tfms = aug_transforms(size=384) → at inference: squish to 384x384
#   Normalize.from_stats(*imagenet_stats)
#
# FastAI's Resize default is ResizeMethod.Squish → PIL BILINEAR stretch to square
transform = transforms.Compose([
    transforms.Resize((384, 384), interpolation=Image.BILINEAR),  # squish to 384x384
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── Inference ─────────────────────────────────────────────────────────────────
def classify_image(image):
    image  = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    return {classes[i]: float(probs[i]) for i in range(NUM_CLASSES)}

# ── Gradio UI ──────────────────────────────────────────────────────────────────
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Plant Health Predictor",
    description="Upload a plant leaf image to detect disease."
)

interface.launch()