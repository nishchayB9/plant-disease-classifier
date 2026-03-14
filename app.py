import gradio as gr
import torch
import torch.nn as nn
import timm
import json
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

# ── Read head dimensions directly from saved tensors ──────────────────────────
bn1_size    = raw_sd["1.2.running_mean"].shape[0]   # 3072
linear1_out = raw_sd["1.4.weight"].shape[0]         # 512
bn2_size    = raw_sd["1.6.running_mean"].shape[0]   # 512

print(f"Head dims: BN1={bn1_size}, Linear1→{linear1_out}, BN2={bn2_size}, out={NUM_CLASSES}")

# ── Remap backbone keys ────────────────────────────────────────────────────────
backbone_sd = {
    k[len("0.model."):]: v
    for k, v in raw_sd.items()
    if k.startswith("0.model.")
}

# ── Build backbone — num_classes=0 returns raw features ───────────────────────
backbone = timm.create_model(
    "tf_efficientnetv2_b3.in21k",
    pretrained=False,
    num_classes=0
)
backbone.load_state_dict(backbone_sd, strict=True)

# ── FastAI uses AdaptiveConcatPool → backbone output is 3072 (1536*2) ─────────
# We implement this in the forward pass below

# ── Build FastAI head state dict and load via load_state_dict ─────────────────
# Build layers with correct sizes, then load all weights at once using
# load_state_dict — this correctly handles running_mean/var as buffers
class FastAIHead(nn.Module):
    def __init__(self, bn1_size, linear1_out, bn2_size, num_classes):
        super().__init__()
        self.bn1     = nn.BatchNorm1d(bn1_size)
        self.drop1   = nn.Dropout(p=0.25)
        self.lin1    = nn.Linear(bn1_size, linear1_out, bias=False)
        self.relu    = nn.ReLU(inplace=True)
        self.bn2     = nn.BatchNorm1d(bn2_size)
        self.drop2   = nn.Dropout(p=0.5)
        self.lin2    = nn.Linear(linear1_out, num_classes, bias=False)

    def forward(self, x):
        x = self.bn1(x)
        x = self.drop1(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.lin2(x)
        return x

head = FastAIHead(bn1_size, linear1_out, bn2_size, NUM_CLASSES)

# Load via proper state dict mapping
head_sd = {
    "bn1.weight":          raw_sd["1.2.weight"].float(),
    "bn1.bias":            raw_sd["1.2.bias"].float(),
    "bn1.running_mean":    raw_sd["1.2.running_mean"].float(),
    "bn1.running_var":     raw_sd["1.2.running_var"].float(),
    "bn1.num_batches_tracked": raw_sd["1.2.num_batches_tracked"],
    "lin1.weight":         raw_sd["1.4.weight"].float(),
    "bn2.weight":          raw_sd["1.6.weight"].float(),
    "bn2.bias":            raw_sd["1.6.bias"].float(),
    "bn2.running_mean":    raw_sd["1.6.running_mean"].float(),
    "bn2.running_var":     raw_sd["1.6.running_var"].float(),
    "bn2.num_batches_tracked": raw_sd["1.6.num_batches_tracked"],
    "lin2.weight":         raw_sd["1.8.weight"].float(),
}
head.load_state_dict(head_sd, strict=False)

# ── Full model with AdaptiveConcatPool ────────────────────────────────────────
class PlantDiseaseModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, x):
        x = self.backbone.forward_features(x)   # (B, C, H, W)
        # AdaptiveConcatPool: avg + max → concat → flatten → (B, C*2)
        avg = x.mean(dim=[2, 3])
        mx  = x.amax(dim=[2, 3])
        x   = torch.cat([avg, mx], dim=1)        # (B, 3072)
        x   = self.head(x)
        return x

model = PlantDiseaseModel(backbone, head)
model.eval()

# ── Image preprocessing ────────────────────────────────────────────────────────
data_config = timm.data.resolve_model_data_config(backbone)
transform   = timm.data.create_transform(**data_config, is_training=False)

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