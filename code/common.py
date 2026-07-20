"""Shared utilities: data loading, backbone construction, preprocessing."""
import os, json, glob
import numpy as np
import torch, torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import torchvision.models as tvm

torch.manual_seed(42); np.random.seed(42)
torch.set_num_threads(4)

# Paths derive from the repo location so the artifact is portable.
# Override any of them with environment variables if desired.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PVXAI_ROOT", os.path.dirname(_HERE))   # repo root (parent of code/)
DATA_ROOT = os.environ.get("PVXAI_DATA", os.path.join(ROOT, "data", "pv"))
CACHE = os.environ.get("PVXAI_CACHE", os.path.join(ROOT, "data", "cache"))
RESULTS = os.environ.get("PVXAI_RESULTS", os.path.join(ROOT, "results"))
ZIP_PATH = os.environ.get("PVXAI_ZIP", os.path.join(ROOT, "data", "data.zip"))
os.makedirs(CACHE, exist_ok=True)
IMG = 160
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_manifest():
    return json.load(open(os.path.join(DATA_ROOT, "manifest.json")))

def list_split(split):
    """Return (paths, labels, class_names) for a split."""
    man = get_manifest(); classes = man["classes"]
    cls2idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for cls, rels in man["manifest"][split].items():
        for r in rels:
            paths.append(os.path.join(DATA_ROOT, r)); labels.append(cls2idx[cls])
    return paths, np.array(labels), classes

# preprocessing: resize->centercrop-like (square resize) then normalize
_resize = T.Resize((IMG, IMG))
_norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

def load_tensor(path):
    """Return normalized CHW float tensor and the raw [0,1] tensor (for CAM overlay/perturbation)."""
    img = Image.open(path).convert("RGB")
    img = _resize(img)
    raw = T.ToTensor()(img)            # [0,1], CHW
    x = _norm(raw.clone())
    return x, raw

class BackboneHead(nn.Module):
    """Feature backbone (frozen) + AdaptiveAvgPool + Linear head. CAM-compatible."""
    def __init__(self, features, pool, head, kind):
        super().__init__()
        self.features = features
        self.pool = pool
        self.head = head
        self.kind = kind
    def forward(self, x):
        f = self.features(x)
        if self.kind == "vit":
            z = f  # already pooled CLS token [B, D]
        else:
            z = self.pool(f).flatten(1)
        return self.head(z)

def build_backbone(name):
    """Return (feature_extractor, feat_dim, target_layer_getter, kind).
    feature_extractor(x) -> conv feature map (CNN) or CLS token (ViT)."""
    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        feats = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                              m.layer1, m.layer2, m.layer3, m.layer4)
        return feats, 2048, ("cnn", None), "cnn"
    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        feats = m.features
        return feats, 1280, ("cnn", None), "cnn"
    if name == "vit_b_16":
        m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
        class ViTFeat(nn.Module):
            def __init__(self, vit): super().__init__(); self.vit = vit
            def forward(self, x):
                v = self.vit
                x = v._process_input(x)
                n = x.shape[0]
                cls = v.class_token.expand(n, -1, -1)
                x = torch.cat([cls, x], dim=1)
                x = v.encoder(x)
                return x[:, 0]  # CLS token [B,768]
        return ViTFeat(m), 768, ("vit", None), "vit"
    raise ValueError(name)

def cnn_target_layer(model, name):
    if name == "resnet50":
        return model.features[-1][-1]        # last bottleneck of layer4
    if name == "efficientnet_b0":
        return model.features[-1]            # last conv block
    raise ValueError(name)
