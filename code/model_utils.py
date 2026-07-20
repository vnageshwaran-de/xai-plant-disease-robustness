"""Assemble CAM-compatible full models from cached heads."""
import torch, torch.nn as nn
import common as C

RES_DIR = C.RESULTS

def load_model(name):
    feats, D, _, kind = C.build_backbone(name)
    ckpt = torch.load(f"{RES_DIR}/models/{name}_head.pt", map_location="cpu")
    classes = ckpt["classes"]
    head = nn.Linear(D, len(classes)); head.load_state_dict(ckpt["head_state"])
    pool = nn.AdaptiveAvgPool2d(1)
    model = C.BackboneHead(feats, pool, head, kind)
    model.eval()  # eval mode (no BN/dropout updates); grads stay enabled for CAM
    return model, classes

def target_layer(model, name):
    if name == "resnet50":
        return model.features[-1][-1]   # last bottleneck of layer4
    if name == "efficientnet_b0":
        return model.features[-1]       # last Conv2dNormActivation
    raise ValueError(name)
