"""
10_scorecam_diag.py  Diagnose Score-CAM's fine-tuned collapse.
Hypothesis: Score-CAM weights = softmax over per-channel target LOGITS. Once the
model is highly confident, logit magnitudes are large, the channel-softmax becomes
near winner-take-all, and the map collapses onto a single perturbation-sensitive
channel -> near-zero stability. Test: (a) target-logit magnitude & channel-weight
entropy frozen vs fine-tuned; (b) does a normalized (min-max) channel weighting
restore stability?
"""
import os, json, numpy as np, torch, torch.nn as nn
import torchvision.models as tvm
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import common as C, model_utils as MU, xai_lib as X

RES = C.RESULTS
FT = "/tmp/ftr/efficientnet_b0_ft.pt"
if not os.path.exists(FT): FT = f"{RES}/finetuned/efficientnet_b0_ft.pt"
import sys
NIMG = int(os.environ.get("NIMG", "2"))
sub = json.load(open(f"{RES}/raw/xai_subset.json"))["images"][:NIMG]

def load_ft():
    ck = torch.load(FT, map_location="cpu"); sd = ck["model"] if "model" in ck else ck
    m = tvm.efficientnet_b0(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 38)
    m.load_state_dict(sd); m.eval()
    return m
ft = load_ft(); ft_layer = [ft.features[-1]]
fr, _ = MU.load_model("efficientnet_b0"); fr_layer = [MU.target_layer(fr, "efficientnet_b0")]

def scorecam_weights(model, layer, x, cls, norm="softmax", T=1.0):
    """Return (cam[HxW], weight_vector) replicating library ScoreCAM with configurable
    channel-score normalization."""
    acts = {}
    h = layer[0].register_forward_hook(lambda m,i,o: acts.__setitem__('a', o.detach()))
    with torch.no_grad(): model(x)
    h.remove()
    A = acts['a']  # [1,Cc,h,w]
    up = torch.nn.UpsamplingBilinear2d(size=x.shape[-2:])(A)  # [1,Cc,H,W]
    mn = up.amin((2,3), keepdim=True); mx = up.amax((2,3), keepdim=True)
    upn = (up - mn) / (mx - mn + 1e-8)
    masked = x[:, None] * upn[:, :, None]  # [1,Cc,3,H,W]
    masked = masked[0]  # [Cc,3,H,W]
    scores = []
    with torch.no_grad():
        for i in range(0, masked.shape[0], 128):
            out = model(masked[i:i+128])
            scores.append(out[:, cls])
    scores = torch.cat(scores)  # [Cc] raw logits
    if norm == "softmax":
        w = torch.softmax(scores / T, dim=0)
    elif norm == "minmax":
        w = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        w = w / w.sum()
    cam = (w[None,:,None,None] * upn).sum(1)[0].numpy()
    cam = np.maximum(cam, 0); cam = (cam - cam.min())/(cam.max()-cam.min()+1e-8)
    return cam, w.numpy(), float(scores.abs().mean())

def entropy(w):
    w = w/ (w.sum()+1e-12); return float(-(w*np.log(w+1e-12)).sum())

print("Per-image: target-logit |mag|, ScoreCAM channel-weight entropy (higher=more spread)")
for tag, model, layer in [("frozen", fr, fr_layer), ("fine-tuned", ft, ft_layer)]:
    logmags, ents, maxws = [], [], []
    for rec in sub:
        _, raw = C.load_tensor(rec["path"]); x = X._norm(raw).unsqueeze(0)
        with torch.no_grad(): cls = int(model(x).argmax(1))
        cam, w, lm = scorecam_weights(model, layer, x, cls, "softmax")
        logmags.append(lm); ents.append(entropy(w)); maxws.append(float(w.max()))
    Cc = 1280
    print(f"  {tag:10s}: logit|mag|={np.mean(logmags):7.2f}  weight_entropy={np.mean(ents):.3f} "
          f"(max possible {np.log(Cc):.2f})  max_weight={np.mean(maxws):.3f}")

print("done diagnostic")
