"""
01_prepare_data.py
Build a subsampled, LEAF-SPLIT subset of the PlantVillage 'color' configuration
(mohanty/PlantVillage on Hugging Face). The dataset repo ships split files that
already encode a leaf-based train/test split (prevents leakage between augmented
views of the same physical leaf); we honour that split and subsample within it
with a fixed seed. Images are extracted from the locally downloaded data.zip.
"""
import os, sys, random, json, collections, zipfile
from huggingface_hub import hf_hub_download

SEED = 42
N_TRAIN_PER_CLASS = 35
N_TEST_PER_CLASS  = 25
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PVXAI_ROOT", os.path.dirname(_HERE))
DATA_ROOT = os.environ.get("PVXAI_DATA", os.path.join(ROOT, "data", "pv"))
ZIP_PATH  = os.environ.get("PVXAI_ZIP", os.path.join(ROOT, "data", "data.zip"))

os.makedirs(DATA_ROOT, exist_ok=True)

def load_split(name):
    p = hf_hub_download("mohanty/PlantVillage", f"splits/color_{name}.txt", repo_type="dataset")
    return [l.strip() for l in open(p) if l.strip()]

def by_class(lines):
    d = collections.defaultdict(list)
    for ln in lines:
        d[ln.split("/")[2]].append(ln)
    return d

def subsample(d, n):
    out = {}
    for cls, files in d.items():
        fs = sorted(files); random.Random(SEED).shuffle(fs)
        out[cls] = fs[:min(n, len(fs))]
    return out

train = by_class(load_split("train"))
test  = by_class(load_split("test"))
classes = sorted(set(train) | set(test))
train_sel = subsample(train, N_TRAIN_PER_CLASS)
test_sel  = subsample(test,  N_TEST_PER_CLASS)

manifest = {"train": {}, "test": {}}
jobs = []
for split, sel in (("train", train_sel), ("test", test_sel)):
    for cls, files in sel.items():
        for m in files:
            local_rel = os.path.join(split, cls, os.path.basename(m))
            manifest[split].setdefault(cls, []).append(local_rel)
            jobs.append((m, os.path.join(DATA_ROOT, local_rel)))

with open(os.path.join(DATA_ROOT, "manifest.json"), "w") as f:
    json.dump({"seed": SEED, "n_train_per_class": N_TRAIN_PER_CLASS,
               "n_test_per_class": N_TEST_PER_CLASS, "classes": classes,
               "manifest": manifest}, f, indent=1)

z = zipfile.ZipFile(ZIP_PATH)
names = set(z.namelist())
ok = miss = 0
for m, dst in jobs:
    if os.path.exists(dst):
        ok += 1; continue
    if m not in names:
        miss += 1; continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with z.open(m) as s, open(dst, "wb") as o:
        o.write(s.read())
    ok += 1
print(f"classes={len(classes)} extracted/ok={ok} missing={miss} total={len(jobs)}")
print("train per class:", N_TRAIN_PER_CLASS, "test per class:", N_TEST_PER_CLASS)
