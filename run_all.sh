#!/usr/bin/env bash
# One-command reproduction of every table and figure in the paper.
# Usage:   bash run_all.sh
# CPU-only. First run downloads PlantVillage (~2.18 GB). Seed = 42 throughout.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PVXAI_ROOT="$ROOT"
mkdir -p "$ROOT/data"
cd "$ROOT/code"

echo "== [1/7] Download PlantVillage archive (if absent) =="
if [ ! -f "$ROOT/data/data.zip" ]; then
  python3 - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download('mohanty/PlantVillage', 'data.zip', repo_type='dataset')
shutil.copy(p, os.path.join(os.environ['PVXAI_ROOT'], 'data', 'data.zip'))
print('archive ready')
PY
fi

echo "== [2/7] Prepare leaf-split subset =="
python3 01_prepare_data.py

echo "== [3/7] Train backbones (linear probing) =="
for bb in resnet50 efficientnet_b0 vit_b_16; do
  until python3 02_train.py "$bb" | tee /dev/stderr | grep -q METRICS; do
    echo "  ...resuming $bb"; done
done

echo "== [4/7] Prediction robustness + XAI subset =="
python3 03a_pred_robustness.py

echo "== [5/7] Explanation stability + faithfulness (resumable) =="
until python3 03b_xai_eval.py | tee /dev/stderr | grep -q ALL_DONE; do
  echo "  ...resuming XAI eval"; done

echo "== [6/7] Aggregate, statistics, figures =="
python3 04_analyze.py
python3 05_qualitative.py    # first montage
python3 05_qualitative.py    # second montage (skip-if-exists)

echo "== [7/7] Compile the paper =="
cd "$ROOT/paper"
pdflatex -interaction=nonstopmode paper.tex >/dev/null
bibtex paper >/dev/null
pdflatex -interaction=nonstopmode paper.tex >/dev/null
pdflatex -interaction=nonstopmode paper.tex >/dev/null

echo "DONE -> $ROOT/paper/paper.pdf"
echo "Tables: $ROOT/results/tables   Figures: $ROOT/results/figures"
