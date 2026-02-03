#!/usr/bin/env bash
set -euo pipefail

CFG=${CFG:-configs/c1_paper.py}

# NOTE: adjust --device/--dtype if needed.

python train.py "$CFG" --optimizer sgd     --diag --run_name C1_sgd
python train.py "$CFG" --optimizer adamw   --diag --run_name C1_adamw
python train.py "$CFG" --optimizer lion    --diag --run_name C1_lion
python train.py "$CFG" --optimizer muon    --diag --run_name C1_muon
python train.py "$CFG" --optimizer sophiag --diag --run_name C1_sophiag
python train.py "$CFG" --optimizer lamb    --diag --run_name C1_lamb
