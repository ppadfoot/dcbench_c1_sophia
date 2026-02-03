#!/usr/bin/env bash
set -euo pipefail

CFG=${CFG:-configs/c1_paper.py}

# Selector uses the candidate list from the config by default.
# You can override with: --selector_candidates "adamw,lion,muon,sophiag,lamb,sgd"

python train.py "$CFG" --optimizer selector --diag --run_name C1_selector
