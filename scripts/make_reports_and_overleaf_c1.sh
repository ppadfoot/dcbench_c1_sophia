#!/usr/bin/env bash
set -euo pipefail

RUNS_ROOT=${RUNS_ROOT:-runs/C1}

# Per-run PDF reports
for d in "$RUNS_ROOT"/*; do
  if [[ -d "$d" ]]; then
    echo "[report] $d"
    python tools/make_run_report_c1.py --run_dir "$d" || true
  fi
done

# Aggregate figs for Overleaf
python tools/make_overleaf_figs_c1.py --runs_root "$RUNS_ROOT" --out_dir overleaf_figs/figures
