#!/usr/bin/env bash
set -euo pipefail

# Aggressive cleanup of generated artifacts. Safe to run before committing to GitHub.

# Python caches
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

# Common garbage
find . -type f -name ".DS_Store" -delete
find . -type f -name "=*" -delete
find . -type f -name "~" -delete

# Generated outputs
rm -rf runs
rm -rf overleaf_figs/figures
rm -rf figs

echo "[clean] done"
