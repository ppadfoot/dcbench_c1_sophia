# dcbench-c1-sophia

A **clean, paper-oriented** training + diagnostics + selector repo for **C1** (GPT-style LM) experiments.

This repository is **purposefully minimal** and **aggressively de-legacy'd**:
- **no Shampoo**
- no unused scripts/assets
- one training entrypoint (`train.py`)
- one figure pipeline that produces `overleaf_figs/figures/*.pdf` ready to copy into Overleaf

It is based on the original Sophia training codebase (Liuhong99/Sophia) and extends it with:

- Baselines: **SGD, AdamW, Lion, Muon, SophiaG, LAMB**
- A practical **DC-based selector** that switches optimizers **without losing momentum/state**
- Diagnostics aligned with the paper: **P_hat**, **G_hat**, **E_hat**, **DC_hat**, SSE/ALN probes, heavy-tail sampling
- Per-run PDF report: `figs/<RUN_ID>_report.pdf`

---

## 0) Environment

Recommended:

```bash
conda create -n dcbench python=3.10 -y
conda activate dcbench
pip install -U pip
```

Install dependencies (option A: external optimizers via pip/git):

```bash
pip install -r requirements.txt
```

---

## 1) Data (OpenWebText)

Prepare OpenWebText dataset (same as the Sophia base repo):

```bash
python data/openwebtext/prepare.py
```

This downloads/creates `data/openwebtext/train.bin` and `val.bin`.

---

## 2) Run C1 baselines

All runs are stored under `runs/C1/<RUN_ID>/` with logs:
- `out/step.jsonl` (loss/lr/time/tokens)
- `out/diag.jsonl` (DC + diagnostics, if enabled)
- `meta.json` (run metadata)

Example (one run):

```bash
python train.py \
  --exp C1 \
  --optimizer adamw \
  --config configs/c1_paper.py \
  --seed 0 \
  --run_id C1_paper_adamw_s0
```

Convenience scripts:

```bash
bash scripts/run_baselines_c1.sh
```

---

## 3) Run the selector

The selector compares a portfolio of optimizers by **DC_hat** and switches episodically.
By default it uses **on-trajectory state updates**, meaning each candidate keeps its internal state (momentum/EMA/etc.) up-to-date.

```bash
bash scripts/run_selector_c1.sh
```

Selector logs:
- `out/selector.jsonl` (chosen optimizer, scores, switching events)

---

## 4) Build per-run reports (PDF)

```bash
bash scripts/make_reports_c1.sh
```

This creates one PDF per run:
- `figs/<RUN_ID>_report.pdf`

---

## 5) Build Overleaf-ready figures

```bash
bash scripts/make_overleaf_figs_c1.sh
```

Outputs:
- `overleaf_figs/figures/*.pdf`

Copy everything under `overleaf_figs/figures/` into your Overleaf project `figures/` folder.

---

## Notes on correctness (selector + state)

Switching optimizers naively loses momentum/state and breaks both practice and DC scoring.
This repo implements the selector in the **"experts with persistent state"** sense:
- each optimizer has its own state
- inactive optimizers still receive `lr=0` state-only updates (configurable)
- DC_hat uses probe batches and **does not mutate** the real optimizer states

---

## Repo layout

- `train.py` : unified training entrypoint (baselines + selector)
- `c1bench/` : selector, diagnostics, JSONL logging, utilities
- `configs/c1_paper.py` : C1 training hyperparams (steps/warmup/etc.)
- `tools/` : report + overleaf figure builders
- `scripts/` : ready-to-run bash scripts

---

## License

Base code: Sophia (see `LICENSE`).
External optimizer libraries are installed via `pip`/`git` and keep their own licenses.
# dcbench_c1_sophia
# dcbench_c1_sophia
