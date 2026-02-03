"""Create a single PDF report for a run.

Input: a run directory produced by train.py (runs/C1/...)
Output: <run_dir>/report.pdf

The report is designed for quick sanity checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _maybe_ax_title(ax, title: str) -> None:
    if title:
        ax.set_title(title)


def _plot_step_curves(pp: PdfPages, steps: pd.DataFrame) -> None:
    if steps.empty:
        return
    fig = plt.figure(figsize=(8.0, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(steps["it"], steps["loss"], label="train")
    if "val_loss" in steps.columns:
        ax.plot(steps["it"], steps["val_loss"], label="val")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    pp.savefig(fig)
    plt.close(fig)

    if "t_sec" in steps.columns:
        fig = plt.figure(figsize=(8.0, 4.5))
        ax = fig.add_subplot(111)
        ax.plot(steps["t_sec"], steps["loss"], label="train")
        if "val_loss" in steps.columns:
            ax.plot(steps["t_sec"], steps["val_loss"], label="val")
        ax.set_xlabel("time (sec)")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        pp.savefig(fig)
        plt.close(fig)


def _plot_dc(pp: PdfPages, diag: pd.DataFrame) -> None:
    if diag.empty:
        return

    # DC_hat
    if "dc_hat" in diag.columns:
        fig = plt.figure(figsize=(8.0, 4.5))
        ax = fig.add_subplot(111)
        ax.plot(diag["it"], diag["dc_hat"], label="DC_hat")
        ax.set_xlabel("iteration")
        ax.set_ylabel("DC_hat")
        ax.grid(True, alpha=0.3)
        ax.legend()
        pp.savefig(fig)
        plt.close(fig)

    # P,G,E
    for col in ["p_hat", "g_hat", "e_hat", "tau"]:
        if col not in diag.columns:
            continue
        fig = plt.figure(figsize=(8.0, 4.5))
        ax = fig.add_subplot(111)
        ax.plot(diag["it"], diag[col], label=col)
        ax.set_xlabel("iteration")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        ax.legend()
        pp.savefig(fig)
        plt.close(fig)

    # SSE scatter proxy: u_norm vs g_norm
    if "g_norm" in diag.columns and "u_norm" in diag.columns:
        fig = plt.figure(figsize=(6.0, 6.0))
        ax = fig.add_subplot(111)
        x = np.maximum(diag["g_norm"].to_numpy(), 1e-12)
        y = np.maximum(diag["u_norm"].to_numpy(), 1e-12)
        ax.scatter(np.log10(x), np.log10(y), s=12)
        ax.set_xlabel("log10 ||g||")
        ax.set_ylabel("log10 ||u||")
        ax.grid(True, alpha=0.3)
        pp.savefig(fig)
        plt.close(fig)


def _plot_selector(pp: PdfPages, sel: pd.DataFrame) -> None:
    if sel.empty:
        return

    fig = plt.figure(figsize=(10.0, 3.5))
    ax = fig.add_subplot(111)
    ax.plot(sel["it"], sel["active_idx"], drawstyle="steps-post")
    ax.set_xlabel("iteration")
    ax.set_ylabel("active optimizer index")
    ax.grid(True, alpha=0.3)
    pp.savefig(fig)
    plt.close(fig)

    # DC EMA per candidate if present
    ema_cols = [c for c in sel.columns if c.startswith("dc_ema_")]
    if ema_cols:
        fig = plt.figure(figsize=(10.0, 4.5))
        ax = fig.add_subplot(111)
        for c in ema_cols:
            ax.plot(sel["it"], sel[c], label=c.replace("dc_ema_", ""))
        ax.set_xlabel("iteration")
        ax.set_ylabel("DC EMA")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)
        pp.savefig(fig)
        plt.close(fig)


def _plot_tails(pp: PdfPages, run_dir: Path) -> None:
    npz_path = run_dir / "tail_samples.npz"
    if not npz_path.exists():
        return
    data = np.load(npz_path)
    g = data.get("grad_abs")
    u = data.get("step_abs")
    if g is None or u is None:
        return

    def ccdf(x: np.ndarray):
        x = x[np.isfinite(x)]
        x = x[x > 0]
        if x.size == 0:
            return None
        x = np.sort(x)
        y = 1.0 - np.arange(1, x.size + 1) / x.size
        return x, y

    for arr, name in [(g, "|grad|"), (u, "|step|")]:
        out = ccdf(arr.astype(np.float64))
        if out is None:
            continue
        x, y = out
        fig = plt.figure(figsize=(6.0, 6.0))
        ax = fig.add_subplot(111)
        ax.loglog(x, y)
        ax.set_xlabel(name)
        ax.set_ylabel("CCDF")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"Tail CCDF: {name}")
        pp.savefig(fig)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out is not None else run_dir / "report.pdf"

    steps = _read_jsonl(run_dir / "step.jsonl")
    diag = _read_jsonl(run_dir / "diag.jsonl")
    sel = _read_jsonl(run_dir / "sel.jsonl")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pp:
        # cover page
        fig = plt.figure(figsize=(8.5, 11.0))
        ax = fig.add_subplot(111)
        ax.axis("off")
        title = run_dir.name
        meta_path = run_dir / "meta.json"
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        lines = [title, "", f"run_dir: {run_dir}"]
        if meta:
            cfg = meta.get("config", {})
            opt = cfg.get("optimizer", "?")
            lines.append(f"optimizer: {opt}")
            lines.append(f"dataset: {cfg.get('dataset', '?')}")
            lines.append(f"max_iters: {cfg.get('max_iters', '?')}")
        ax.text(0.02, 0.98, "\n".join(lines), va="top")
        pp.savefig(fig)
        plt.close(fig)

        _plot_step_curves(pp, steps)
        _plot_dc(pp, diag)
        _plot_selector(pp, sel)
        _plot_tails(pp, run_dir)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
