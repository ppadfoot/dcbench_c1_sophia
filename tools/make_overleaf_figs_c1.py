"""Aggregate runs into Overleaf-ready figures.

Usage:
  python tools/make_overleaf_figs_c1.py --runs_root runs/C1 --out_dir overleaf_figs/figures

This script looks for run directories produced by train.py containing:
  - meta.json
  - step.jsonl
  - diag.jsonl (optional)
  - sel.jsonl (optional)

It writes PDF figures into the output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def _discover_runs(runs_root: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for d in sorted(runs_root.glob("*")):
        if not d.is_dir():
            continue
        meta = d / "meta.json"
        if not meta.exists():
            continue
        try:
            cfg = json.loads(meta.read_text(encoding="utf-8")).get("config", {})
        except Exception:
            cfg = {}
        opt = cfg.get("optimizer", d.name)
        out.append((str(opt), d))
    return out


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out}")


def _plot_loss_vs_iter(runs: Dict[str, pd.DataFrame], out: Path) -> None:
    fig = plt.figure(figsize=(7.5, 4.5))
    ax = fig.add_subplot(111)
    for name, df in runs.items():
        if df.empty or "it" not in df.columns or "loss" not in df.columns:
            continue
        ax.plot(df["it"], df["loss"], label=name)
    ax.set_xlabel("iteration")
    ax.set_ylabel("train loss")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    _save(fig, out / "c1_loss_vs_iter.pdf")


def _plot_loss_vs_time(runs: Dict[str, pd.DataFrame], out: Path) -> None:
    fig = plt.figure(figsize=(7.5, 4.5))
    ax = fig.add_subplot(111)
    for name, df in runs.items():
        if df.empty or "time_s" not in df.columns or "loss" not in df.columns:
            continue
        ax.plot(df["time_s"], df["loss"], label=name)
    ax.set_xlabel("wall clock (s)")
    ax.set_ylabel("train loss")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    _save(fig, out / "c1_loss_vs_time.pdf")


def _plot_dc_vs_iter(runs: Dict[str, pd.DataFrame], out: Path) -> None:
    fig = plt.figure(figsize=(7.5, 4.5))
    ax = fig.add_subplot(111)
    for name, df in runs.items():
        if df.empty or "it" not in df.columns or "dc_hat" not in df.columns:
            continue
        ax.plot(df["it"], df["dc_hat"], label=name)
    ax.set_xlabel("iteration")
    ax.set_ylabel("DC^ hat")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    _save(fig, out / "c1_dc_vs_iter.pdf")


def _plot_selector_timeline(sel: pd.DataFrame, out: Path) -> None:
    if sel.empty or "it" not in sel.columns or "active_idx" not in sel.columns:
        return
    fig = plt.figure(figsize=(7.5, 2.5))
    ax = fig.add_subplot(111)
    ax.plot(sel["it"], sel["active_idx"], drawstyle="steps-post")
    ax.set_xlabel("iteration")
    ax.set_ylabel("active optimizer")
    ax.grid(True, alpha=0.3)
    _save(fig, out / "c1_selector_timeline.pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)

    discovered = _discover_runs(runs_root)
    if not discovered:
        raise SystemExit(f"No runs found in: {runs_root}")

    step_runs: Dict[str, pd.DataFrame] = {}
    dc_runs: Dict[str, pd.DataFrame] = {}
    selector_df: pd.DataFrame = pd.DataFrame()

    # If multiple runs with same optimizer exist, we take the first one (paper scripts usually run 1x).
    for opt, d in discovered:
        steps = _read_jsonl(d / "step.jsonl")
        if not steps.empty and opt not in step_runs:
            step_runs[opt] = steps

        diag = _read_jsonl(d / "diag.jsonl")
        if not diag.empty and opt not in dc_runs:
            dc_runs[opt] = diag

        if opt == "selector" and selector_df.empty:
            selector_df = _read_jsonl(d / "sel.jsonl")

    _plot_loss_vs_iter(step_runs, out_dir)
    _plot_loss_vs_time(step_runs, out_dir)
    _plot_dc_vs_iter(dc_runs, out_dir)
    _plot_selector_timeline(selector_df, out_dir)


if __name__ == "__main__":
    main()
