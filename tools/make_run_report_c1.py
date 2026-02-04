"""
Create a single PDF report for a run.

Input: a run directory produced by train.py (runs/C1/...)
Output: <run_dir>/report.pdf

Compatible with current logs:
- step.jsonl: uses "iter" (and/or "it"), "loss", optional "event":"eval" with val_loss
- diag.jsonl: uses "dc_hat", "dc_hat_raw", "P_hat","G_hat","E_hat","E_cap","tau_used", "g_norm","u_norm"
- sel.jsonl: uses dict fields "dc_hat"/"dc_ema"/"P_hat"/"G_hat"/"E_hat" + "active"/"best"
- tail_samples.npz: uses either (g_abs,u_abs) or (grad_abs,step_abs)

NEW:
- drops the first (earliest) point from every time-series plot (train/eval/diag/selector),
  because the first point (it=0) is usually noisy and ruins aesthetics.
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


# ----------------------------
# I/O helpers
# ----------------------------
def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _coerce_iter_col(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "it" not in df.columns and "iter" in df.columns:
        df["it"] = df["iter"]
    return df


def _coerce_time_col(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "t_sec" not in df.columns and "time_s" in df.columns:
        df["t_sec"] = df["time_s"]
    return df


def _drop_first_by_it(df: pd.DataFrame, drop_first: int) -> pd.DataFrame:
    """Drop earliest points by iteration, robustly."""
    if df.empty or drop_first <= 0:
        return df
    if "it" not in df.columns:
        return df
    df = df.copy()
    df = df.sort_values("it", kind="mergesort").reset_index(drop=True)
    if len(df) <= drop_first:
        return df.iloc[0:0]  # empty
    return df.iloc[drop_first:].reset_index(drop=True)


def _expand_dict_col(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    """Expand a column containing dicts into separate columns."""
    if df.empty or col not in df.columns:
        return df
    if not any(isinstance(x, dict) for x in df[col].dropna().tolist()):
        return df
    expanded = df[col].apply(lambda x: x if isinstance(x, dict) else {}).apply(pd.Series)
    expanded = expanded.add_prefix(prefix)
    df2 = df.drop(columns=[col])
    return pd.concat([df2, expanded], axis=1)


def _auto_log_y(ax, y: np.ndarray) -> None:
    """Use log scale if values are positive and span many orders of magnitude."""
    y = y.astype(np.float64)
    y = y[np.isfinite(y)]
    y = y[y > 0]
    if y.size < 2:
        return
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    if ymin <= 0:
        return
    if ymax / max(ymin, 1e-300) > 1e3:
        ax.set_yscale("log")


# ----------------------------
# Plotting
# ----------------------------
def _plot_step_curves(pp: PdfPages, steps_raw: pd.DataFrame, drop_first: int) -> None:
    if steps_raw.empty:
        return

    steps_raw = _coerce_iter_col(steps_raw)
    steps_raw = _coerce_time_col(steps_raw)

    train_df = steps_raw[steps_raw.get("loss").notna()] if "loss" in steps_raw.columns else pd.DataFrame()

    eval_df = pd.DataFrame()
    if "event" in steps_raw.columns:
        eval_df = steps_raw[steps_raw["event"].astype(str) == "eval"]
    if eval_df.empty and "val_loss" in steps_raw.columns:
        eval_df = steps_raw[steps_raw.get("val_loss").notna()]

    # Sort + drop first point (noise)
    train_df = _drop_first_by_it(train_df, drop_first)
    eval_df = _drop_first_by_it(eval_df, drop_first)

    # loss vs iteration
    if not train_df.empty and "it" in train_df.columns:
        fig = plt.figure(figsize=(8.5, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(train_df["it"], train_df["loss"], label="train loss")
        if not eval_df.empty and "it" in eval_df.columns and "val_loss" in eval_df.columns:
            ax.plot(eval_df["it"], eval_df["val_loss"], label="val loss", marker="o", linestyle="None")
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        pp.savefig(fig)
        plt.close(fig)

    # loss vs time
    if not train_df.empty and "t_sec" in train_df.columns:
        fig = plt.figure(figsize=(8.5, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(train_df["t_sec"], train_df["loss"], label="train loss")
        if not eval_df.empty and "t_sec" in eval_df.columns and "val_loss" in eval_df.columns:
            ax.plot(eval_df["t_sec"], eval_df["val_loss"], label="val loss", marker="o", linestyle="None")
        ax.set_xlabel("time (sec)")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        pp.savefig(fig)
        plt.close(fig)


def _plot_dc(pp: PdfPages, diag_raw: pd.DataFrame, drop_first: int) -> None:
    if diag_raw.empty:
        return
    diag = _coerce_iter_col(diag_raw)
    diag = _drop_first_by_it(diag, drop_first)

    # DC traces
    for col in ["dc_hat", "dc_hat_raw"]:
        if col in diag.columns:
            fig = plt.figure(figsize=(8.5, 4.8))
            ax = fig.add_subplot(111)
            ax.plot(diag["it"], diag[col], label=col)
            ax.set_xlabel("iteration")
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)
            ax.legend()
            pp.savefig(fig)
            plt.close(fig)

    # P/G/E + caps
    cols = [c for c in ["P_hat", "G_hat", "E_hat", "E_cap", "tau_used"] if c in diag.columns]
    for col in cols:
        fig = plt.figure(figsize=(8.5, 4.8))
        ax = fig.add_subplot(111)
        y = diag[col].to_numpy(dtype=float)
        ax.plot(diag["it"], y, label=col)
        ax.set_xlabel("iteration")
        ax.set_ylabel(col)
        _auto_log_y(ax, y)
        ax.grid(True, alpha=0.3)
        ax.legend()
        pp.savefig(fig)
        plt.close(fig)

    # SSE proxy scatter: u_norm vs g_norm (log-log)
    if "g_norm" in diag.columns and "u_norm" in diag.columns:
        fig = plt.figure(figsize=(6.5, 6.5))
        ax = fig.add_subplot(111)
        x = np.maximum(diag["g_norm"].to_numpy(dtype=float), 1e-12)
        y = np.maximum(diag["u_norm"].to_numpy(dtype=float), 1e-12)
        ax.scatter(np.log10(x), np.log10(y), s=12)
        ax.set_xlabel("log10 ||g||")
        ax.set_ylabel("log10 ||u||")
        ax.grid(True, alpha=0.3)
        ax.set_title("SSE proxy: log||u|| vs log||g||")
        pp.savefig(fig)
        plt.close(fig)


def _plot_selector(pp: PdfPages, sel_raw: pd.DataFrame, drop_first: int) -> None:
    if sel_raw.empty:
        return
    sel = _coerce_iter_col(sel_raw)
    sel = _drop_first_by_it(sel, drop_first)

    # Expand dict columns if present
    sel = _expand_dict_col(sel, "dc_ema", "dc_ema_")
    sel = _expand_dict_col(sel, "dc_hat", "dc_hat_")

    # Candidate list
    cand = [c.replace("dc_ema_", "") for c in sel.columns if c.startswith("dc_ema_")]
    if not cand:
        cand = [c.replace("dc_hat_", "") for c in sel.columns if c.startswith("dc_hat_")]
    if not cand and "active" in sel.columns:
        cand = sorted(set(sel["active"].dropna().astype(str).tolist()))
    cand = list(dict.fromkeys(cand))

    idx = {name: i for i, name in enumerate(cand)}
    if "active" in sel.columns and cand:
        sel["active_idx"] = sel["active"].astype(str).map(lambda x: idx.get(x, -1))

        fig = plt.figure(figsize=(10.5, 3.8))
        ax = fig.add_subplot(111)
        ax.plot(sel["it"], sel["active_idx"], drawstyle="steps-post")
        ax.set_xlabel("iteration")
        ax.set_ylabel("active optimizer")
        ax.set_yticks(range(len(cand)))
        ax.set_yticklabels(cand)
        ax.grid(True, alpha=0.3)
        ax.set_title("Selector: active optimizer over time (first point dropped)")
        pp.savefig(fig)
        plt.close(fig)

    # DC EMA per candidate
    ema_cols = [c for c in sel.columns if c.startswith("dc_ema_")]
    if ema_cols:
        fig = plt.figure(figsize=(10.5, 4.8))
        ax = fig.add_subplot(111)
        for c in sorted(ema_cols):
            ax.plot(sel["it"], sel[c].to_numpy(dtype=float), label=c.replace("dc_ema_", ""))
        ax.set_xlabel("iteration")
        ax.set_ylabel("DC EMA")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        ax.set_title("Selector: DC EMA per candidate (first point dropped)")
        pp.savefig(fig)
        plt.close(fig)


def _plot_tails(pp: PdfPages, run_dir: Path) -> None:
    npz_path = run_dir / "tail_samples.npz"
    if not npz_path.exists():
        return
    data = np.load(npz_path)

    # support both old and new key names
    g = data.get("g_abs", None)
    u = data.get("u_abs", None)
    if g is None:
        g = data.get("grad_abs", None)
    if u is None:
        u = data.get("step_abs", None)
    if g is None or u is None:
        return

    def ccdf(x: np.ndarray):
        x = x.astype(np.float64)
        x = x[np.isfinite(x)]
        x = x[x > 0]
        if x.size == 0:
            return None
        x = np.sort(x)
        y = 1.0 - np.arange(1, x.size + 1) / x.size
        return x, y

    for arr, name in [(g, "|grad|"), (u, "|step|")]:
        out = ccdf(arr)
        if out is None:
            continue
        x, y = out
        fig = plt.figure(figsize=(6.5, 6.5))
        ax = fig.add_subplot(111)
        ax.loglog(x, y)
        ax.set_xlabel(name)
        ax.set_ylabel("CCDF")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"Tail CCDF: {name}")
        pp.savefig(fig)
        plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--drop_first", type=int, default=1, help="Drop earliest points in all time-series plots (default: 1).")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out is not None else run_dir / "report.pdf"

    steps = _read_jsonl(run_dir / "step.jsonl")
    diag = _read_jsonl(run_dir / "diag.jsonl")
    sel = _read_jsonl(run_dir / "sel.jsonl")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pp:
        # Cover page
        fig = plt.figure(figsize=(8.5, 11.0))
        ax = fig.add_subplot(111)
        ax.axis("off")
        title = run_dir.name
        meta_path = run_dir / "meta.json"
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        lines = [title, "", f"run_dir: {run_dir}", f"drop_first: {args.drop_first}"]
        if meta:
            cfg = meta.get("config", {})
            opt = cfg.get("optimizer", meta.get("optimizer", "?"))
            lines.append(f"optimizer: {opt}")
            lines.append(f"dataset: {cfg.get('dataset', '?')}")
            lines.append(f"max_iters: {cfg.get('max_iters', '?')}")
        ax.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=11)
        pp.savefig(fig)
        plt.close(fig)

        _plot_step_curves(pp, steps, args.drop_first)
        _plot_dc(pp, diag, args.drop_first)
        _plot_selector(pp, sel, args.drop_first)
        _plot_tails(pp, run_dir)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
