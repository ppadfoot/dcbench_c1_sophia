"""
Create a single PDF report for a run.

Drops the first (earliest) point from all time-series plots (usually it=0).

Adds "best paper" diagnostics:
- frac_G_nonpos
- G_hat vs G_used
- score_hat (P^2/E proxy) + score_hat_raw
- dc_hat + dc_hat_raw (+ uncapped if present)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

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
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _coerce_iter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "it" not in df.columns and "iter" in df.columns:
        df["it"] = df["iter"]
    return df


def _coerce_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "t_sec" not in df.columns and "time_s" in df.columns:
        df["t_sec"] = df["time_s"]
    return df


def _drop_first(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty or n <= 0 or "it" not in df.columns:
        return df
    df = df.copy().sort_values("it", kind="mergesort").reset_index(drop=True)
    if len(df) <= n:
        return df.iloc[0:0]
    return df.iloc[n:].reset_index(drop=True)


def _auto_log_y(ax, y: np.ndarray) -> None:
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


def _expand_dict_col(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    if not any(isinstance(x, dict) for x in df[col].dropna().tolist()):
        return df
    expanded = df[col].apply(lambda x: x if isinstance(x, dict) else {}).apply(pd.Series)
    expanded = expanded.add_prefix(prefix)
    df2 = df.drop(columns=[col])
    return pd.concat([df2, expanded], axis=1)


def _plot_series(pp: PdfPages, x, y, title: str, xlabel: str, ylabel: str, logy: bool = False) -> None:
    fig = plt.figure(figsize=(8.5, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        _auto_log_y(ax, np.asarray(y))
    ax.grid(True, alpha=0.3)
    pp.savefig(fig)
    plt.close(fig)


def _plot_steps(pp: PdfPages, steps: pd.DataFrame) -> None:
    if steps.empty:
        return
    steps = _coerce_iter(_coerce_time(steps))

    train = steps[steps.get("loss").notna()] if "loss" in steps.columns else pd.DataFrame()
    eval_df = pd.DataFrame()
    if "event" in steps.columns:
        eval_df = steps[steps["event"].astype(str) == "eval"]
    if eval_df.empty and "val_loss" in steps.columns:
        eval_df = steps[steps.get("val_loss").notna()]

    if not train.empty and "it" in train.columns:
        fig = plt.figure(figsize=(8.5, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(train["it"], train["loss"], label="train loss")
        if not eval_df.empty and "it" in eval_df.columns and "val_loss" in eval_df.columns:
            ax.plot(eval_df["it"], eval_df["val_loss"], label="val loss", marker="o", linestyle="None")
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        pp.savefig(fig)
        plt.close(fig)

    if not train.empty and "t_sec" in train.columns:
        fig = plt.figure(figsize=(8.5, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(train["t_sec"], train["loss"], label="train loss")
        if not eval_df.empty and "t_sec" in eval_df.columns and "val_loss" in eval_df.columns:
            ax.plot(eval_df["t_sec"], eval_df["val_loss"], label="val loss", marker="o", linestyle="None")
        ax.set_xlabel("time (sec)")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        pp.savefig(fig)
        plt.close(fig)


def _plot_diag(pp: PdfPages, diag: pd.DataFrame) -> None:
    if diag.empty:
        return
    diag = _coerce_iter(diag)

    # DC hat / raw
    for col in ["dc_hat", "dc_hat_raw", "dc_hat_uncapped", "dc_hat_raw_uncapped"]:
        if col in diag.columns:
            _plot_series(
                pp,
                diag["it"],
                diag[col].to_numpy(dtype=float),
                title=f"{col} vs iteration",
                xlabel="iteration",
                ylabel=col,
                logy=False,
            )

    # Score (P^2/E) proxies (more directly tied to master inequality)
    for col in ["score_hat", "score_hat_raw"]:
        if col in diag.columns:
            _plot_series(
                pp,
                diag["it"],
                diag[col].to_numpy(dtype=float),
                title=f"{col} vs iteration",
                xlabel="iteration",
                ylabel=col,
                logy=True,
            )

    # P/G/E
    for col in ["P_hat", "P_pos", "G_hat", "G_pos_mean", "G_used", "E_hat", "E_cap", "tau_used", "g2a_hat", "g2b_hat"]:
        if col in diag.columns:
            _plot_series(
                pp,
                diag["it"],
                diag[col].to_numpy(dtype=float),
                title=f"{col} vs iteration",
                xlabel="iteration",
                ylabel=col,
                logy=True,
            )

    # show G_hat vs G_used together (key for stability)
    if "G_hat" in diag.columns and "G_used" in diag.columns:
        fig = plt.figure(figsize=(8.5, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(diag["it"], diag["G_hat"].to_numpy(dtype=float), label="G_hat (mean gab)")
        ax.plot(diag["it"], diag["G_used"].to_numpy(dtype=float), label="G_used (pos-mean / floor)")
        ax.set_xlabel("iteration")
        ax.set_ylabel("G")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _auto_log_y(ax, diag["G_used"].to_numpy(dtype=float))
        pp.savefig(fig)
        plt.close(fig)

    # frac_G_nonpos
    if "frac_G_nonpos" in diag.columns:
        fig = plt.figure(figsize=(8.5, 3.8))
        ax = fig.add_subplot(111)
        ax.plot(diag["it"], diag["frac_G_nonpos"].to_numpy(dtype=float), label="frac_G_nonpos")
        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.set_xlabel("iteration")
        ax.set_ylabel("fraction")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.set_title("Fraction of probes with G_probe <= 0 (instability indicator)")
        pp.savefig(fig)
        plt.close(fig)

    # SSE proxy scatter
    if "g_norm" in diag.columns and "u_norm" in diag.columns:
        fig = plt.figure(figsize=(6.5, 6.5))
        ax = fig.add_subplot(111)
        x = np.maximum(diag["g_norm"].to_numpy(dtype=float), 1e-12)
        y = np.maximum(diag["u_norm"].to_numpy(dtype=float), 1e-12)
        ax.scatter(np.log10(x), np.log10(y), s=12)
        ax.set_xlabel("log10 g_norm")
        ax.set_ylabel("log10 u_norm")
        ax.grid(True, alpha=0.3)
        ax.set_title("SSE proxy: log||u|| vs log||g||")
        pp.savefig(fig)
        plt.close(fig)


def _plot_selector(pp: PdfPages, sel_raw: pd.DataFrame) -> None:
    if sel_raw.empty:
        return
    sel = _coerce_iter(sel_raw)

    sel = _expand_dict_col(sel, "dc_ema", "dc_ema_")
    sel = _expand_dict_col(sel, "dc_hat", "dc_hat_")

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
        ax.set_title("Selector: active optimizer over time")
        pp.savefig(fig)
        plt.close(fig)

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
        ax.set_title("Selector: DC EMA per candidate")
        pp.savefig(fig)
        plt.close(fig)


def _plot_tails(pp: PdfPages, run_dir: Path) -> None:
    npz_path = run_dir / "tail_samples.npz"
    if not npz_path.exists():
        return
    data = np.load(npz_path)

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--drop_first", type=int, default=1)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out is not None else run_dir / "report.pdf"

    steps = _read_jsonl(run_dir / "step.jsonl")
    diag = _read_jsonl(run_dir / "diag.jsonl")
    sel = _read_jsonl(run_dir / "sel.jsonl")

    steps = _drop_first(_coerce_iter(_coerce_time(steps)), args.drop_first)
    diag = _drop_first(_coerce_iter(diag), args.drop_first)
    sel = _drop_first(_coerce_iter(sel), args.drop_first)

    meta_path = run_dir / "meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pp:
        fig = plt.figure(figsize=(8.5, 11.0))
        ax = fig.add_subplot(111)
        ax.axis("off")

        lines = [
            run_dir.name,
            "",
            f"run_dir: {run_dir}",
            f"drop_first: {args.drop_first}",
        ]
        if meta:
            cfg = meta.get("config", {})
            opt = cfg.get("optimizer", meta.get("optimizer", "?"))
            lines.append(f"optimizer: {opt}")
            lines.append(f"max_iters: {cfg.get('max_iters', '?')}")
            lines.append(f"warmup_iters: {cfg.get('warmup_iters', '?')}")
            lines.append(f"diag_every: {cfg.get('diag_every', '?')}")
            lines.append(f"diag_probes: {cfg.get('diag_probes', '?')}")

        ax.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=11)
        pp.savefig(fig)
        plt.close(fig)

        _plot_steps(pp, steps)
        _plot_diag(pp, diag)
        _plot_selector(pp, sel)
        _plot_tails(pp, run_dir)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
