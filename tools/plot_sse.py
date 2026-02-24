#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/plot_sse.py

Readable SSE plots for component magnitudes |g_i| and |U_i| saved by c1bench/sse_diag.py.

Key improvements (vs scatter):
- Use 2D density (hexbin) in (log10|g|, log10|U|) instead of millions of points.
- Overlay binned quantile summary: median + q10/q90 in log-|g| bins.
- Add CCDF(|g_i|) and CCDF(|U_i|) pages (log-log) for each group (tail comparison).
- Robust to slight g/u length mismatch: align by truncation to min(len).
- Drops first aligned point to avoid iter=0 / lr=0 artifacts.

Outputs:
  <run_dir>/figures_sse/
    - sse__iterXXXXXXX__ALL_GROUPS.pdf (multipage: per group)
    - sse__iterXXXXXXX__<group>.pdf    (optional per-group)

Usage:
  python tools/plot_sse.py --run_dir out/A0_adamw_all_diag --mode iters --iters 200,800,1990
"""

import argparse
import glob
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# -----------------------------
# Utils
# -----------------------------
def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _list_npz(run_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(run_dir, "sse", "sse_iter*.npz")), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No SSE npz files found under {run_dir}/sse/")
    return files


def _parse_iter(path: str) -> int:
    m = re.search(r"sse_iter(\d+)\.npz", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _load_groups(npz_path: str) -> List[str]:
    with np.load(npz_path) as z:
        keys = list(z.keys())
    groups = set()
    for k in keys:
        if k.startswith("g_abs__"):
            groups.add(k.split("__", 1)[1])
        if k.startswith("u_abs__"):
            groups.add(k.split("__", 1)[1])
    return sorted(groups)


def _get_arr(z, prefix: str, group: str) -> Optional[np.ndarray]:
    key = f"{prefix}__{group}"
    if key not in z:
        return None
    x = z[key].astype(np.float64, copy=False)
    x = x[np.isfinite(x)]
    return x


def _align_pair(g: np.ndarray, u: np.ndarray, drop_first: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    n = min(g.size, u.size)
    if n <= 0:
        return g[:0], u[:0]
    g = g[:n]
    u = u[:n]
    if drop_first and n >= 2:
        g = g[1:]
        u = u[1:]
    return g, u


# -----------------------------
# Plot helpers
# -----------------------------
def _binned_quantiles(
    x: np.ndarray,
    y: np.ndarray,
    nbins: int = 40,
    q_lo: float = 0.10,
    q_hi: float = 0.90,
    min_per_bin: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin x in log-space. For y in each bin: q10, median, q90.
    Returns: centers, qlo, q50, qhi
    """
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[m]
    y = y[m]
    if x.size < 1000:
        return np.array([]), np.array([]), np.array([]), np.array([])

    xmin = float(np.quantile(x, 0.01))
    xmax = float(np.quantile(x, 0.99))
    xmin = max(xmin, float(x.min()))
    xmax = max(xmax, xmin * 1.001)

    edges = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)
    bin_id = np.digitize(x, edges) - 1

    centers, ql, q50, qh = [], [], [], []
    for b in range(nbins):
        mb = bin_id == b
        if int(mb.sum()) < min_per_bin:
            continue
        yy = y[mb]
        centers.append(np.sqrt(edges[b] * edges[b + 1]))
        ql.append(np.quantile(yy, q_lo))
        q50.append(np.quantile(yy, 0.50))
        qh.append(np.quantile(yy, q_hi))

    return np.array(centers), np.array(ql), np.array(q50), np.array(qh)


def _ccdf_curve(x: np.ndarray, q_lo=0.001, q_hi=0.99999, nbins=700):
    """
    CCDF on log-spaced x-grid between quantiles.
    Returns grid_x, ccdf
    """
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size < 1000:
        return np.array([]), np.array([])
    x = np.sort(x)
    n = x.size
    xmin = float(np.quantile(x, q_lo))
    xmax = float(np.quantile(x, q_hi))
    xmin = max(xmin, float(x[0]))
    xmax = max(xmax, xmin * 1.001)

    g = np.logspace(np.log10(xmin), np.log10(xmax), nbins)
    idx = np.searchsorted(x, g, side="right")
    cc = (n - idx) / n
    cc = np.clip(cc, 1.0/(n+1.0), 1.0)
    return g, cc


def _fit_beta_tail_on_median(bx: np.ndarray, by: np.ndarray, qmin=0.80, qmax=0.98):
    """
    Fit log(by) = a + beta*log(bx) on tail bins (quantile window).
    Returns beta, r2 or (nan,nan) if not enough.
    """
    if bx.size < 10:
        return np.nan, np.nan
    m = np.isfinite(bx) & np.isfinite(by) & (bx > 0) & (by > 0)
    bx = bx[m]; by = by[m]
    if bx.size < 10:
        return np.nan, np.nan

    lo = np.quantile(bx, qmin)
    hi = np.quantile(bx, qmax)
    mm = (bx >= lo) & (bx <= hi)
    if mm.sum() < 5:
        return np.nan, np.nan

    X = np.log(bx[mm])
    Y = np.log(by[mm])
    slope, intercept = np.polyfit(X, Y, 1)
    Yhat = intercept + slope * X
    ss_res = float(np.sum((Y - Yhat) ** 2))
    ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(r2)


def _plot_density_quantiles(ax, g, u, title: str):
    """
    2D density in log-space + quantile summary.
    """
    m = np.isfinite(g) & np.isfinite(u) & (g > 0) & (u > 0)
    g = g[m]; u = u[m]
    if g.size < 2000:
        ax.text(0.1, 0.5, "not enough points", transform=ax.transAxes)
        return

    lg = np.log10(g)
    lu = np.log10(u)

    hb = ax.hexbin(lg, lu, gridsize=140, bins="log", mincnt=5)
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("log10(counts)")

    bx, q10, q50, q90 = _binned_quantiles(g, u, nbins=45, q_lo=0.10, q_hi=0.90, min_per_bin=200)
    if bx.size > 0:
        ax.plot(np.log10(bx), np.log10(q50), linewidth=2.2, label="median")
        ax.plot(np.log10(bx), np.log10(q10), linewidth=1.5, linestyle="--", label="q10/q90")
        ax.plot(np.log10(bx), np.log10(q90), linewidth=1.5, linestyle="--")

        beta, r2 = _fit_beta_tail_on_median(bx, q50, qmin=0.80, qmax=0.98)
        ax.text(
            0.02, 0.98,
            f"tail fit on median: beta={beta:.3g}, R²={r2:.3f}",
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
        )

    ax.set_xlabel("log10 |g_i|")
    ax.set_ylabel("log10 |U_i|")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")


def _plot_ccdf_pair(ax, g_abs, u_abs, title: str):
    """
    CCDF(|g_i|) and CCDF(|U_i|) on log-log.
    """
    gx, gcc = _ccdf_curve(g_abs, q_lo=0.001, q_hi=0.99999, nbins=700)
    ux, ucc = _ccdf_curve(u_abs, q_lo=0.001, q_hi=0.99999, nbins=700)

    if gx.size == 0 or ux.size == 0:
        ax.text(0.1, 0.5, "not enough points", transform=ax.transAxes)
        return

    ax.plot(gx, gcc, linewidth=2.0, label="CCDF(|g_i|)")
    ax.plot(ux, ucc, linewidth=2.0, label="CCDF(|U_i|)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("x")
    ax.set_ylabel("P(>|x|)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="best")


# -----------------------------
# Main plotting
# -----------------------------
def plot_one(npz_path: str, out_dir: str, groups: Optional[List[str]] = None, save_per_group: bool = False) -> None:
    it = _parse_iter(npz_path)
    os.makedirs(out_dir, exist_ok=True)

    all_groups = _load_groups(npz_path)
    if groups is None:
        groups = all_groups
    else:
        missing = [g for g in groups if g not in all_groups]
        if missing:
            print(f"[warn] groups not present in {npz_path}: {missing}; available={all_groups}")

    combined_path = os.path.join(out_dir, f"sse__iter{it:07d}__ALL_GROUPS.pdf")
    with PdfPages(combined_path) as combined:
        with np.load(npz_path) as z:
            for grp in groups:
                g = _get_arr(z, "g_abs", grp)
                u = _get_arr(z, "u_abs", grp)
                if g is None or u is None:
                    continue

                g, u = _align_pair(g, u, drop_first=True)
                m = np.isfinite(g) & np.isfinite(u) & (g > 0) & (u > 0)
                g = g[m]; u = u[m]
                if g.size < 2000:
                    continue

                # Page 1: density + quantiles
                fig = plt.figure(figsize=(8.5, 6.0))
                ax = fig.add_subplot(111)
                _plot_density_quantiles(ax, g, u, title=f"SSE (iter={it}, group={grp})")
                combined.savefig(fig, bbox_inches="tight")
                if save_per_group:
                    fig.savefig(os.path.join(out_dir, f"sse__iter{it:07d}__{grp}__density.pdf"), bbox_inches="tight")
                plt.close(fig)

                # Page 2: CCDF(|g|) vs CCDF(|U|)
                fig = plt.figure(figsize=(8.5, 5.5))
                ax = fig.add_subplot(111)
                _plot_ccdf_pair(ax, g, u, title=f"Component tails (iter={it}, group={grp})")
                combined.savefig(fig, bbox_inches="tight")
                if save_per_group:
                    fig.savefig(os.path.join(out_dir, f"sse__iter{it:07d}__{grp}__ccdf.pdf"), bbox_inches="tight")
                plt.close(fig)

    print(f"[ok] wrote {combined_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--mode", choices=["latest", "iters", "all"], default="latest")
    ap.add_argument("--iters", default="", help="comma-separated iters if mode=iters")
    ap.add_argument("--groups", default="", help="comma-separated groups; default all")
    ap.add_argument("--save_per_group", action="store_true", help="also save per-group PDFs")
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or os.path.join(run_dir, "figures_sse")
    groups = [g.strip() for g in args.groups.split(",") if g.strip()] or None

    files = _list_npz(run_dir)

    if args.mode == "latest":
        chosen = [files[-1]]
    elif args.mode == "all":
        chosen = files
    else:
        if not args.iters.strip():
            raise ValueError("--mode iters requires --iters")
        iters = [int(x.strip()) for x in args.iters.split(",") if x.strip()]
        have = { _parse_iter(p): p for p in files }
        chosen = []
        for it in iters:
            if it not in have:
                raise FileNotFoundError(f"sse_iter{it:07d}.npz not found in {run_dir}/sse/")
            chosen.append(have[it])

    for p in chosen:
        plot_one(p, out_dir, groups=groups, save_per_group=args.save_per_group)


if __name__ == "__main__":
    main()
    