#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/plot_sse.py

Plots SSE diagnostics saved by c1bench/sse_diag.py.

Fixes:
- g_abs and u_abs arrays may have slightly different lengths (due to per-tensor sampling/concatenation).
  We now align by truncating to min(len(g), len(u)) BEFORE any boolean masks, so broadcasting errors cannot happen.
- Drops the very first data point (often lr=0 at iter=0), which can create log-scale / fit artifacts.

Outputs (default):
  <run_dir>/figures_tail/
    - sse__iterXXXXXXX__<group>.pdf (per group per iter)
    - sse__iterXXXXXXX__ALL_GROUPS.pdf (combined multipage per iter)

Usage:
  python tools/plot_sse.py --run_dir out/A0_adamw_base --mode iters --iters 700,1800,1900
"""

import argparse
import glob
import os
import re
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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


def _align_pair(g: np.ndarray, u: np.ndarray, drop_first: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Align g and u arrays by truncating to same length (min len).
    Optionally drop the first aligned point (to remove iter=0 artifacts).
    """
    n = min(g.size, u.size)
    if n <= 0:
        return g[:0], u[:0]
    g2 = g[:n]
    u2 = u[:n]
    if drop_first and n >= 2:
        g2 = g2[1:]
        u2 = u2[1:]
    return g2, u2


def _binned_median(x: np.ndarray, y: np.ndarray, nbins: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin x in log-space, compute median and IQR for y in each bin.
    Returns: bin_centers, med, iqr (q75-q25).
    """
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[m]
    y = y[m]
    if x.size < 10:
        return np.array([]), np.array([]), np.array([])

    xmin = np.quantile(x, 0.01)
    xmax = np.quantile(x, 0.99)
    xmin = max(xmin, x.min())
    xmax = max(xmax, xmin * 1.001)

    edges = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)
    idx = np.digitize(x, edges) - 1

    centers = []
    meds = []
    iqrs = []
    for b in range(nbins):
        mb = idx == b
        if mb.sum() < 20:
            continue
        yy = y[mb]
        centers.append(np.sqrt(edges[b] * edges[b + 1]))
        q25 = np.quantile(yy, 0.25)
        q50 = np.quantile(yy, 0.50)
        q75 = np.quantile(yy, 0.75)
        meds.append(q50)
        iqrs.append(q75 - q25)
    return np.array(centers), np.array(meds), np.array(iqrs)


def _fit_beta_loglog(x: np.ndarray, y: np.ndarray, qmin: float = 0.90, qmax: float = 0.995) -> Optional[tuple[float, float]]:
    """
    Fit y ≈ A * x^beta on the tail by linear regression on log-log,
    using x-range between quantiles [qmin, qmax].
    Returns (beta, r2) or None.
    """
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[m]
    y = y[m]
    if x.size < 200:
        return None

    lo = np.quantile(x, qmin)
    hi = np.quantile(x, qmax)
    mm = (x >= lo) & (x <= hi)
    if mm.sum() < 50:
        return None

    X = np.log(x[mm])
    Y = np.log(y[mm])
    slope, intercept = np.polyfit(X, Y, 1)  # Y = intercept + slope*X
    Yhat = intercept + slope * X
    ss_res = float(np.sum((Y - Yhat) ** 2))
    ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    beta = float(slope)
    return beta, r2


def plot_one(npz_path: str, out_dir: str, groups: Optional[List[str]] = None, nbins: int = 30) -> None:
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

                # Align lengths safely and drop first point (iter=0 artifacts)
                g, u = _align_pair(g, u, drop_first=True)

                # Filter valid positive pairs (after alignment!)
                m = np.isfinite(g) & np.isfinite(u) & (g > 0) & (u > 0)
                g = g[m]
                u = u[m]
                if g.size < 200:
                    continue

                # Optional downsample for scatter visibility
                if g.size > 200_000:
                    rng = np.random.default_rng(0)
                    idx = rng.choice(g.size, size=200_000, replace=False)
                    g_s = g[idx]
                    u_s = u[idx]
                else:
                    g_s = g
                    u_s = u

                # Binned median curve
                bx, by, biqr = _binned_median(g, u, nbins=nbins)

                # Tail fit beta
                fit = _fit_beta_loglog(g, u, qmin=0.90, qmax=0.995)
                if fit is not None:
                    beta, r2 = fit
                    fit_txt = f"fit on tail: beta={beta:.3g}, R²={r2:.3f}"
                else:
                    beta, r2 = np.nan, np.nan
                    fit_txt = "fit on tail: n/a"

                fig, ax = plt.subplots(figsize=(7.5, 5.2))
                ax.scatter(g_s, u_s, s=2, alpha=0.25, label="samples (|grad|, |U|)")
                if bx.size > 0:
                    ax.plot(bx, by, linewidth=2.0, label="binned median")

                # Add fitted line for visualization if available
                if fit is not None and bx.size > 0:
                    # build a fit line in x-range of binned curve
                    xx = np.logspace(np.log10(bx.min()), np.log10(bx.max()), 200)
                    # y = A x^beta; estimate A from median point
                    A = by[len(by)//2] / (bx[len(bx)//2] ** beta)
                    yy = A * xx ** beta
                    ax.plot(xx, yy, linewidth=2.0, label="power fit (visual)")

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel(r"$\|g\|$ samples (abs coord)")
                ax.set_ylabel(r"$\|U\|$ samples (abs coord)")
                ax.set_title(f"SSE scatter (iter={it}, group={grp})\n{fit_txt}\n(first point dropped)")
                ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
                ax.legend(loc="best")

                out_path = os.path.join(out_dir, f"sse__iter{it:07d}__{grp}.pdf")
                fig.savefig(out_path, bbox_inches="tight")
                combined.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    print(f"[ok] wrote {combined_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--mode", choices=["latest", "iters", "all"], default="latest")
    ap.add_argument("--iters", default="", help="comma-separated iters if mode=iters")
    ap.add_argument("--groups", default="", help="comma-separated groups; default all")
    ap.add_argument("--nbins", type=int, default=30, help="bins for binned median")
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or os.path.join(run_dir, "figures_tail")
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
        plot_one(p, out_dir, groups=groups, nbins=args.nbins)


if __name__ == "__main__":
    main()
    