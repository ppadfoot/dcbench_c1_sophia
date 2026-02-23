#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/plot_tail_metrics.py  (paper-ready labels)

What changed vs previous versions:
- Titles are short and clean.
- All fit stats (alpha, R^2, lambda) and mask params are shown in small annotation boxes.
- Underscores in group/series names are replaced for readability in titles.
- CCDF panel overlays fitted curves and shows a compact legend.
- B_eff computed tail-only + trimmed edges (avoids the “tooth” near mask boundary).

Supports series keys:
- grad, noise, delta, mean -> grad_abs__G, noise_abs__G, delta_abs__G, mean_abs__G
- step, step_noise, step_delta, step_mean -> step_abs__G, step_noise_abs__G, step_delta_abs__G, step_mean_abs__G

Outputs:
- <run_dir>/figures_tail/tail_tails_iterXXXXXXX__ALL_PLOTS.pdf (bundle)
- <run_dir>/figures_tail/tail_iterXXXXXXX__<series>__<group>.pdf (single pages)
- <run_dir>/figures_tail/PAPER_tail_iterXXXXXXX__<series>.pdf (paper mode)

Example (paper mode):
  RUN=E4_layer_bucket_v2
  python tools/plot_tail_metrics.py --run_dir out/$RUN --mode iters --iters 1800 \
    --paper --paper_groups all,attn_late,mlp_late --series delta
"""

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# -------------------------
# Helpers
# -------------------------

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def parse_iter(npz_path: str) -> int:
    base = os.path.basename(npz_path)
    m = re.search(r"iter(\d+)\.npz", base)
    return int(m.group(1)) if m else -1

def list_npz(run_dir: str, npz_glob: str) -> List[str]:
    patt = os.path.join(run_dir, "tails", npz_glob)
    files = sorted(glob.glob(patt), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No files matching {npz_glob!r} found in {run_dir}/tails")
    return files

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def safe_positive(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    return x

def pretty_name(s: str) -> str:
    # make labels less “underscore-y” for paper
    return s.replace("_", "-")

def fmt_float(x: float, nd: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.{nd}g}"

def fmt_lambda(lam: float) -> str:
    if lam is None or not np.isfinite(lam):
        return "n/a"
    if np.isinf(lam):
        return "∞"
    return f"{lam:.3g}"

# -------------------------
# Mapping from series name to npz key prefix
# -------------------------

SERIES_TO_PREFIX = {
    "grad": "grad_abs",
    "noise": "noise_abs",
    "delta": "delta_abs",
    "mean": "mean_abs",
    "step": "step_abs",
    "step_noise": "step_noise_abs",
    "step_delta": "step_delta_abs",
    "step_mean": "step_mean_abs",
}

# -------------------------
# CCDF / grid / fits
# -------------------------

def ccdf_on_grid(x_sorted: np.ndarray, x_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = x_sorted.size
    idx = np.searchsorted(x_sorted, x_grid, side="right")
    ccdf = (n - idx) / n
    ccdf = np.clip(ccdf, 1.0 / (n + 1.0), 1.0)
    tail_count = n * ccdf
    return ccdf, tail_count

def make_grid_logspace(x: np.ndarray, grid_points: int) -> np.ndarray:
    x = safe_positive(x)
    lo = max(np.quantile(x, 0.001), x.min())
    hi = max(np.quantile(x, 0.99999), lo * 1.001)
    return np.logspace(np.log10(lo), np.log10(hi), grid_points)

def make_grid_quantile(x_sorted: np.ndarray, grid_points: int) -> np.ndarray:
    ps = np.linspace(0.001, 0.99999, grid_points)
    xg = np.quantile(x_sorted, ps)
    xg = np.unique(xg)
    if xg.size < 2:
        lo = max(np.quantile(x_sorted, 0.001), x_sorted.min())
        hi = max(np.quantile(x_sorted, 0.99999), lo * 1.001)
        xg = np.logspace(np.log10(lo), np.log10(hi), grid_points)
    return xg

def fit_powerlaw_logccdf(logx: np.ndarray, logccdf: np.ndarray) -> Tuple[float, float, float]:
    if logx.size < 20:
        return float("nan"), float("nan"), float("nan")
    b, a = np.polyfit(logx, logccdf, 1)  # logccdf = a + b*logx
    yhat = a + b * logx
    ss_res = np.sum((logccdf - yhat) ** 2)
    ss_tot = np.sum((logccdf - logccdf.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    alpha = -b
    return float(alpha), float(a), float(r2)

def fit_tempered_logccdf(logx: np.ndarray, x: np.ndarray, logccdf: np.ndarray) -> Tuple[float, float, float, float, float]:
    if logx.size < 20:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    A = np.stack([np.ones_like(logx), logx, x], axis=1)
    coef, *_ = np.linalg.lstsq(A, logccdf, rcond=None)
    c0, b1, b2 = coef
    yhat = A @ coef
    ss_res = np.sum((logccdf - yhat) ** 2)
    ss_tot = np.sum((logccdf - logccdf.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    alpha = -b1
    lam = float("inf")
    if b2 < 0:
        lam = float(-1.0 / b2)
    return float(alpha), lam, float(c0), float(b2), float(r2)

# -------------------------
# B_eff (tail-only + trim)
# -------------------------

def local_slope_ls(logx: np.ndarray, logy: np.ndarray, window: int) -> np.ndarray:
    n = logx.size
    w = max(1, window // 2)
    slopes = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        j0 = max(0, i - w)
        j1 = min(n, i + w + 1)
        X = logx[j0:j1]
        Y = logy[j0:j1]
        if X.size < 2:
            continue
        Xc = X - X.mean()
        denom = np.sum(Xc * Xc)
        if denom <= 0:
            continue
        slopes[i] = np.sum(Xc * (Y - Y.mean())) / denom
    return slopes

@dataclass
class TailView:
    x: np.ndarray
    ccdf: np.ndarray
    tail_count: np.ndarray
    n: int

def compute_tail_view(x: np.ndarray, grid_mode: str, grid_points: int) -> TailView:
    x = safe_positive(x)
    x_sorted = np.sort(x)
    if grid_mode == "quantile":
        x_grid = make_grid_quantile(x_sorted, grid_points)
    else:
        x_grid = make_grid_logspace(x, grid_points)
    ccdf, tail_count = ccdf_on_grid(x_sorted, x_grid)
    return TailView(x=x_grid, ccdf=ccdf, tail_count=tail_count, n=x_sorted.size)

def tail_mask(view: TailView, min_tail_count: int, ccdf_max: float) -> np.ndarray:
    return (
        np.isfinite(view.ccdf)
        & (view.ccdf > 0)
        & (view.tail_count >= min_tail_count)
        & (view.ccdf <= ccdf_max)
        & (view.x > 0)
    )

def compute_beff_tail_only(view: TailView, mask: np.ndarray, window: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    idx = np.where(mask)[0]
    # need enough points for stable slope
    if idx.size < max(40, window + 10):
        return None, None, None

    x_tail = view.x[idx]
    ccdf_tail = view.ccdf[idx]
    logx = np.log(x_tail)
    logy = np.log(np.clip(ccdf_tail, 1e-300, 1.0))

    slopes = local_slope_ls(logx, logy, window=window)
    beff = -slopes

    # trim edges to avoid boundary artifacts (the "tooth")
    t = window // 2
    if idx.size <= 2 * t + 20:
        t = max(0, (idx.size - 20) // 2)
    if t > 0:
        x_tail = x_tail[t:-t]
        beff = beff[t:-t]

    m = np.isfinite(beff)
    x_tail = x_tail[m]
    beff = beff[m]
    if x_tail.size < 10:
        return None, None, None

    med = float(np.median(beff))
    return x_tail, beff, med

# -------------------------
# Density (kept for bundle/appendix)
# -------------------------

def density_log_binned(x: np.ndarray, nbins: int) -> Tuple[np.ndarray, np.ndarray]:
    x = safe_positive(x)
    if x.size < 2:
        return np.array([]), np.array([])
    lo = max(np.quantile(x, 0.001), x.min())
    hi = max(np.quantile(x, 0.99999), lo * 1.001)
    edges = np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
    counts, _ = np.histogram(x, bins=edges)
    widths = edges[1:] - edges[:-1]
    centers = np.sqrt(edges[:-1] * edges[1:])
    dens = counts / (x.size * widths)
    m = dens > 0
    return centers[m], dens[m]

# -------------------------
# Plot formatting helpers
# -------------------------

def add_box(ax, text: str, loc: str = "tr"):
    # loc: "tr" top-right, "tl", "br", "bl"
    x = 0.98 if "r" in loc else 0.02
    y = 0.98 if "t" in loc else 0.02
    ha = "right" if "r" in loc else "left"
    va = "top" if "t" in loc else "bottom"
    ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va=va,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
        fontsize=10,
    )

def plot_ccdf_with_fits(ax, view: TailView, x_fit: np.ndarray,
                        alpha_pw: float, a_pw: float, r2_pw: float,
                        alpha_tp: float, lam_tp: float, c0_tp: float, b2_tp: float, r2_tp: float):
    ax.plot(view.x, view.ccdf, linewidth=2, label="CCDF")

    # overlay fits only on x_fit
    if x_fit is not None and x_fit.size > 0:
        if np.isfinite(alpha_pw) and np.isfinite(a_pw):
            logx = np.log(x_fit)
            logcc = a_pw + (-alpha_pw) * logx
            ax.plot(x_fit, np.exp(logcc), linewidth=2, label="Power-law fit")
        if np.isfinite(alpha_tp) and np.isfinite(c0_tp) and np.isfinite(b2_tp):
            logx = np.log(x_fit)
            logcc = c0_tp + (-alpha_tp) * logx + b2_tp * x_fit
            ax.plot(x_fit, np.exp(logcc), linewidth=2, label="Tempered fit")
        # x_min line
        ax.axvline(float(x_fit.min()), linestyle="--", linewidth=1.0, color="gray")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("x (log)")
    ax.set_ylabel("CCDF (log)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="best", fontsize=10)

    box = (
        f"Power-law:  alpha={fmt_float(alpha_pw)}  R²={fmt_float(r2_pw,3)}\n"
        f"Tempered:   alpha={fmt_float(alpha_tp)}  lambda={fmt_lambda(lam_tp)}  R²={fmt_float(r2_tp,3)}\n"
        f"x_min={fmt_float(float(x_fit.min()) if x_fit is not None and x_fit.size>0 else np.nan)}"
    )
    add_box(ax, box, loc="tr")

def plot_beff(ax, x_tail: np.ndarray, beff: np.ndarray, med: float,
              min_tail_count: int, ccdf_max: float, window: int):
    ax.plot(x_tail, beff, linewidth=2, label=r"$B_{\mathrm{eff}}(x)$")
    ax.axhline(med, linestyle="--", linewidth=1.2, label=f"median={med:.3g}")

    ax.set_xscale("log")
    ax.set_xlabel("x (log)")
    ax.set_ylabel(r"$B_{\mathrm{eff}}$")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="best", fontsize=10)

    box = f"mask: n·CCDF ≥ {min_tail_count}, CCDF ≤ {ccdf_max}\nwindow={window} (tail-only, trimmed)"
    add_box(ax, box, loc="tr")

# -------------------------
# Page builders
# -------------------------

def plot_triplet_page(
    pdf: PdfPages,
    single_path: str,
    it: int,
    group: str,
    series: str,
    x: np.ndarray,
    grid_mode: str,
    grid_points: int,
    density_bins: int,
    beff_window: int,
    min_tail_count: int,
    ccdf_max: float,
):
    x = safe_positive(x)
    if x.size < 500:
        return

    view = compute_tail_view(x, grid_mode=grid_mode, grid_points=grid_points)
    m = tail_mask(view, min_tail_count=min_tail_count, ccdf_max=ccdf_max)
    x_tail, beff_tail, med = compute_beff_tail_only(view, m, window=beff_window)

    # fit on SAME tail-only points
    alpha_pw = a_pw = r2_pw = float("nan")
    alpha_tp = lam_tp = c0_tp = b2_tp = r2_tp = float("nan")
    x_fit = np.array([])

    if x_tail is not None and x_tail.size >= 30:
        x_fit = x_tail
        idx = np.searchsorted(view.x, x_fit)
        idx = np.clip(idx, 0, view.x.size - 1)
        ccdf_fit = view.ccdf[idx]
        logx = np.log(x_fit)
        logcc = np.log(np.clip(ccdf_fit, 1e-300, 1.0))
        alpha_pw, a_pw, r2_pw = fit_powerlaw_logccdf(logx, logcc)
        alpha_tp, lam_tp, c0_tp, b2_tp, r2_tp = fit_tempered_logccdf(logx, x_fit, logcc)

    dx, dy = density_log_binned(x, nbins=density_bins)

    fig = plt.figure(figsize=(8.2, 10.5))
    fig.suptitle(f"iter={it}   series={pretty_name(series)}   group={pretty_name(group)}", y=0.995, fontsize=14)

    ax1 = fig.add_subplot(3, 1, 1)
    if dx.size > 0:
        ax1.plot(dx, dy, linewidth=2)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("x (log)")
    ax1.set_ylabel("density (log)")
    ax1.set_title("Density (log-binned)", fontsize=12)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_title("CCDF (log–log) + fits", fontsize=12)
    plot_ccdf_with_fits(ax2, view, x_fit,
                        alpha_pw, a_pw, r2_pw,
                        alpha_tp, lam_tp, c0_tp, b2_tp, r2_tp)

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_title(r"$B_{\mathrm{eff}}(x)$ on tail (stable view)", fontsize=12)
    if x_tail is not None and beff_tail is not None:
        plot_beff(ax3, x_tail, beff_tail, med, min_tail_count, ccdf_max, beff_window)
    else:
        add_box(ax3, "No stable tail after mask.\nTry smaller min_tail_count or larger samples.", loc="tl")
        ax3.set_xscale("log")
        ax3.set_xlabel("x (log)")
        ax3.set_ylabel(r"$B_{\mathrm{eff}}$")
        ax3.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)

    fig.savefig(single_path, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

def plot_paper_pdf(
    out_dir: str,
    npz_path: str,
    it: int,
    groups: List[str],
    series: str,
    grid_mode: str,
    grid_points: int,
    beff_window: int,
    min_tail_count: int,
    ccdf_max: float,
):
    ensure_dir(out_dir)
    prefix = SERIES_TO_PREFIX[series]

    paper_path = os.path.join(out_dir, f"PAPER_tail_iter{it:07d}__{series}.pdf")
    with np.load(npz_path) as z, PdfPages(paper_path) as pdf:
        for grp in groups:
            key = f"{prefix}__{grp}"
            if key not in z:
                continue
            x = safe_positive(z[key])
            if x.size < 500:
                continue

            view = compute_tail_view(x, grid_mode=grid_mode, grid_points=grid_points)
            m = tail_mask(view, min_tail_count=min_tail_count, ccdf_max=ccdf_max)
            x_tail, beff_tail, med = compute_beff_tail_only(view, m, window=beff_window)

            alpha_pw = a_pw = r2_pw = float("nan")
            alpha_tp = lam_tp = c0_tp = b2_tp = r2_tp = float("nan")
            x_fit = np.array([])

            if x_tail is not None and x_tail.size >= 30:
                x_fit = x_tail
                idx = np.searchsorted(view.x, x_fit)
                idx = np.clip(idx, 0, view.x.size - 1)
                ccdf_fit = view.ccdf[idx]
                logx = np.log(x_fit)
                logcc = np.log(np.clip(ccdf_fit, 1e-300, 1.0))
                alpha_pw, a_pw, r2_pw = fit_powerlaw_logccdf(logx, logcc)
                alpha_tp, lam_tp, c0_tp, b2_tp, r2_tp = fit_tempered_logccdf(logx, x_fit, logcc)

            fig = plt.figure(figsize=(8.2, 7.2))
            fig.suptitle(f"iter={it}   series={pretty_name(series)}   group={pretty_name(grp)}", y=0.995, fontsize=14)

            ax1 = fig.add_subplot(2, 1, 1)
            ax1.set_title("CCDF (log–log) + fits", fontsize=12)
            plot_ccdf_with_fits(ax1, view, x_fit,
                                alpha_pw, a_pw, r2_pw,
                                alpha_tp, lam_tp, c0_tp, b2_tp, r2_tp)

            ax2 = fig.add_subplot(2, 1, 2)
            ax2.set_title(r"$B_{\mathrm{eff}}(x)$ on tail (tail-only + trim)", fontsize=12)
            if x_tail is not None and beff_tail is not None:
                plot_beff(ax2, x_tail, beff_tail, med, min_tail_count, ccdf_max, beff_window)
            else:
                add_box(ax2, "No stable tail after mask.\nTry smaller min_tail_count or larger samples.", loc="tl")
                ax2.set_xscale("log")
                ax2.set_xlabel("x (log)")
                ax2.set_ylabel(r"$B_{\mathrm{eff}}$")
                ax2.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"[ok] wrote {paper_path}")

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--npz_glob", default="tails_iter*.npz")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--mode", choices=["latest", "iters", "all"], default="latest")
    ap.add_argument("--iters", default="")
    ap.add_argument("--series", default="grad,noise,delta,mean")
    ap.add_argument("--groups", default="")

    ap.add_argument("--grid_mode", choices=["quantile", "logspace"], default="quantile")
    ap.add_argument("--grid_points", type=int, default=450)
    ap.add_argument("--density_bins", type=int, default=90)

    ap.add_argument("--beff_window", type=int, default=31)
    ap.add_argument("--beff_min_tail_count", type=int, default=500)
    ap.add_argument("--beff_ccdf_max", type=float, default=0.3)

    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--paper_groups", default="all,attn_late,mlp_late")
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or os.path.join(run_dir, "figures_tail")
    ensure_dir(out_dir)

    files = list_npz(run_dir, args.npz_glob)

    # choose npz files
    if args.mode == "latest":
        chosen = [files[-1]]
    elif args.mode == "all":
        chosen = files
    else:
        iters = [int(x.strip()) for x in args.iters.split(",") if x.strip()]
        have = {parse_iter(p): p for p in files}
        chosen = []
        for it in iters:
            if it not in have:
                print(f"[warn] iter {it} not found for glob={args.npz_glob}")
                continue
            chosen.append(have[it])
        if not chosen:
            raise FileNotFoundError("No requested iters found.")

    series_list = [s.strip() for s in args.series.split(",") if s.strip()]
    for s in series_list:
        if s not in SERIES_TO_PREFIX:
            raise ValueError(f"Unknown series {s}. Allowed: {sorted(SERIES_TO_PREFIX.keys())}")

    group_filter = [g.strip() for g in args.groups.split(",") if g.strip()] if args.groups.strip() else None

    for npz_path in chosen:
        it = parse_iter(npz_path)
        with np.load(npz_path) as z:
            keys = set(z.keys())
            groups_found = set()
            for s in series_list:
                pref = SERIES_TO_PREFIX[s]
                for k in keys:
                    if k.startswith(pref + "__"):
                        groups_found.add(k.split("__", 1)[1])
            groups = sorted(groups_found, key=natural_key)
            if group_filter is not None:
                groups = [g for g in groups if g in group_filter]

        if args.paper:
            pgroups = [g.strip() for g in args.paper_groups.split(",") if g.strip()]
            for s in series_list:
                plot_paper_pdf(
                    out_dir=out_dir,
                    npz_path=npz_path,
                    it=it,
                    groups=pgroups,
                    series=s,
                    grid_mode=args.grid_mode,
                    grid_points=args.grid_points,
                    beff_window=args.beff_window,
                    min_tail_count=args.beff_min_tail_count,
                    ccdf_max=args.beff_ccdf_max,
                )
            continue

        bundle_path = os.path.join(out_dir, f"tail_tails_iter{it:07d}__ALL_PLOTS.pdf")
        with np.load(npz_path) as z, PdfPages(bundle_path) as pdf:
            for grp in groups:
                for s in series_list:
                    pref = SERIES_TO_PREFIX[s]
                    key = f"{pref}__{grp}"
                    if key not in z:
                        continue
                    arr = safe_positive(z[key])
                    if arr.size < 500:
                        continue
                    single_path = os.path.join(out_dir, f"tail_iter{it:07d}__{s}__{grp}.pdf")
                    plot_triplet_page(
                        pdf=pdf,
                        single_path=single_path,
                        it=it,
                        group=grp,
                        series=s,
                        x=arr,
                        grid_mode=args.grid_mode,
                        grid_points=args.grid_points,
                        density_bins=args.density_bins,
                        beff_window=args.beff_window,
                        min_tail_count=args.beff_min_tail_count,
                        ccdf_max=args.beff_ccdf_max,
                    )

        print(f"[ok] wrote {bundle_path}")


if __name__ == "__main__":
    main()

