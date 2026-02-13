#!/usr/bin/env python3
"""tools/plot_tail_metrics.py

Plot heavy-tail diagnostics logged by :class:`c1bench.tail_diag.TailDiagnostics`.

What this script does
---------------------
Given a run directory that contains ``tails/tails_iterXXXXXXX.npz`` files, this script:

1) Makes *individual* PDFs for each (group × series × plot-kind).
2) Also makes a *single bundled PDF* per iteration that contains all plots.
3) Optionally estimates tail exponents using two standard approaches:
   - Clauset-style continuous Pareto fit (KS-min over candidate xmin)
   - Hill plot (top-k estimator) for sanity/robustness

Important conventions
---------------------
We work with positive variables, typically absolute values:
  - |g|         (one-batch gradient sample)
  - |noise|     where noise = g - mean(g over K batches)
  - |delta|     where delta = g_a - g_b (two independent batches; simple noise proxy)

If the *tail* is Pareto:
  CCDF:   P(X > x) ~ x^{-B}
  PDF:    p(x)     ~ x^{-(B+1)}

So:
  - B is the slope magnitude on a log–log CCDF plot.
  - B_eff(x) is a *local* slope estimate of the log–log CCDF.

Why B_eff is noisy
------------------
B_eff involves a numerical derivative of log(CCDF). Derivatives amplify noise.
We therefore compute B_eff using a sliding-window linear regression in log–log space
(``--beff_window``). Increasing the window smooths B_eff.

Typical usage
-------------

Plot the latest diagnostics (and bundle them):

  python tools/plot_tail_metrics.py --run_dir out/tail_k32_cosine

Plot specific iterations:

  python tools/plot_tail_metrics.py --run_dir out/tail_k32_cosine \
    --mode iters --iters 200,400,800,1200,1600,1900

Make an "alpha over time" summary (requires multiple iterations selected):

  python tools/plot_tail_metrics.py --run_dir out/tail_k32_cosine --mode all --over_time
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# -----------------------------
# Utilities
# -----------------------------


def _clean_pos(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    return x


def _geomean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(a * b)


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, 1e-300, None))


def _extract_iter(path: Path) -> int:
    m = re.search(r"tails_iter(\d+)\.npz$", path.name)
    return int(m.group(1)) if m else -1


def _list_tail_npz_files(tails_dir: Path) -> List[Path]:
    return sorted(tails_dir.glob("tails_iter*.npz"))


def _pick_files(files: Sequence[Path], *, mode: str, iters: Sequence[int]) -> List[Path]:
    if mode == "latest":
        return [max(files, key=_extract_iter)] if files else []
    if mode == "all":
        return list(files)
    if mode == "iters":
        want = set(int(x) for x in iters)
        return [f for f in files if _extract_iter(f) in want]
    raise ValueError(f"Unknown mode: {mode}")


def _available_groups(npz: Dict[str, np.ndarray]) -> List[str]:
    groups = []
    for k in npz.keys():
        if k.startswith("grad_abs__"):
            groups.append(k.split("__", 1)[1])
    return sorted(set(groups))


def _available_series_for_group(npz: Dict[str, np.ndarray], group: str) -> List[str]:
    series = []
    for s in ["grad", "noise", "delta", "mean"]:
        key = f"{s}_abs__{group}"
        if key in npz:
            series.append(s)
    return series


# -----------------------------
# Core plotting primitives
# -----------------------------


def log_hist_density(x: np.ndarray, *, bins: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    """Log-binned histogram density estimate for positive x."""
    x = _clean_pos(x)
    if x.size < 20:
        return np.array([]), np.array([])
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if not (xmax > xmin > 0):
        return np.array([]), np.array([])
    edges = np.logspace(np.log10(xmin), np.log10(xmax), bins + 1)
    counts, edges = np.histogram(x, bins=edges)
    widths = edges[1:] - edges[:-1]
    centers = _geomean(edges[1:], edges[:-1])
    dens = counts.astype(np.float64) / (x.size * widths)
    m = dens > 0
    return centers[m], dens[m]


def ccdf_on_log_grid(x: np.ndarray, *, bins: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """Compute CCDF on a log-spaced x-grid."""
    x = _clean_pos(x)
    if x.size < 20:
        return np.array([]), np.array([])
    xs = np.sort(x)
    n = xs.size
    xmin, xmax = float(xs[0]), float(xs[-1])
    if not (xmax > xmin > 0):
        return np.array([]), np.array([])
    grid = np.logspace(np.log10(xmin), np.log10(xmax), bins)
    idx = np.searchsorted(xs, grid, side="right")
    ccdf = (n - idx).astype(np.float64) / float(n)
    m = (ccdf > 0) & np.isfinite(ccdf)
    return grid[m], ccdf[m]


def beff_from_ccdf_window_reg(
    x: np.ndarray,
    ccdf: np.ndarray,
    *,
    window: int = 21,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute B_eff(x) via sliding-window linear regression in log–log space.

    Returns (x_mid, B_eff).
    """
    if x.size < 10:
        return np.array([]), np.array([])
    lx = _safe_log(x)
    ly = _safe_log(ccdf)
    n = lx.size
    w = int(max(5, window))
    if w % 2 == 0:
        w += 1
    half = w // 2
    if n <= w:
        # fall back to simple diff
        dx = np.diff(lx)
        dy = np.diff(ly)
        m = dx != 0
        beff = -(dy[m] / dx[m])
        xmid = np.exp(0.5 * (lx[1:][m] + lx[:-1][m]))
        return xmid, beff

    out_x: List[float] = []
    out_b: List[float] = []
    for i in range(half, n - half):
        xx = lx[i - half : i + half + 1]
        yy = ly[i - half : i + half + 1]
        # y = a + b x
        b, a = np.polyfit(xx, yy, deg=1)
        out_x.append(float(np.exp(lx[i])))
        out_b.append(float(-b))
    return np.asarray(out_x), np.asarray(out_b)


# -----------------------------
# Tail exponent estimation
# -----------------------------


@dataclass
class ParetoFit:
    xmin: float
    a_pdf: float  # PDF exponent: p(x) ~ x^{-a_pdf}
    B_ccdf: float  # CCDF exponent: P(X>x) ~ x^{-B_ccdf}  (B_ccdf = a_pdf - 1)
    ks: float
    n_tail: int
    se_B: float


def clauset_pareto_fit(
    x: np.ndarray,
    *,
    xmin_q_lo: float = 0.90,
    xmin_q_hi: float = 0.995,
    xmin_grid: int = 50,
    min_tail: int = 1000,
) -> Optional[ParetoFit]:
    """Clauset-style continuous Pareto fit.

    We scan candidate xmin values (quantile grid) and pick the xmin that minimizes KS distance.
    Uses continuous Pareto MLE for the PDF exponent:
        a = 1 + n / sum log(x_i/xmin)
    Then CCDF exponent is B = a - 1.

    Returns None if we cannot fit stably (too few points, degenerate data, etc.).
    """
    x = _clean_pos(x)
    if x.size < max(100, min_tail):
        return None

    # Candidate xmins from quantiles (unique, sorted)
    q_lo = float(np.clip(xmin_q_lo, 0.0, 1.0))
    q_hi = float(np.clip(xmin_q_hi, q_lo + 1e-6, 1.0))
    qs = np.linspace(q_lo, q_hi, int(max(5, xmin_grid)))
    xmins = np.unique(np.quantile(x, qs))
    xmins = xmins[np.isfinite(xmins)]
    xmins = xmins[xmins > 0]
    if xmins.size == 0:
        return None

    best: Optional[ParetoFit] = None

    x_sorted = np.sort(x)
    n_total = x_sorted.size

    for xmin in xmins:
        # tail selection
        start = int(np.searchsorted(x_sorted, xmin, side="left"))
        tail = x_sorted[start:]
        n = tail.size
        if n < min_tail:
            continue
        if not np.all(tail >= xmin):
            continue

        # MLE for PDF exponent
        logs = np.log(tail / float(xmin))
        s = float(np.sum(logs))
        if not np.isfinite(s) or s <= 0:
            continue
        a_pdf = 1.0 + n / s
        B = a_pdf - 1.0
        if not (np.isfinite(a_pdf) and a_pdf > 1.0 and np.isfinite(B) and B > 0):
            continue

        # KS distance on tail (continuous Pareto CDF)
        # model CDF: F(x)=1-(x/xmin)^{-B}
        emp_cdf = (np.arange(1, n + 1) / float(n)).astype(np.float64)
        model_cdf = 1.0 - (tail / float(xmin)) ** (-B)
        ks = float(np.max(np.abs(emp_cdf - model_cdf)))
        if not np.isfinite(ks):
            continue

        se_B = float(B / math.sqrt(n))

        fit = ParetoFit(xmin=float(xmin), a_pdf=float(a_pdf), B_ccdf=float(B), ks=float(ks), n_tail=int(n), se_B=se_B)
        if (best is None) or (fit.ks < best.ks):
            best = fit

    return best


def hill_alpha(x: np.ndarray, k: int) -> Optional[float]:
    """Hill estimator for CCDF exponent (tail index) on positive samples.

    For Pareto tail with CCDF exponent B, Hill estimates B.
    """
    x = _clean_pos(x)
    n = x.size
    k = int(k)
    if n < 50 or k <= 5 or k >= n:
        return None
    xs = np.sort(x)[::-1]  # descending
    xk1 = xs[k]
    if not (np.isfinite(xk1) and xk1 > 0):
        return None
    top = xs[:k]
    logs = np.log(np.clip(top / xk1, 1e-300, None))
    m = float(np.mean(logs))
    if not (np.isfinite(m) and m > 0):
        return None
    return float(1.0 / m)


def hill_curve(x: np.ndarray, *, k_min: int = 50, k_max: int = 5000, n_points: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    x = _clean_pos(x)
    n = x.size
    if n < 200:
        return np.array([]), np.array([])
    k_max = int(min(k_max, n - 1))
    k_min = int(max(10, k_min))
    if k_max <= k_min + 5:
        return np.array([]), np.array([])
    ks = np.unique(np.round(np.logspace(np.log10(k_min), np.log10(k_max), n_points)).astype(int))
    alphas = []
    ks2 = []
    for k in ks:
        a = hill_alpha(x, int(k))
        if a is None:
            continue
        ks2.append(int(k))
        alphas.append(float(a))
    if not ks2:
        return np.array([]), np.array([])
    return np.asarray(ks2), np.asarray(alphas)


@dataclass
class CCDFLinearFit:
    B: float
    intercept: float
    r2: float


@dataclass
class CCDTemperedFit:
    B: float
    lam: float
    intercept: float
    r2: float


def _linfit(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return (beta, r2)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta, float(r2)


def fit_ccdf_powerlaw(grid: np.ndarray, ccdf: np.ndarray, *, xmin: float) -> Optional[CCDFLinearFit]:
    m = (grid >= float(xmin)) & np.isfinite(ccdf) & (ccdf > 0)
    if np.sum(m) < 10:
        return None
    x = grid[m]
    y = np.log(ccdf[m])
    lx = np.log(x)
    # y = c - B log x
    X = np.stack([np.ones_like(lx), -lx], axis=1)
    beta, r2 = _linfit(y, X)
    c, B = float(beta[0]), float(beta[1])
    if not (np.isfinite(B) and B > 0):
        return None
    return CCDFLinearFit(B=B, intercept=c, r2=r2)


def fit_ccdf_tempered(grid: np.ndarray, ccdf: np.ndarray, *, xmin: float) -> Optional[CCDTemperedFit]:
    m = (grid >= float(xmin)) & np.isfinite(ccdf) & (ccdf > 0)
    if np.sum(m) < 20:
        return None
    x = grid[m]
    y = np.log(ccdf[m])
    lx = np.log(x)
    # y = c - B log x - lam x
    X = np.stack([np.ones_like(lx), -lx, -x], axis=1)
    beta, r2 = _linfit(y, X)
    c, B, lam = float(beta[0]), float(beta[1]), float(beta[2])
    if not (np.isfinite(B) and B > 0 and np.isfinite(lam) and lam >= 0):
        return None
    return CCDTemperedFit(B=B, lam=lam, intercept=c, r2=r2)


# -----------------------------
# Figure helpers
# -----------------------------


def _save(fig: plt.Figure, path: Path, *, bundle: Optional[PdfPages]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if bundle is not None:
        bundle.savefig(fig)
    plt.close(fig)


def _fig_text_page(title: str, lines: Sequence[str]) -> plt.Figure:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(title, fontsize=14)
    txt = "\n".join(lines)
    # NOTE: avoid emoji on purpose (many PDF fonts do not contain those glyphs).
    fig.text(0.03, 0.95, txt, va="top", family="DejaVu Sans Mono")
    return fig


def _fig_checklist_table_page(
    *,
    title: str,
    meta_lines: Sequence[str],
    rows: Sequence[Dict[str, str]],
) -> plt.Figure:
    """A one-page summary with a colored checklist table.

    Each row is expected to have keys: group, series, status, B, xmin, ks, r2, note.
    """
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(title, fontsize=14)

    # Meta block
    meta_txt = "\n".join(meta_lines)
    fig.text(0.03, 0.94, meta_txt, va="top", family="DejaVu Sans Mono", fontsize=10)

    # Table
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.70])
    ax.axis("off")
    cols = ["status", "group", "series", "B", "xmin", "KS", "R²", "note"]
    cell_text = [[r.get(c, "") for c in cols] for r in rows]
    col_labels = ["OK?", "group", "series", "B", "xmin", "KS", "R²", "note"]
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.15)

    # Header style
    for (ri, ci), cell in tbl.get_celld().items():
        if ri == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")

    # Status coloring (first column)
    for i, r in enumerate(rows, start=1):
        status = (r.get("status", "") or "").strip().lower()
        cell = tbl[(i, 0)]
        if status in ("ok", "✓", "yes"):
            cell.set_facecolor("#d5f5d5")  # light green
        else:
            cell.set_facecolor("#ffe1b3")  # light orange

    return fig


def _plot_density_page(
    x: np.ndarray,
    *,
    title: str,
    bins: int,
) -> plt.Figure:
    centers, dens = log_hist_density(x, bins=bins)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if centers.size:
        ax.loglog(centers, dens, marker="o", linestyle="none", markersize=2)
    ax.set_xlabel("x (log scale)")
    ax.set_ylabel("density per bin (log scale)")
    ax.set_title(title + "\n(log–log: x in log scale AND y in log scale)")
    ax.grid(True, which="both", alpha=0.3)
    # NOTE: explicitly annotate x bins are log-spaced
    ax.text(0.02, 0.02, "x-axis: log scale\nbins: log-spaced", transform=ax.transAxes, fontsize=9)
    return fig


def _plot_ccdf_page(
    grid: np.ndarray,
    ccdf: np.ndarray,
    *,
    title: str,
    xmin: Optional[float] = None,
    pl_fit: Optional[CCDFLinearFit] = None,
    t_fit: Optional[CCDTemperedFit] = None,
) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if grid.size:
        ax.loglog(grid, ccdf, marker="o", linestyle="none", markersize=2, label="CCDF")
    if xmin is not None and np.isfinite(xmin):
        ax.axvline(float(xmin), linestyle="--", linewidth=1.0)
    # overlay fitted lines (CCDF space)
    if (pl_fit is not None) and grid.size:
        m = grid >= (xmin if xmin is not None else grid[0])
        x = grid[m]
        y = np.exp(pl_fit.intercept) * (x ** (-pl_fit.B))
        ax.loglog(x, y, linewidth=1.2, label=f"power-law fit (B≈{pl_fit.B:.2f}, R²={pl_fit.r2:.3f})")
    if (t_fit is not None) and grid.size:
        m = grid >= (xmin if xmin is not None else grid[0])
        x = grid[m]
        y = np.exp(t_fit.intercept) * (x ** (-t_fit.B)) * np.exp(-t_fit.lam * x)
        ax.loglog(x, y, linewidth=1.2, label=f"tempered fit (B≈{t_fit.B:.2f}, λ≈{t_fit.lam:.2e}, R²={t_fit.r2:.3f})")

    ax.set_xlabel("x (log scale)")
    ax.set_ylabel("P(X > x) (log scale)")
    ax.set_title(title + " — CCDF\n(log–log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.text(0.02, 0.02, "x-axis: log scale", transform=ax.transAxes, fontsize=9)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8)
    return fig


def _plot_beff_page(
    grid: np.ndarray,
    ccdf: np.ndarray,
    *,
    title: str,
    window: int,
    xmin: Optional[float] = None,
    B_ref: Optional[float] = None,
) -> plt.Figure:
    xb, beff = beff_from_ccdf_window_reg(grid, ccdf, window=window)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if xb.size:
        ax.semilogx(xb, beff, marker="o", linestyle="none", markersize=2, label=f"B_eff (window={window})")
    if xmin is not None and np.isfinite(xmin):
        ax.axvline(float(xmin), linestyle="--", linewidth=1.0)
    if B_ref is not None and np.isfinite(B_ref):
        ax.axhline(float(B_ref), linestyle="--", linewidth=1.0)
    ax.set_xlabel("x (log scale)")
    ax.set_ylabel("B_eff(x)  (local CCDF slope)")
    ax.set_title(title + " — B_eff\n(x in log scale; y in linear scale)")
    ax.grid(True, which="both", alpha=0.3)
    ax.text(0.02, 0.02, "x-axis: log scale", transform=ax.transAxes, fontsize=9)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8)
    return fig


def _plot_hill_page(k: np.ndarray, alpha: np.ndarray, *, title: str) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if k.size:
        ax.semilogx(k, alpha, marker="o", linestyle="none", markersize=2)
    ax.set_xlabel("k (log scale)  — # top order statistics")
    ax.set_ylabel("Hill estimate (CCDF exponent B)")
    ax.set_title(title + " — Hill plot (diagnostic)")
    ax.grid(True, which="both", alpha=0.3)
    return fig


# -----------------------------
# Checklist heuristics
# -----------------------------


@dataclass
class TailQuality:
    ok: bool
    reason: str


def assess_tail_quality(
    *,
    fit: Optional[ParetoFit],
    ccdf_pl: Optional[CCDFLinearFit],
    beff_x: np.ndarray,
    beff: np.ndarray,
) -> TailQuality:
    """Heuristic: mark as "good" if we see something close to a stable power-law region."""
    if fit is None or ccdf_pl is None:
        return TailQuality(ok=False, reason="fit failed")

    # Need a reasonably large tail sample
    if fit.n_tail < 2000:
        return TailQuality(ok=False, reason=f"tail too small (n_tail={fit.n_tail})")

    # CCDF should look close to linear on log–log on the fitted range
    if not (np.isfinite(ccdf_pl.r2) and ccdf_pl.r2 > 0.985):
        return TailQuality(ok=False, reason=f"CCDF not linear enough (R²={ccdf_pl.r2:.3f})")

    # B_eff should be relatively flat above xmin
    m = (beff_x >= fit.xmin)
    if np.sum(m) < 20:
        return TailQuality(ok=False, reason="not enough B_eff points above xmin")

    vals = beff[m]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    if not (np.isfinite(med) and np.isfinite(mad)):
        return TailQuality(ok=False, reason="invalid B_eff stats")
    if mad > 0.25:
        return TailQuality(ok=False, reason=f"B_eff too noisy (MAD={mad:.2f})")

    # And it should be in the ballpark of fitted B
    if abs(med - fit.B_ccdf) > 0.5:
        return TailQuality(ok=False, reason=f"B_eff median {med:.2f} far from fit B {fit.B_ccdf:.2f}")

    return TailQuality(ok=True, reason="clear-ish tail region")


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="run directory (contains tails/)")
    ap.add_argument("--out_dir", type=str, default=None, help="output directory for plots")
    ap.add_argument("--mode", type=str, default="latest", choices=["latest", "all", "iters"])
    ap.add_argument("--iters", type=str, default="", help="comma-separated iters if mode=iters")
    ap.add_argument("--groups", type=str, default="", help="comma-separated groups to plot (default: all in file)")
    ap.add_argument(
        "--series",
        type=str,
        default="grad,noise,delta",
        help="comma-separated series to plot among {grad,noise,delta,mean} (missing keys are skipped)",
    )
    ap.add_argument("--bins_density", type=int, default=80)
    ap.add_argument("--bins_ccdf", type=int, default=400)
    ap.add_argument("--beff_window", type=int, default=21, help="smoothing window (in CCDF grid points)")

    # Tail fit controls
    ap.add_argument("--no_fit", action="store_true", help="disable alpha / tail fits")
    ap.add_argument("--xmin_q_lo", type=float, default=0.90)
    ap.add_argument("--xmin_q_hi", type=float, default=0.995)
    ap.add_argument("--xmin_grid", type=int, default=50)
    ap.add_argument("--min_tail", type=int, default=1000)

    ap.add_argument("--hill", action="store_true", help="also include Hill plots")
    ap.add_argument("--hill_k_max", type=int, default=10000)

    # Bundle PDF
    ap.add_argument("--no_bundle", action="store_true", help="do not write per-iter bundled PDF")
    ap.add_argument("--over_time", action="store_true", help="if multiple iters selected, plot fitted B over time")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    tails_dir = run_dir / "tails"
    if not tails_dir.exists():
        raise SystemExit(f"tails dir not found: {tails_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "figures_tail")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _list_tail_npz_files(tails_dir)
    if not files:
        raise SystemExit(f"No tails_iter*.npz found in {tails_dir}")

    iters = []
    if args.iters.strip():
        iters = [int(x) for x in args.iters.split(",") if x.strip()]
    picked = _pick_files(files, mode=args.mode, iters=iters)
    if not picked:
        raise SystemExit("No files selected (check --mode/--iters).")

    want_groups = [g.strip() for g in args.groups.split(",") if g.strip()] if args.groups.strip() else None
    want_series = [s.strip() for s in args.series.split(",") if s.strip()]

    do_fit = not bool(args.no_fit)
    do_bundle = not bool(args.no_bundle)

    # Keep fits for optional over-time plots
    fits_over_time: Dict[Tuple[str, str], List[Tuple[int, float, float]]] = {}

    for f in picked:
        it = _extract_iter(f)
        npz = dict(np.load(f, allow_pickle=False))
        avail_groups = _available_groups(npz)
        groups = avail_groups if want_groups is None else [g for g in want_groups if g in avail_groups]

        bundle: Optional[PdfPages] = None
        bundle_path = out_dir / f"tail_tails_iter{it:07d}__ALL_PLOTS.pdf"
        if do_bundle:
            bundle = PdfPages(bundle_path)

        # Cover/checklist page
        meta_lines: List[str] = []
        meta_lines.append(f"run_dir: {run_dir}")
        meta_lines.append(f"npz: {f.relative_to(run_dir) if run_dir in f.parents else f}")
        meta_lines.append(f"iter: {it}")
        if "k_batches" in npz:
            meta_lines.append(f"k_batches: {int(npz['k_batches'][0])}")
        if "samples_per_group" in npz:
            meta_lines.append(f"samples_per_group: {int(npz['samples_per_group'][0])}")
        meta_lines.append("")
        meta_lines.append("Checklist (heuristic; see JSON next to this PDF for raw numbers).")

        # Table rows for the cover page
        table_rows: List[Dict[str, str]] = []
        checklist_json: Dict[str, Dict[str, Dict[str, object]]] = {}

        # We compute checklist after we compute fits per series.

        # First pass: compute all fits + store in memory for checklist page
        computed: Dict[Tuple[str, str], Dict[str, object]] = {}

        for g in groups:
            checklist_json[g] = {}
            series_avail = _available_series_for_group(npz, g)
            series_to_plot = [s for s in want_series if s in series_avail]

            for s in series_to_plot:
                x = npz.get(f"{s}_abs__{g}")
                if x is None:
                    continue
                x = _clean_pos(x)
                if x.size < 50:
                    continue

                grid, ccdf = ccdf_on_log_grid(x, bins=int(args.bins_ccdf))
                fit = None
                pl_fit = None
                t_fit = None
                hill_k = np.array([])
                hill_a = np.array([])

                if do_fit:
                    fit = clauset_pareto_fit(
                        x,
                        xmin_q_lo=float(args.xmin_q_lo),
                        xmin_q_hi=float(args.xmin_q_hi),
                        xmin_grid=int(args.xmin_grid),
                        min_tail=int(args.min_tail),
                    )
                    if (fit is not None) and grid.size:
                        pl_fit = fit_ccdf_powerlaw(grid, ccdf, xmin=fit.xmin)
                        t_fit = fit_ccdf_tempered(grid, ccdf, xmin=fit.xmin)
                    if args.hill:
                        hill_k, hill_a = hill_curve(x, k_min=50, k_max=int(args.hill_k_max))

                # B_eff (smoothed)
                beff_x, beff = beff_from_ccdf_window_reg(grid, ccdf, window=int(args.beff_window))

                q = assess_tail_quality(fit=fit, ccdf_pl=pl_fit, beff_x=beff_x, beff=beff)

                checklist_json[g][s] = {
                    "ok": bool(q.ok),
                    "reason": q.reason,
                    "fit": None if fit is None else {
                        "xmin": fit.xmin,
                        "a_pdf": fit.a_pdf,
                        "B_ccdf": fit.B_ccdf,
                        "ks": fit.ks,
                        "n_tail": fit.n_tail,
                        "se_B": fit.se_B,
                    },
                    "ccdf_powerlaw": None if pl_fit is None else {"B": pl_fit.B, "r2": pl_fit.r2},
                    "ccdf_tempered": None if t_fit is None else {"B": t_fit.B, "lam": t_fit.lam, "r2": t_fit.r2},
                }

                computed[(g, s)] = {
                    "x": x,
                    "grid": grid,
                    "ccdf": ccdf,
                    "fit": fit,
                    "pl_fit": pl_fit,
                    "t_fit": t_fit,
                    "beff_x": beff_x,
                    "beff": beff,
                    "hill_k": hill_k,
                    "hill_a": hill_a,
                }

                if fit is not None:
                    fits_over_time.setdefault((g, s), []).append((it, fit.B_ccdf, fit.se_B))

                # For the cover table
                row: Dict[str, str] = {
                    "status": "OK" if q.ok else "WARN",
                    "group": g,
                    "series": s,
                    "B": "" if fit is None else f"{fit.B_ccdf:.2f}",
                    "xmin": "" if fit is None else f"{fit.xmin:.3g}",
                    "KS": "" if fit is None else f"{fit.ks:.3f}",
                    "R²": "" if pl_fit is None else f"{pl_fit.r2:.3f}",
                    "note": q.reason,
                }
                table_rows.append(row)

        meta_lines.append("")
        meta_lines.append("Notes:")
        meta_lines.append("- If CCDF bends down at large x: likely truncation/tempering (clipping, finite batch, etc.).")
        meta_lines.append("- B_eff is a derivative; increase --beff_window to smooth it.")

        cover = _fig_checklist_table_page(
            title=f"Tail diagnostics @ iter={it}",
            meta_lines=meta_lines,
            rows=table_rows,
        )
        _save(cover, out_dir / f"tail_checklist__iter{it:07d}.pdf", bundle=bundle)

        # Also write machine-readable checklist JSON
        checklist_path = out_dir / f"tail_checklist__iter{it:07d}.json"
        with checklist_path.open("w", encoding="utf-8") as fp:
            json.dump(checklist_json, fp, indent=2, ensure_ascii=False)

        # Second pass: actual plots
        for g in groups:
            series_avail = _available_series_for_group(npz, g)
            series_to_plot = [s for s in want_series if s in series_avail]
            for s in series_to_plot:
                rec = computed.get((g, s))
                if rec is None:
                    continue
                x = rec["x"]
                grid = rec["grid"]
                ccdf = rec["ccdf"]
                fit = rec["fit"]
                pl_fit = rec["pl_fit"]
                t_fit = rec["t_fit"]
                hill_k = rec["hill_k"]
                hill_a = rec["hill_a"]

                # density
                fig = _plot_density_page(x, title=f"|{s}| density (group={g}, iter={it})", bins=int(args.bins_density))
                _save(fig, out_dir / f"density_{s}__{g}__iter{it:07d}.pdf", bundle=bundle)

                # CCDF
                fig = _plot_ccdf_page(
                    grid,
                    ccdf,
                    title=f"|{s}| (group={g}, iter={it})",
                    xmin=None if fit is None else fit.xmin,
                    pl_fit=pl_fit,
                    t_fit=t_fit,
                )
                _save(fig, out_dir / f"ccdf_{s}__{g}__iter{it:07d}.pdf", bundle=bundle)

                # B_eff
                fig = _plot_beff_page(
                    grid,
                    ccdf,
                    title=f"|{s}| (group={g}, iter={it})",
                    window=int(args.beff_window),
                    xmin=None if fit is None else fit.xmin,
                    B_ref=None if fit is None else fit.B_ccdf,
                )
                _save(fig, out_dir / f"beff_{s}__{g}__iter{it:07d}.pdf", bundle=bundle)

                # Hill plot (optional)
                if args.hill:
                    fig = _plot_hill_page(hill_k, hill_a, title=f"|{s}| (group={g}, iter={it})")
                    _save(fig, out_dir / f"hill_{s}__{g}__iter{it:07d}.pdf", bundle=bundle)

        if bundle is not None:
            bundle.close()
            print(f"[OK] Bundled PDF: {bundle_path}")

        print(f"[OK] Plotted iter={it} from {f.name} -> {out_dir}")

    # Optional: B over time
    if args.over_time and len(picked) >= 2:
        for (g, s), rows in fits_over_time.items():
            rows = sorted(rows, key=lambda t: t[0])
            its = np.array([r[0] for r in rows], dtype=np.int64)
            Bs = np.array([r[1] for r in rows], dtype=np.float64)
            se = np.array([r[2] for r in rows], dtype=np.float64)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.errorbar(its, Bs, yerr=se, fmt="o", markersize=3, capsize=2)
            ax.set_xlabel("iteration")
            ax.set_ylabel("B (CCDF exponent)  — Clauset fit")
            ax.set_title(f"Tail exponent over time: series={s}, group={g}")
            ax.grid(True, which="both", alpha=0.3)
            _save(fig, out_dir / f"B_over_time__{s}__{g}.pdf", bundle=None)

        print(f"[OK] Wrote over-time plots -> {out_dir}")

    print("[DONE]")


if __name__ == "__main__":
    main()
