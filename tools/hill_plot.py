#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _list_npz(run_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(run_dir, "tails", "tails_iter*.npz")), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No npz under {run_dir}/tails/")
    return files


def _parse_iter(path: str) -> int:
    m = re.search(r"tails_iter(\d+)\.npz", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _load_groups(npz_path: str, prefix: str) -> List[str]:
    with np.load(npz_path) as z:
        keys = list(z.keys())
    groups = set()
    for k in keys:
        if k.startswith(prefix + "__"):
            groups.add(k.split("__", 1)[1])
    return sorted(groups)


def _get_samples(z, prefix: str, group: str) -> Optional[np.ndarray]:
    key = f"{prefix}__{group}"
    if key not in z:
        return None
    x = z[key].astype(np.float64, copy=False)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    return x if x.size > 10 else None


def hill_alpha(sorted_desc: np.ndarray, k_values: np.ndarray) -> np.ndarray:
    """
    Hill estimator for alpha (CCDF exponent) given x sorted in descending order.

    Using 0-indexed array x[0] >= ... >= x[n-1].
    For each k: threshold = x[k] (i.e., x_{(k+1)} in 1-indexed)
      alpha_hat(k) = 1 / ( (1/k) * sum_{i=0..k-1} log(x[i]/x[k]) )
                  = 1 / ( (mean(log x[0:k]) - log x[k]) )
    """
    x = sorted_desc
    n = x.size
    logx = np.log(x)
    ps = np.cumsum(logx)  # prefix sum

    out = np.full(k_values.shape, np.nan, dtype=np.float64)
    for idx, k in enumerate(k_values):
        k = int(k)
        if k < 2 or k >= n:
            continue
        thr = logx[k]
        mean_top = ps[k - 1] / k
        denom = mean_top - thr
        if denom <= 0 or not np.isfinite(denom):
            continue
        out[idx] = 1.0 / denom
    return out


def find_plateau(k: np.ndarray, alpha: np.ndarray, slope_tol: float = 0.08, min_points: int = 12):
    """
    Very lightweight plateau heuristic:
    - compute slope of alpha vs log(k)
    - keep region where |slope| < slope_tol
    - return the longest contiguous segment
    """
    m = np.isfinite(alpha) & (alpha > 0)
    if m.sum() < min_points:
        return None

    kk = k[m]
    aa = alpha[m]
    logk = np.log(kk)

    # slope via finite diff
    d = np.gradient(aa, logk)
    good = np.abs(d) < slope_tol

    # longest contiguous segment in good
    best = None
    start = None
    for i, g in enumerate(good):
        if g and start is None:
            start = i
        if (not g or i == len(good) - 1) and start is not None:
            end = i if not g else i + 1
            length = end - start
            if length >= min_points:
                seg = (start, end)
                if best is None or (seg[1] - seg[0]) > (best[1] - best[0]):
                    best = seg
            start = None

    if best is None:
        return None

    a0, a1 = best
    return {
        "k_min": int(kk[a0]),
        "k_max": int(kk[a1 - 1]),
        "alpha_med": float(np.median(aa[a0:a1])),
        "alpha_q25": float(np.quantile(aa[a0:a1], 0.25)),
        "alpha_q75": float(np.quantile(aa[a0:a1], 0.75)),
        "idx0": int(a0),
        "idx1": int(a1),
    }


def plot_hill(
    out_path: str,
    title: str,
    k: np.ndarray,
    alpha: np.ndarray,
    plateau: Optional[dict],
):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(k, alpha, linewidth=2)

    ax.set_xscale("log")
    ax.set_xlabel("k (number of top order stats, log scale)")
    ax.set_ylabel(r"Hill estimate $\hat{\alpha}(k)$  (CCDF exponent)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)

    if plateau is not None:
        ax.axhline(plateau["alpha_med"], linestyle="--", linewidth=1.2)
        ax.axvline(plateau["k_min"], linestyle="--", linewidth=1.0)
        ax.axvline(plateau["k_max"], linestyle="--", linewidth=1.0)
        ax.text(
            0.02, 0.95,
            f"plateau: k∈[{plateau['k_min']},{plateau['k_max']}]\n"
            f"alpha≈{plateau['alpha_med']:.3g} (IQR {plateau['alpha_q25']:.3g}–{plateau['alpha_q75']:.3g})",
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
            fontsize=10,
        )
    else:
        ax.text(
            0.02, 0.95,
            "no clear plateau (likely tempered / mixture / too little tail)",
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
            fontsize=10,
        )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--mode", choices=["latest", "iters", "all"], default="latest")
    ap.add_argument("--iters", default="", help="comma-separated iters if mode=iters")
    ap.add_argument("--prefix", default="delta_abs", help="delta_abs|noise_abs|grad_abs|mean_abs")
    ap.add_argument("--groups", default="", help="comma-separated groups; default=all found")
    ap.add_argument("--k_min", type=int, default=50)
    ap.add_argument("--k_max_frac", type=float, default=0.05, help="max k as fraction of n (default 5%)")
    ap.add_argument("--k_points", type=int, default=120, help="how many k values to evaluate (log-spaced)")
    ap.add_argument("--plateau_slope_tol", type=float, default=0.08)
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or os.path.join(run_dir, "figures_tail")
    os.makedirs(out_dir, exist_ok=True)

    npz_files = _list_npz(run_dir)
    if args.mode == "latest":
        chosen = [npz_files[-1]]
    elif args.mode == "all":
        chosen = npz_files
    else:
        if not args.iters.strip():
            raise ValueError("--mode iters requires --iters")
        iters = [int(x.strip()) for x in args.iters.split(",") if x.strip()]
        chosen = []
        have = { _parse_iter(p): p for p in npz_files }
        for it in iters:
            if it in have:
                chosen.append(have[it])
            else:
                raise FileNotFoundError(f"tails_iter{it:07d}.npz not found in {run_dir}/tails")

    for npz_path in chosen:
        it = _parse_iter(npz_path)
        groups_all = _load_groups(npz_path, args.prefix)

        if args.groups.strip():
            groups = [g.strip() for g in args.groups.split(",") if g.strip()]
        else:
            groups = groups_all

        with np.load(npz_path) as z:
            for grp in groups:
                x = _get_samples(z, args.prefix, grp)
                if x is None:
                    print(f"[skip] iter={it} group={grp} missing {args.prefix}")
                    continue

                x = np.sort(x)[::-1]  # descending
                n = x.size
                kmax = int(max(args.k_min + 2, min(n - 2, int(n * args.k_max_frac))))
                if kmax <= args.k_min + 2:
                    print(f"[skip] iter={it} group={grp} too small n={n}")
                    continue

                # log-spaced k values
                ks = np.unique(np.round(np.logspace(np.log10(args.k_min), np.log10(kmax), args.k_points)).astype(int))
                ks = ks[(ks >= 2) & (ks < n - 1)]

                alpha = hill_alpha(x, ks)
                plateau = find_plateau(ks, alpha, slope_tol=args.plateau_slope_tol)

                title = f"Hill plot: iter={it}  group={grp}  prefix={args.prefix}  (n={n})"
                out_path = os.path.join(out_dir, f"hill__{args.prefix}__{grp}__iter{it:07d}.pdf")
                plot_hill(out_path, title, ks, alpha, plateau)
                print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()

