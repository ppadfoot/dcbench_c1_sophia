#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Scan tails_iter*.npz over time and summarize tail exponents.

This tool processes stored arrays (no retraining) and produces:
  - tail_over_time.csv
  - alpha_over_time__<kind>.pdf
  - model_compare_over_time__<kind>.pdf

It estimates:
  - alpha_med = median(B_eff) on a tail mask, where B_eff is a robust local slope of log(CCDF) vs log(x)
  - R^2 of pure powerlaw fit: log(CCDF) ≈ c - alpha log x
  - R^2 of tempered fit: log(CCDF) ≈ c - alpha log x - x/lambda

Kinds:
  grad, noise, delta, mean

Run:
  python tools/scan_tail_over_time.py --run_dir out/<run_name>
"""

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_npz(run_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(run_dir, "tails", "tails_iter*.npz")), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No npz found under {run_dir}/tails/")
    return files


def parse_iter(npz_path: str) -> int:
    base = os.path.basename(npz_path)
    m = re.search(r"tails_iter(\d+)\.npz", base)
    if not m:
        return -1
    return int(m.group(1))


def load_groups(npz_path: str) -> List[str]:
    with np.load(npz_path) as z:
        keys = list(z.keys())
    groups = set()
    for k in keys:
        m = re.match(r"^(grad_abs|noise_abs|mean_abs|delta_abs)__([A-Za-z0-9_]+)$", k)
        if m:
            groups.add(m.group(2))
    return sorted(groups)


def get_arr(z, prefix: str, group: str) -> Optional[np.ndarray]:
    key = f"{prefix}__{group}"
    if key not in z:
        return None
    x = z[key].astype(np.float64, copy=False)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    return x if x.size > 0 else None


@dataclass
class Curves:
    x: np.ndarray
    ccdf: np.ndarray
    beff: np.ndarray
    tail_count: np.ndarray
    n: int


def make_x_grid_quantile(x_sorted: np.ndarray, grid_points=250, q_low=0.001, q_high=0.99999) -> np.ndarray:
    ps = np.linspace(q_low, q_high, grid_points)
    xg = np.quantile(x_sorted, ps)
    xg = np.unique(xg)
    if xg.size < 2:
        xmin = max(np.quantile(x_sorted, q_low), x_sorted.min())
        xmax = max(np.quantile(x_sorted, q_high), xmin * 1.001)
        xg = np.logspace(np.log10(xmin), np.log10(xmax), grid_points)
    return xg


def theil_sen_local_slope(logx: np.ndarray, logy: np.ndarray, window: int) -> np.ndarray:
    n = logx.size
    w = max(1, window // 2)
    slopes = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        j0 = max(0, i - w)
        j1 = min(n, i + w + 1)
        X = logx[j0:j1]
        Y = logy[j0:j1]
        m = X.size
        if m < 2:
            continue
        s_list = []
        for a in range(m):
            for b in range(a + 1, m):
                dx = X[b] - X[a]
                if dx == 0:
                    continue
                s_list.append((Y[b] - Y[a]) / dx)
        if s_list:
            slopes[i] = float(np.median(np.array(s_list, dtype=np.float64)))
    return slopes


def ccdf_beff(x: np.ndarray, grid_points=250, beff_window=21) -> Curves:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    x.sort()
    n = x.size
    xg = make_x_grid_quantile(x, grid_points=grid_points)
    idx = np.searchsorted(x, xg, side="right")
    ccdf = (n - idx) / n
    ccdf = np.clip(ccdf, 1.0 / (n + 1.0), 1.0)
    tail_count = n * ccdf

    logx = np.log(xg)
    logy = np.log(ccdf)
    slope = theil_sen_local_slope(logx, logy, window=beff_window)
    beff = -slope
    return Curves(x=xg, ccdf=ccdf, beff=beff, tail_count=tail_count, n=n)


def tail_mask(cur: Curves, min_tail_count=200, ccdf_max=0.2):
    return (
        np.isfinite(cur.ccdf)
        & np.isfinite(cur.beff)
        & (cur.tail_count >= min_tail_count)
        & (cur.ccdf <= ccdf_max)
    )


def fit_powerlaw(cur: Curves, mask: np.ndarray) -> Optional[Tuple[float, float]]:
    x = cur.x[mask]
    y = cur.ccdf[mask]
    if x.size < 10:
        return None
    X = np.log(x)
    Y = np.log(y)
    b, a = np.polyfit(X, Y, 1)
    Yhat = a + b * X
    ss_res = float(np.sum((Y - Yhat) ** 2))
    ss_tot = float(np.sum((Y - float(np.mean(Y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    alpha = -b
    return float(alpha), float(r2)


def fit_tempered(cur: Curves, mask: np.ndarray) -> Optional[Tuple[float, float, float]]:
    x = cur.x[mask]
    y = cur.ccdf[mask]
    if x.size < 10:
        return None
    X1 = np.log(x)
    X2 = x
    Y = np.log(y)
    A = np.stack([np.ones_like(X1), X1, X2], axis=1)
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    c0, b1, b2 = coef
    Yhat = A @ coef
    ss_res = float(np.sum((Y - Yhat) ** 2))
    ss_tot = float(np.sum((Y - float(np.mean(Y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    alpha = -b1
    lam = (-1.0 / b2) if b2 < 0 else float("inf")
    return float(alpha), float(lam), float(r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--groups", default="", help="comma-separated subset (default: all)")
    ap.add_argument("--kinds", default="grad,noise,delta,mean", help="comma-separated: grad,noise,delta,mean")
    ap.add_argument("--min_tail_count", type=int, default=200)
    ap.add_argument("--ccdf_max", type=float, default=0.2)
    ap.add_argument("--grid_points", type=int, default=250)
    ap.add_argument("--beff_window", type=int, default=21)
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or os.path.join(run_dir, "figures_tail")
    os.makedirs(out_dir, exist_ok=True)

    npz_files = list_npz(run_dir)
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    kind2prefix = {"grad": "grad_abs", "noise": "noise_abs", "mean": "mean_abs", "delta": "delta_abs"}
    kinds = [k for k in kinds if k in kind2prefix]

    all_groups = load_groups(npz_files[-1])
    if args.groups.strip():
        groups = [g.strip() for g in args.groups.split(",") if g.strip()]
        groups = [g for g in groups if g in all_groups]
    else:
        groups = all_groups

    rows = []
    for f in npz_files:
        it = parse_iter(f)
        with np.load(f) as z:
            for grp in groups:
                for kind in kinds:
                    x = get_arr(z, kind2prefix[kind], grp)
                    if x is None or x.size < 100:
                        continue
                    cur = ccdf_beff(x, grid_points=args.grid_points, beff_window=args.beff_window)
                    m = tail_mask(cur, min_tail_count=args.min_tail_count, ccdf_max=args.ccdf_max)
                    if int(np.sum(m)) < 10:
                        continue
                    alpha_med = float(np.median(cur.beff[m]))
                    alpha_q25 = float(np.quantile(cur.beff[m], 0.25))
                    alpha_q75 = float(np.quantile(cur.beff[m], 0.75))
                    pw = fit_powerlaw(cur, m)
                    tp = fit_tempered(cur, m)
                    alpha_pw, r2_pw = (pw if pw else (np.nan, np.nan))
                    alpha_tp, lam_tp, r2_tp = (tp if tp else (np.nan, np.nan, np.nan))
                    dr2 = (r2_tp - r2_pw) if (np.isfinite(r2_tp) and np.isfinite(r2_pw)) else np.nan
                    rows.append([
                        it, grp, kind,
                        cur.n, int(np.sum(m)),
                        alpha_med, alpha_q25, alpha_q75,
                        alpha_pw, r2_pw,
                        alpha_tp, lam_tp, r2_tp,
                        dr2,
                    ])

    if not rows:
        print("[warn] no rows produced")
        return

    header = [
        "iter","group","kind",
        "n_total","n_mask",
        "alpha_med","alpha_q25","alpha_q75",
        "alpha_powerlaw","r2_powerlaw",
        "alpha_tempered","lambda_tempered","r2_tempered",
        "dr2_tempered_minus_powerlaw",
    ]
    csv_path = os.path.join(out_dir, "tail_over_time.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    print(f"[ok] wrote {csv_path}")

    rows_np = np.array(rows, dtype=object)
    uniq_kinds = sorted(set(rows_np[:,2].tolist()))
    uniq_groups = sorted(set(rows_np[:,1].tolist()), key=natural_key)

    # alpha plots
    for kind in uniq_kinds:
        plt.figure(figsize=(8,4.8))
        for grp in uniq_groups:
            mask = (rows_np[:,2] == kind) & (rows_np[:,1] == grp)
            if int(np.sum(mask)) == 0:
                continue
            xs = rows_np[mask][:,0].astype(int)
            ys = rows_np[mask][:,5].astype(float)
            order = np.argsort(xs)
            plt.plot(xs[order], ys[order], linewidth=2, label=grp)
        plt.xlabel("iter")
        plt.ylabel("alpha_hat (median B_eff on tail mask)")
        plt.title(f"alpha over time (kind={kind})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(ncol=2, fontsize=9)
        outp = os.path.join(out_dir, f"alpha_over_time__{kind}.pdf")
        plt.savefig(outp, bbox_inches="tight")
        plt.close()
        print(f"[ok] wrote {outp}")

    # model compare plots
    for kind in uniq_kinds:
        plt.figure(figsize=(8,4.8))
        for grp in uniq_groups:
            mask = (rows_np[:,2] == kind) & (rows_np[:,1] == grp)
            if int(np.sum(mask)) == 0:
                continue
            xs = rows_np[mask][:,0].astype(int)
            ys = rows_np[mask][:,13].astype(float)
            order = np.argsort(xs)
            plt.plot(xs[order], ys[order], linewidth=2, label=grp)
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("iter")
        plt.ylabel("ΔR² = R²_tempered - R²_powerlaw (tail)")
        plt.title(f"tempered vs powerlaw (kind={kind})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(ncol=2, fontsize=9)
        outp = os.path.join(out_dir, f"model_compare_over_time__{kind}.pdf")
        plt.savefig(outp, bbox_inches="tight")
        plt.close()
        print(f"[ok] wrote {outp}")


if __name__ == "__main__":
    main()
