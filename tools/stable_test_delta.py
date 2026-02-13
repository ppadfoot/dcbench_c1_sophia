#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Alpha-stable / stability-under-sums test for delta samples.

Background
----------
If noise is symmetric alpha-stable (SαS), then for independent samples X1, X2:
    X1 + X2  ≍  2^(1/α) X
where ≍ denotes equality in distribution.

This tool uses that idea on stored delta samples.

Inputs
------
We expect tails npz produced by c1bench/tail_diag.py:
    delta_signed__<group>  (preferred)
    delta_abs__<group>     (fallback)

Output
------
Writes a PDF with:
  - CCDF(|delta|) vs Pareto fit
  - Stability check: CCDF(|X|) vs CCDF(|(X1+X2)/2^(1/α)|)
  - Quantile ratio diagnostics

Note
----
This is a diagnostic, not a proof. It is most meaningful if:
  - delta_signed exists (signed values)
  - sample size is large (>= 50k)
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_npz(run_dir: str, mode: str, it: int | None) -> str:
    if mode == "latest":
        files = sorted(glob.glob(os.path.join(run_dir, "tails", "tails_iter*.npz")), key=natural_key)
        if not files:
            raise FileNotFoundError(f"No npz files under {run_dir}/tails")
        return files[-1]
    if mode == "iter":
        assert it is not None
        cand = os.path.join(run_dir, "tails", f"tails_iter{it:07d}.npz")
        if not os.path.exists(cand):
            raise FileNotFoundError(f"Not found: {cand}")
        return cand
    raise ValueError("mode must be latest|iter")


def ccdf(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    xs = np.sort(x)
    n = xs.size
    idx = np.searchsorted(xs, grid, side="right")
    y = (n - idx) / n
    return np.clip(y, 1.0 / (n + 1.0), 1.0)


def pareto_mle_alpha(x_tail: np.ndarray, xmin: float) -> float:
    # CCDF exponent alpha for Pareto tail
    # alpha_hat = n / sum(log(x/xmin))
    return float(x_tail.size / np.sum(np.log(x_tail / xmin)))


def pareto_fit_mle_ks(x_abs: np.ndarray, q_candidates: np.ndarray, min_tail: int) -> dict | None:
    x = np.asarray(x_abs, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size < min_tail:
        return None
    x.sort()

    best = None
    for q in q_candidates:
        xmin = float(np.quantile(x, q))
        tail = x[x >= xmin]
        if tail.size < min_tail:
            continue
        alpha = pareto_mle_alpha(tail, xmin)
        # KS on tail using empirical CDF and model CDF
        t = tail
        F_emp = np.arange(1, t.size + 1) / t.size
        F_model = 1.0 - (t / xmin) ** (-alpha)
        ks = float(np.max(np.abs(F_emp - F_model)))
        if best is None or ks < best["ks"]:
            best = {"alpha": float(alpha), "xmin": xmin, "n_tail": int(t.size), "ks": ks, "q": float(q)}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--mode", choices=["latest", "iter"], default="latest")
    ap.add_argument("--iter", type=int, default=None)
    ap.add_argument("--group", type=str, default="all")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--min_tail", type=int, default=2000)
    ap.add_argument("--q_low", type=float, default=0.90)
    ap.add_argument("--q_high", type=float, default=0.995)
    ap.add_argument("--q_steps", type=int, default=25)
    ap.add_argument("--grid_points", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    npz_path = find_npz(args.run_dir, args.mode, args.iter)
    tag = os.path.splitext(os.path.basename(npz_path))[0]

    with np.load(npz_path) as z:
        key_signed = f"delta_signed__{args.group}"
        key_abs = f"delta_abs__{args.group}"
        if key_signed in z:
            delta = z[key_signed].astype(np.float64)
            signed = True
        elif key_abs in z:
            delta = z[key_abs].astype(np.float64)
            signed = False
        else:
            raise KeyError(f"No {key_signed} or {key_abs} in {npz_path}")

    delta = delta[np.isfinite(delta)]
    if signed:
        delta = delta[delta != 0]
        x_abs = np.abs(delta)
    else:
        delta = delta[delta > 0]
        x_abs = delta

    rng = np.random.default_rng(args.seed)

    fit = pareto_fit_mle_ks(
        x_abs,
        q_candidates=np.linspace(args.q_low, args.q_high, args.q_steps),
        min_tail=args.min_tail,
    )
    if fit is None:
        raise RuntimeError(f"Not enough samples for tail fit (need >= {args.min_tail})")

    alpha = fit["alpha"]
    xmin = fit["xmin"]

    # Build CCDF grid on [xmin, high quantile]
    x_max = float(np.quantile(x_abs, 0.9999))
    grid = np.logspace(np.log10(max(xmin, 1e-12)), np.log10(max(x_max, xmin * 1.01)), args.grid_points)
    ccdf_emp = ccdf(x_abs, grid)

    # Pareto model CCDF: (x/xmin)^(-alpha)
    ccdf_pareto = np.clip((grid / xmin) ** (-alpha), 1e-300, 1.0)

    # Stability test
    # Use two independent halves (or bootstrap) to form X1 and X2
    n = delta.size
    if n < 2000:
        raise RuntimeError("Not enough delta samples for stability test")

    # sample without replacement for independence-ish
    idx = rng.permutation(n)
    h = n // 2
    X1 = delta[idx[:h]]
    X2 = delta[idx[h:2*h]]

    if signed:
        S = (X1 + X2) / (2.0 ** (1.0 / alpha))
        A1 = np.abs(X1)
        A2 = np.abs(S)
    else:
        # fallback: use abs-only (we can only do a rough check)
        A1 = np.abs(X1)
        A2 = np.abs(X2)  # can't form sums; just compare two halves

    # CCDF comparison on same grid (restricted)
    ccdf_1 = ccdf(A1[A1 > 0], grid)
    ccdf_2 = ccdf(A2[A2 > 0], grid)

    # Quantile ratio diagnostics
    qs = np.array([0.9, 0.95, 0.99, 0.995])
    q1 = np.quantile(A1, qs)
    q2 = np.quantile(A2, qs)
    qratio = q2 / (q1 + 1e-12)

    # Plot
    out = args.out or os.path.join(args.run_dir, "figures_tail", f"stable_test__{tag}__{args.group}.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    fig = plt.figure(figsize=(8, 9))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(grid, ccdf_emp, label="empirical CCDF(|delta|)")
    ax1.plot(grid, ccdf_pareto, linestyle="--", label=f"Pareto fit: alpha={alpha:.3g}, xmin={xmin:.3g}, KS={fit['ks']:.3g}")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("x")
    ax1.set_ylabel("CCDF")
    ax1.set_title(f"{tag}  group={args.group}  signed={signed}")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend()

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(grid, ccdf_1, label="CCDF(|X|)")
    if signed:
        ax2.plot(grid, ccdf_2, label=r"CCDF(|(X1+X2)/2^(1/alpha)|)")
        ax2.set_title("Stability-under-sums check (should overlap if SαS)")
    else:
        ax2.plot(grid, ccdf_2, label="CCDF(|X|) on second half")
        ax2.set_title("Fallback check (abs-only): two halves should be similar")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("x")
    ax2.set_ylabel("CCDF")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend()

    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(qs, qratio, marker="o")
    ax3.axhline(1.0, linestyle="--")
    ax3.set_xlabel("quantile")
    ax3.set_ylabel("ratio")
    ax3.set_title("Quantile ratios (target ~1 if stable)")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.text(0.02, 0.95, f"ratios: {qratio}", transform=ax3.transAxes, va="top")

    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)

    print(f"[ok] wrote {out}")
    print(f"fit: {fit}")
    print(f"quantile ratios (q2/q1): {list(zip(qs.tolist(), qratio.tolist()))}")


if __name__ == "__main__":
    main()
