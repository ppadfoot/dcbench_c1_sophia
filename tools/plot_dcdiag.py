#!/usr/bin/env python3
# tools/plot_dcdiag.py
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def read_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def series(rows, key):
    it, y = [], []
    for r in rows:
        if "it" not in r:
            continue
        it.append(int(r["it"]))
        v = r.get(key, None)
        y.append(np.nan if v is None else float(v))
    return np.asarray(it, dtype=np.int64), np.asarray(y, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--out_pdf", type=str, default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    diag_path = run_dir / "diag.jsonl"
    if not diag_path.exists():
        raise FileNotFoundError(f"missing {diag_path}")

    rows = read_jsonl(diag_path)

    out_pdf = Path(args.out_pdf) if args.out_pdf else (run_dir / "figures_dcdiag" / "dcdiag_traces.pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    keys = [
        "dc_hat", "cos_hat", "score_hat",
        "P_hat", "G_hat", "G_used", "E_hat", "E_cap",
        "c_aln_hat", "c_aln_pos",
        "frac_G_nonpos", "G_used_is_floor",
    ]
    S = {k: series(rows, k) for k in keys}

    with PdfPages(str(out_pdf)) as pdf:
        # 1) DC/cos/score
        fig = plt.figure(figsize=(10, 5))
        for k in ["dc_hat", "cos_hat", "score_hat"]:
            it, y = S[k]
            plt.plot(it, y, label=k)
        plt.grid(True, alpha=0.3)
        plt.xlabel("iter")
        plt.title("DC_hat / cos_hat / score_hat")
        plt.legend()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # 2) P/G/E (symlog)
        fig = plt.figure(figsize=(10, 5))
        for k in ["P_hat", "G_hat", "E_hat"]:
            it, y = S[k]
            plt.plot(it, y, label=k)
        plt.yscale("symlog", linthresh=1e-6)
        plt.grid(True, alpha=0.3)
        plt.xlabel("iter")
        plt.title("P_hat / G_hat / E_hat (symlog)")
        plt.legend()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # 3) ALN proxies
        fig = plt.figure(figsize=(10, 5))
        for k in ["c_aln_hat", "c_aln_pos"]:
            it, y = S[k]
            plt.plot(it, y, label=k)
        plt.grid(True, alpha=0.3)
        plt.xlabel("iter")
        plt.title("ALN proxies (raw vs stable)")
        plt.legend()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # 4) reliability
        fig = plt.figure(figsize=(10, 5))
        it, y = S["frac_G_nonpos"]
        plt.plot(it, y, label="frac_G_nonpos")
        it2, y2 = S["G_used_is_floor"]
        plt.plot(it2, y2, label="G_used_is_floor")
        plt.grid(True, alpha=0.3)
        plt.xlabel("iter")
        plt.title("Reliability indicators")
        plt.legend()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print("[saved]", out_pdf)


if __name__ == "__main__":
    main()

