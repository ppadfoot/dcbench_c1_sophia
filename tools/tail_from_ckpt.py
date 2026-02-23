#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/tail_from_ckpt.py

Post-hoc "tail diagnostics" for grad/noise/delta at a fixed checkpoint (NO retraining):
- loads model at iter
- runs K independent minibatches (forward/backward)
- collects many coordinate samples per parameter group (layer_bucket supported)
- computes:
    grad_abs__group  : |g|
    mean_abs__group  : |mean_k g^{(k)}|
    noise_abs__group : |g^{(k)} - mean|
    delta_abs__group : |g^{(a)} - g^{(b)}| from adjacent pairs of batches
- writes npz into <run_dir>/tails/tails_iterXXXXXXX.npz (by default overwrites existing)
  optionally backup old file.

This increases tail sample counts WITHOUT training, ideal for making CCDF/B_eff/Hill smoother.

Example:
  python tools/tail_from_ckpt.py --run_dir out/E4_layer_bucket_v2 --iter 1800 \
    --ckpt_policy floor --k_batches 128 --samples_per_group 2000000 --coord_budget_per_group 400000 \
    --pairs adjacent --backup
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch

# -----------------------------
# Fix: ensure repo root importable
# -----------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model import GPT, GPTConfig  # noqa: E402
from c1bench.utils import set_seed, mkdir_p  # noqa: E402


# -----------------------------
# Data loader (same as train.py)
# -----------------------------
def openwebtext_get_batch(data_dir: Path, split: str, batch_size: int, block_size: int, device: str):
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


# -----------------------------
# Helpers: config, checkpoints
# -----------------------------
def load_config_resolved(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "config_resolved.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    p2 = run_dir / "meta.json"
    if p2.exists():
        meta = json.loads(p2.read_text(encoding="utf-8"))
        if "config" in meta and isinstance(meta["config"], dict):
            return meta["config"]
    raise FileNotFoundError("Could not find config_resolved.json or meta.json in run_dir")


def dtype_from_str(s: str) -> torch.dtype:
    s = str(s).lower()
    if s == "float32":
        return torch.float32
    if s == "bfloat16":
        return torch.bfloat16
    return torch.float16


def list_ckpts(ckpt_dir: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for p in sorted(ckpt_dir.glob("ckpt_iter*.pt")):
        m = re.search(r"ckpt_iter(\d+)\.pt", p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def pick_ckpt(ckpts: Dict[int, Path], requested: int, policy: str) -> Tuple[int, Path]:
    if not ckpts:
        raise FileNotFoundError("No ckpt_iter*.pt found.")
    if policy == "exact":
        if requested not in ckpts:
            raise FileNotFoundError(f"Checkpoint not found for iter={requested:07d}")
        return requested, ckpts[requested]
    iters = sorted(ckpts.keys())
    if policy == "floor":
        le = [i for i in iters if i <= requested]
        if not le:
            return iters[0], ckpts[iters[0]]
        chosen = le[-1]
        return chosen, ckpts[chosen]
    if policy == "nearest":
        chosen = min(iters, key=lambda i: abs(i - requested))
        return chosen, ckpts[chosen]
    raise ValueError(f"Unknown ckpt_policy={policy}")


# -----------------------------
# Grouping (layer_bucket)
# -----------------------------
def infer_group(name: str, grouping_mode: str, n_layer: int) -> str:
    nm = name.lower()

    # embeddings / head
    if any(k in nm for k in ["wte", "wpe", "lm_head"]):
        return "embed"
    # norms
    if "ln_" in nm or "ln_f" in nm or "lnf" in nm or ".ln" in nm:
        return "norm"

    if grouping_mode != "layer_bucket":
        return "all"

    m = re.search(r"\.h\.(\d+)\.", nm)
    if m is None:
        return "all"
    li = int(m.group(1))

    b0 = n_layer // 3
    b1 = 2 * n_layer // 3
    if li < b0:
        stage = "early"
    elif li < b1:
        stage = "mid"
    else:
        stage = "late"

    if ".attn." in nm:
        return f"attn_{stage}"
    if ".mlp." in nm:
        return f"mlp_{stage}"
    return "all"


# -----------------------------
# Coordinate sampling
# -----------------------------
class Sampler:
    """Fixed indices into a flattened tensor."""
    def __init__(self, numel: int, m: int, seed: int):
        self.numel = int(numel)
        self.m = int(min(m, numel))
        rng = np.random.default_rng(seed)
        self.idx = rng.choice(self.numel, size=self.m, replace=False) if self.m > 0 else np.zeros((0,), dtype=np.int64)

    def take(self, flat: torch.Tensor) -> torch.Tensor:
        if self.m == 0:
            return flat.new_empty((0,))
        return flat[self.idx]


def build_group_samplers(
    model: torch.nn.Module,
    grouping_mode: str,
    n_layer: int,
    keep_groups: List[str],
    coord_budget_per_group: int,
    seed: int,
) -> Dict[str, List[Tuple[str, Sampler]]]:
    """
    For each group, create a list of (param_name, sampler) such that total coords per group ~ coord_budget_per_group.
    """
    rng = np.random.default_rng(seed)
    # gather params per group
    group_params: Dict[str, List[Tuple[str, torch.Tensor]]] = {g: [] for g in keep_groups}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = infer_group(name, grouping_mode, n_layer)
        if g in keep_groups:
            group_params[g].append((name, p))

    out: Dict[str, List[Tuple[str, Sampler]]] = {g: [] for g in keep_groups}
    for g in keep_groups:
        remaining = int(coord_budget_per_group)
        # deterministic order for reproducibility
        for name, p in group_params[g]:
            if remaining <= 0:
                break
            numel = int(p.numel())
            m = min(remaining, numel)
            # seed per tensor so indices fixed across batches (important for mean/noise and delta)
            s = Sampler(numel=numel, m=m, seed=int(seed + (hash(name) % 100000)))
            out[g].append((name, s))
            remaining -= m

        if remaining > 0 and len(group_params[g]) > 0:
            # if we still didn't fill budget, sample additional tensors again (rare)
            # (keeps it robust if group has few tensors)
            choices = group_params[g]
            while remaining > 0:
                name, p = choices[int(rng.integers(0, len(choices)))]
                numel = int(p.numel())
                m = min(remaining, numel)
                s = Sampler(numel=numel, m=m, seed=int(seed + 777 + (hash(name) % 100000)))
                out[g].append((name, s))
                remaining -= m

    return out


def append_with_cap(rng: np.random.Generator, dst: List[np.ndarray], arr: np.ndarray, cap: int) -> int:
    """
    Append up to `cap` total elements across dst. Return how many elements appended now.
    """
    have = sum(a.size for a in dst)
    rem = cap - have
    if rem <= 0:
        return 0
    if arr.size <= rem:
        dst.append(arr)
        return int(arr.size)
    # sample rem elements without replacement for stability
    idx = rng.choice(arr.size, size=rem, replace=False)
    dst.append(arr[idx])
    return int(rem)


# -----------------------------
# Main logic
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--iter", required=True, type=int)
    ap.add_argument("--ckpt_policy", default="floor", choices=["exact", "floor", "nearest"])
    ap.add_argument("--k_batches", type=int, default=128, help="number of independent minibatches to sample")
    ap.add_argument("--pairs", default="adjacent", choices=["adjacent"], help="delta pairing policy")
    ap.add_argument("--samples_per_group", type=int, default=2_000_000, help="final stored sample count per group per metric")
    ap.add_argument("--coord_budget_per_group", type=int, default=400_000,
                    help="number of coordinate positions tracked per group (for mean/noise/delta); larger => smoother")
    ap.add_argument("--tail_groups", type=str, default="",
                    help="comma-separated groups; default from config tail_groups")
    ap.add_argument("--grouping_mode", type=str, default=None, help="override grouping (basic|layer_bucket)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--backup", action="store_true", help="backup existing tails_iter*.npz as .bak before overwrite")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_config_resolved(run_dir)
    resolved = cfg.get("resolved", {}) if isinstance(cfg, dict) else {}

    # model params
    n_layer = int(cfg.get("n_layer", 12))
    n_head = int(cfg.get("n_head", 12))
    n_embd = int(cfg.get("n_embd", 768))
    block_size = int(cfg.get("block_size", 1024))
    bias = bool(cfg.get("bias", False))
    dropout = float(cfg.get("dropout", 0.0))

    # data
    data_dir = Path(cfg.get("data_dir", "data/openwebtext"))
    batch_size = int(cfg.get("batch_size", 12))

    grouping_mode = args.grouping_mode or str(cfg.get("tail_grouping_mode", "basic"))

    default_groups = cfg.get("tail_groups", ["all", "norm", "embed"])
    if args.tail_groups.strip():
        keep_groups = [g.strip() for g in args.tail_groups.split(",") if g.strip()]
    else:
        keep_groups = list(default_groups)

    device = str(args.device)
    dtype = dtype_from_str(args.dtype)

    set_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed) + 123)

    # choose checkpoint
    ckpts = list_ckpts(run_dir / "checkpoints")
    actual_iter, ckpt_path = pick_ckpt(ckpts, requested=int(args.iter), policy=str(args.ckpt_policy))
    if actual_iter != int(args.iter):
        print(f"[warn] requested iter={args.iter} but using checkpoint iter={actual_iter} via policy={args.ckpt_policy}")

    # build model and load weights
    model_config = GPTConfig(
        block_size=block_size,
        vocab_size=50304,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
    )
    model = GPT(model_config).to(device)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    # build samplers (fixed coordinate sets per group)
    sampler_seed = int(args.seed) + actual_iter * 1009
    group_samplers = build_group_samplers(
        model=model,
        grouping_mode=grouping_mode,
        n_layer=n_layer,
        keep_groups=keep_groups,
        coord_budget_per_group=int(args.coord_budget_per_group),
        seed=sampler_seed,
    )

    # buffers for mean (per group per coordinate stream)
    # We store running sums for each group as a concatenated vector matching the concatenation order of samplers.
    group_coord_sizes: Dict[str, int] = {}
    for g in keep_groups:
        group_coord_sizes[g] = sum(s.m for _, s in group_samplers[g])

    mean_sum: Dict[str, np.ndarray] = {g: np.zeros((group_coord_sizes[g],), dtype=np.float64) for g in keep_groups}
    K = int(args.k_batches)

    # Pass 1: compute mean across K batches (for mean_abs and noise_abs)
    print(f"[info] pass1: computing mean across K={K} batches for groups={keep_groups}")
    for k in range(K):
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

        X, Y = openwebtext_get_batch(data_dir, "train", batch_size, block_size, device)
        with torch.autocast(device_type=device, dtype=dtype, enabled=(dtype != torch.float32)):
            _, loss = model(X, Y)
        loss.backward()

        # extract coordinates per group, concatenate in fixed sampler order
        for g in keep_groups:
            offs = 0
            for name, s in group_samplers[g]:
                p = dict(model.named_parameters())[name]
                grad = p.grad
                if grad is None:
                    vals = np.zeros((s.m,), dtype=np.float64)
                else:
                    flat = grad.detach().flatten().float()
                    vals = s.take(flat).detach().cpu().numpy().astype(np.float64, copy=False)
                mean_sum[g][offs:offs + s.m] += vals
                offs += s.m

    mean_vec: Dict[str, np.ndarray] = {g: (mean_sum[g] / float(K)) for g in keep_groups}

    # Create outputs as capped lists per group
    grad_abs: Dict[str, List[np.ndarray]] = {g: [] for g in keep_groups}
    noise_abs: Dict[str, List[np.ndarray]] = {g: [] for g in keep_groups}
    delta_abs: Dict[str, List[np.ndarray]] = {g: [] for g in keep_groups}
    mean_abs: Dict[str, np.ndarray] = {g: np.abs(mean_vec[g]).astype(np.float32, copy=False) for g in keep_groups}

    cap = int(args.samples_per_group)

    # Pass 2: collect grad_abs and noise_abs with cap; also compute delta_abs from adjacent pairs
    print(f"[info] pass2: collecting distributions (cap per group={cap})")
    prev_coords: Optional[Dict[str, np.ndarray]] = None

    for k in range(K):
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

        X, Y = openwebtext_get_batch(data_dir, "train", batch_size, block_size, device)
        with torch.autocast(device_type=device, dtype=dtype, enabled=(dtype != torch.float32)):
            _, loss = model(X, Y)
        loss.backward()

        coords_now: Dict[str, np.ndarray] = {}
        for g in keep_groups:
            vals_cat = np.empty((group_coord_sizes[g],), dtype=np.float32)
            offs = 0
            for name, s in group_samplers[g]:
                p = dict(model.named_parameters())[name]
                grad = p.grad
                if grad is None:
                    vals = np.zeros((s.m,), dtype=np.float32)
                else:
                    flat = grad.detach().flatten().float()
                    vals = s.take(flat).detach().cpu().numpy().astype(np.float32, copy=False)
                vals_cat[offs:offs + s.m] = vals
                offs += s.m
            coords_now[g] = vals_cat

            # grad_abs cap
            append_with_cap(rng, grad_abs[g], np.abs(vals_cat), cap)

            # noise_abs cap: |g - mean|
            mu = mean_vec[g].astype(np.float32, copy=False)
            # align just in case
            L = min(vals_cat.size, mu.size)
            append_with_cap(rng, noise_abs[g], np.abs(vals_cat[:L] - mu[:L]), cap)

        # delta_abs from adjacent pairs (0&1, 2&3, ...)
        if args.pairs == "adjacent":
            if prev_coords is None:
                prev_coords = coords_now
            else:
                # compute delta for each group, cap
                for g in keep_groups:
                    a = prev_coords[g]
                    b = coords_now[g]
                    L = min(a.size, b.size)
                    append_with_cap(rng, delta_abs[g], np.abs(a[:L] - b[:L]), cap)
                prev_coords = None

    # Build npz dict
    npz: Dict[str, np.ndarray] = {}
    for g in keep_groups:
        if grad_abs[g]:
            npz[f"grad_abs__{g}"] = np.concatenate(grad_abs[g], axis=0).astype(np.float32, copy=False)
        if noise_abs[g]:
            npz[f"noise_abs__{g}"] = np.concatenate(noise_abs[g], axis=0).astype(np.float32, copy=False)
        if delta_abs[g]:
            npz[f"delta_abs__{g}"] = np.concatenate(delta_abs[g], axis=0).astype(np.float32, copy=False)
        # mean_abs is one value per tracked coordinate (may be < cap, ok)
        npz[f"mean_abs__{g}"] = mean_abs[g]

    tails_dir = mkdir_p(run_dir / "tails")
    out_path = tails_dir / f"tails_iter{actual_iter:07d}.npz"

    if args.backup and out_path.exists():
        bak = out_path.with_suffix(out_path.suffix + ".bak")
        out_path.replace(bak)
        print(f"[info] backed up old npz to {bak}")

    np.savez_compressed(out_path, **npz)
    print(f"[ok] wrote {out_path}")
    print(f"[ok] groups={keep_groups}  K={K}  cap={cap}  coord_budget={args.coord_budget_per_group}")
    print(f"[ok] keys sample: {sorted(list(npz.keys()))[:10]}{' ...' if len(npz)>10 else ''}")


if __name__ == "__main__":
    main()

