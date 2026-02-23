#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/step_tail_from_ckpt.py

Compute "tails of step" U from a checkpoint WITHOUT retraining:
- loads model + optimizer state at (or near) iter
- draws K probe batches
- computes step direction U on COPY of optimizer state (so we don't mutate real state)
- saves npz: tails_step_iterXXXXXXX.npz under <run_dir>/tails/

Fixes:
- robust import path (repo root in sys.path)
- choose optimizer from config_resolved["resolved"]["optimizer"] (if available)
- support missing checkpoints at requested iters via --ckpt_policy {exact,floor,nearest}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# -----------------------------
# Fix: ensure repo root is importable
# -----------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model import GPT, GPTConfig  # noqa: E402
from c1bench.optim_factory import make_optimizer  # noqa: E402
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
# Grouping (match tail_diag logic)
# -----------------------------
def infer_group(name: str, grouping_mode: str, n_layer: int) -> str:
    nm = name.lower()

    if any(k in nm for k in ["wte", "wpe", "lm_head"]):
        return "embed"

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
# Sampling helper (fixed indices)
# -----------------------------
class Sampler:
    def __init__(self, numel: int, m: int, seed: int):
        self.numel = int(numel)
        self.m = int(min(m, numel))
        rng = np.random.default_rng(seed)
        self.idx = rng.choice(self.numel, size=self.m, replace=False) if self.m > 0 else np.zeros((0,), dtype=np.int64)

    def take(self, flat: torch.Tensor) -> torch.Tensor:
        if self.m == 0:
            return flat.new_empty((0,))
        return flat[self.idx]


# -----------------------------
# Config / checkpoint helpers
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
    """
    Return {iter: path} for ckpt_iterXXXXXXX.pt
    """
    out: Dict[int, Path] = {}
    for p in sorted(ckpt_dir.glob("ckpt_iter*.pt")):
        m = re.search(r"ckpt_iter(\d+)\.pt", p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def pick_ckpt(ckpts: Dict[int, Path], requested: int, policy: str) -> tuple[int, Path]:
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
            # fallback to smallest
            return iters[0], ckpts[iters[0]]
        chosen = le[-1]
        return chosen, ckpts[chosen]

    if policy == "nearest":
        chosen = min(iters, key=lambda i: abs(i - requested))
        return chosen, ckpts[chosen]

    raise ValueError(f"Unknown ckpt_policy={policy}")


# -----------------------------
# Main
# -----------------------------
@torch.no_grad()
def snapshot_weights(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        out[name] = p.detach().float().clone()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=str)
    ap.add_argument("--iter", required=True, type=int, help="requested iter (may be missing; use ckpt_policy)")
    ap.add_argument("--ckpt_policy", default="floor", choices=["exact", "floor", "nearest"],
                    help="how to choose checkpoint if exact iter is missing (default floor)")
    ap.add_argument("--k_batches", type=int, default=32)
    ap.add_argument("--samples_per_group", type=int, default=200000)
    ap.add_argument("--groups", type=str, default="", help="comma-separated groups to keep (default from config)")
    ap.add_argument("--grouping_mode", type=str, default=None, help="override grouping mode (basic|layer_bucket)")
    ap.add_argument("--log_delta", action="store_true", help="also compute step_delta_abs")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    set_seed(int(args.seed))

    cfg = load_config_resolved(run_dir)
    resolved = cfg.get("resolved", {}) if isinstance(cfg, dict) else {}

    # IMPORTANT: prefer resolved optimizer if present
    optimizer_name = str(resolved.get("optimizer", cfg.get("optimizer", "adamw"))).lower()

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

    # optimizer hparams (use cfg defaults)
    learning_rate = float(cfg.get("learning_rate", 6e-4))
    betas = tuple(cfg.get("betas", (0.9, 0.95)))
    eps = float(cfg.get("eps", 1e-8))
    momentum = float(cfg.get("momentum", 0.9))
    rho = float(cfg.get("rho", 0.03))
    muon_momentum = float(cfg.get("muon_momentum", 0.95))

    grouping_mode = args.grouping_mode or str(cfg.get("tail_grouping_mode", "basic"))
    default_groups = cfg.get("tail_groups", ["all", "norm", "embed"])
    keep_groups = [g.strip() for g in args.groups.split(",") if g.strip()] if args.groups.strip() else list(default_groups)

    device = args.device
    dtype = dtype_from_str(args.dtype)

    # choose checkpoint
    ckpt_dir = run_dir / "checkpoints"
    ckpts = list_ckpts(ckpt_dir)
    actual_iter, ckpt_path = pick_ckpt(ckpts, requested=int(args.iter), policy=str(args.ckpt_policy))

    if actual_iter != int(args.iter):
        print(f"[warn] requested iter={args.iter} but using checkpoint iter={actual_iter} via policy={args.ckpt_policy}")

    # Build model
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

    # Build optimizer and load state
    opt = make_optimizer(
        optimizer_name,
        model,
        lr=learning_rate,
        betas=betas,
        eps=eps,
        momentum=momentum,
        rho=rho,
        muon_momentum=muon_momentum,
    )
    if "optimizer" not in ckpt or ckpt["optimizer"] is None:
        raise RuntimeError("Checkpoint has no 'optimizer' state; cannot compute step tails with moments.")
    opt.load_state_dict(ckpt["optimizer"])

    print(f"[info] optimizer={optimizer_name}  ckpt_iter={actual_iter}  k_batches={args.k_batches}")

    # name->group
    name_to_group: Dict[str, str] = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            name_to_group[name] = infer_group(name, grouping_mode=grouping_mode, n_layer=n_layer)

    # samplers per param
    samplers: Dict[str, Sampler] = {}
    seed0 = int(args.seed) + actual_iter * 1009

    def sampler_for(name: str, numel: int) -> Sampler:
        if name not in samplers:
            samplers[name] = Sampler(numel=numel, m=args.samples_per_group, seed=seed0 + (hash(name) % 100000))
        return samplers[name]

    # per-probe samples for mean/noise/delta
    probe_u: Dict[str, List[np.ndarray]] = {g: [] for g in keep_groups}

    def sample_U_from_pre(pre_w: Dict[str, torch.Tensor], lr_used: float) -> Dict[str, np.ndarray]:
        out: Dict[str, List[np.ndarray]] = {g: [] for g in keep_groups}
        denom = lr_used if lr_used != 0.0 else 1e-12

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            grp = name_to_group.get(name, "all")
            if grp not in keep_groups:
                continue

            w_pre = pre_w[name].to(p.device)
            w_now = p.detach().float()
            dw = (w_pre - w_now).flatten()
            u_flat = (dw / denom).abs()

            s = sampler_for(name, int(u_flat.numel()))
            vals = s.take(u_flat).detach().cpu().numpy().astype(np.float32, copy=False)
            out[grp].append(vals)

        out2: Dict[str, np.ndarray] = {}
        for g in keep_groups:
            out2[g] = np.concatenate(out[g], axis=0) if out[g] else np.array([], dtype=np.float32)
        return out2

    # probes
    for k in range(int(args.k_batches)):
        pre_w = snapshot_weights(model)

        opt.zero_grad(set_to_none=True)
        X, Y = openwebtext_get_batch(data_dir, "train", batch_size, block_size, device)
        with torch.autocast(device_type=device, dtype=dtype, enabled=(dtype != torch.float32)):
            _, loss = model(X, Y)
        loss.backward()

        # copy optimizer state and step once on a copy
        opt_sd = opt.state_dict()
        opt_k = make_optimizer(
            optimizer_name, model, lr=learning_rate, betas=betas, eps=eps,
            momentum=momentum, rho=rho, muon_momentum=muon_momentum
        )
        opt_k.load_state_dict(opt_sd)
        opt_k.step()

        lr_used = float(opt_k.param_groups[0].get("lr", learning_rate))
        group_vals = sample_U_from_pre(pre_w, lr_used=lr_used)

        for g in keep_groups:
            if group_vals[g].size > 0:
                probe_u[g].append(group_vals[g])

        # restore weights
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(pre_w[name].to(p.device, dtype=p.dtype))

    # assemble outputs
    npz: Dict[str, np.ndarray] = {}
    for g in keep_groups:
        probes = [a for a in probe_u[g] if a.size > 0]
        if not probes:
            continue
        min_len = min(a.size for a in probes)
        probes = [a[:min_len] for a in probes]

        step_abs = np.concatenate(probes, axis=0)
        mean_u = np.mean(np.stack(probes, axis=0), axis=0)
        step_mean_abs = np.abs(mean_u).astype(np.float32, copy=False)
        step_noise_abs = np.concatenate([np.abs(a - mean_u) for a in probes], axis=0).astype(np.float32, copy=False)

        npz[f"step_abs__{g}"] = step_abs.astype(np.float32, copy=False)
        npz[f"step_mean_abs__{g}"] = step_mean_abs
        npz[f"step_noise_abs__{g}"] = step_noise_abs

        if args.log_delta and len(probes) >= 2:
            step_delta_abs = np.abs(probes[0] - probes[1]).astype(np.float32, copy=False)
            npz[f"step_delta_abs__{g}"] = step_delta_abs

    tails_dir = mkdir_p(run_dir / "tails")
    out_path = tails_dir / f"tails_step_iter{actual_iter:07d}.npz"
    np.savez_compressed(out_path, **npz)
    print(f"[ok] wrote {out_path} (requested iter={args.iter}, used ckpt iter={actual_iter})")
    print(f"[ok] keys: {sorted(list(npz.keys()))[:10]}{' ...' if len(npz)>10 else ''}")


if __name__ == "__main__":
    main()
    