"""Unified training entrypoint for C1 experiments.

Supported optimizers:
  - sgd
  - adamw
  - lion
  - lamb
  - muon
  - sophiag
  - selector (DC-based switching)

All runs log to a run directory with JSONL logs + checkpoints.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from model import GPT, GPTConfig

from c1bench.dc_diag import DCDiagConfig, DCDiagnostics
from c1bench.optim_factory import DecayGroups, make_optimizer, split_decay_params
from c1bench.selector import Selector, SelectorConfig
from c1bench.utils import JsonlWriter, mkdir_p, now_iso, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str, help="Path to configs/*.py")
    p.add_argument("--optimizer", type=str, default=None, help="sgd|adamw|lion|lamb|muon|sophiag|selector")
    p.add_argument("--run_name", type=str, default=None, help="Optional custom run name")
    p.add_argument("--out_root", type=str, default="runs/C1")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="cuda|cpu|mps (default: auto)")
    p.add_argument("--dtype", type=str, default=None, help="float16|bfloat16|float32 (default: from config)")
    p.add_argument("--diag", action="store_true", help="Enable DC diagnostics")
    p.add_argument("--no_diag", action="store_true", help="Disable DC diagnostics")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint.pt")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config entries: key=value (repeatable). Example: --set batch_size=4 --set gradient_accumulation_steps=12",
    )
    return p.parse_args()


def load_config(py_file: str) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    cfg["__file__"] = py_file
    with open(py_file, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, py_file, "exec"), cfg)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("__")}
    return cfg


def get_lr(it: int, *, learning_rate: float, warmup_iters: int, lr_decay_iters: int, min_lr: float) -> float:
    if it < warmup_iters:
        return learning_rate * it / max(1, warmup_iters)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, (lr_decay_iters - warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def openwebtext_get_batch(data_dir: Path, split: str, batch_size: int, block_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
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


@torch.no_grad()
def estimate_loss(model: GPT, data_dir: Path, batch_size: int, block_size: int, device: str, eval_iters: int) -> Dict[str, float]:
    model.eval()
    out: Dict[str, float] = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = openwebtext_get_batch(data_dir, split, batch_size, block_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = float(losses.mean())
    model.train()
    return out


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def _apply_decoupled_weight_decay(decay_groups: DecayGroups, weight_decay: float, lr: float) -> None:
    if weight_decay <= 0:
        return
    scale = 1.0 - lr * weight_decay
    for p in decay_groups.decay:
        p.mul_(scale)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    for kv in args.set:
        if "=" not in kv:
            raise ValueError(f"--set expects key=value, got: {kv!r}")
        k, v_str = kv.split("=", 1)
        k = k.strip()
        v_str = v_str.strip()
        try:
            v = ast.literal_eval(v_str)
        except Exception:
            v = v_str
        cfg[k] = v

    dataset = cfg.get("dataset", "openwebtext")
    data_dir = Path(cfg.get("data_dir", "data/openwebtext"))

    # model
    n_layer = int(cfg.get("n_layer", 12))
    n_head = int(cfg.get("n_head", 12))
    n_embd = int(cfg.get("n_embd", 768))
    block_size = int(cfg.get("block_size", 1024))
    bias = bool(cfg.get("bias", False))
    dropout = float(cfg.get("dropout", 0.0))

    # training
    batch_size = int(cfg.get("batch_size", 12))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 5))
    max_iters = int(cfg.get("max_iters", 2000))

    learning_rate = float(cfg.get("learning_rate", 6e-4))
    min_lr = float(cfg.get("min_lr", 6e-5))
    warmup_iters = int(cfg.get("warmup_iters", 200))
    lr_decay_iters = int(cfg.get("lr_decay_iters", max_iters))

    weight_decay = float(cfg.get("weight_decay", 0.1))
    grad_clip = float(cfg.get("grad_clip", 1.0))

    eval_interval = int(cfg.get("eval_interval", 200))
    eval_iters = int(cfg.get("eval_iters", 50))
    log_interval = int(cfg.get("log_interval", 10))
    save_interval = int(cfg.get("save_interval", 200))

    # optimizer hyperparams
    optimizer_name = (args.optimizer or cfg.get("optimizer", "adamw")).lower()
    betas = tuple(cfg.get("betas", (0.9, 0.95)))
    eps = float(cfg.get("eps", 1e-8))
    momentum = float(cfg.get("momentum", 0.9))
    rho = float(cfg.get("rho", 0.03))  # SophiaG
    muon_momentum = float(cfg.get("muon_momentum", 0.95))

    # selector config
    selector_candidates = list(cfg.get("selector_candidates", ["sgd", "adamw", "lion", "muon", "sophiag", "lamb"]))
    sel_every = int(cfg.get("sel_every", 100))
    sel_min_ep = int(cfg.get("sel_min_ep", 200))
    sel_patience = int(cfg.get("sel_patience", 200))
    dc_ema_rho = float(cfg.get("dc_ema_rho", 0.5))
    dc_delta = float(cfg.get("dc_delta", 0.05))
    dc_min = float(cfg.get("dc_min", 0.004))

    # diagnostics config
    diag_enabled = bool(cfg.get("diag", False))
    if args.diag:
        diag_enabled = True
    if args.no_diag:
        diag_enabled = False

    diag_every = int(cfg.get("diag_every", 100))
    diag_probes = int(cfg.get("diag_probes", 8))
    diag_tail_samples = int(cfg.get("diag_tail_samples", 80000))
    diag_tau_mult = float(cfg.get("diag_tau_mult", 10.0))

    # Optional: path to AdamW tau_ref.json for consistent E-clipping across optimizers/selector.
    tau_ref_path = cfg.get("tau_ref_path", None)
    if isinstance(tau_ref_path, str) and tau_ref_path:
        os.environ.setdefault("DCBENCH_TAU_REF_PATH", tau_ref_path)

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 1337))
    set_seed(seed)

    device = args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    dtype_str = args.dtype or cfg.get(
        "dtype", "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    )
    if dtype_str == "float32":
        dtype = torch.float32
    elif dtype_str == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    if device == "cuda" and dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("[warn] bfloat16 not supported on this GPU; using float16 instead")
        dtype_str = "float16"
        dtype = torch.float16

    run_name = args.run_name or cfg.get("run_name", None)
    if run_name is None:
        run_name = f"{dataset}__{optimizer_name}__{now_iso()}__seed{seed}"

    out_dir = mkdir_p(Path(args.out_root) / run_name)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    resolved = {
        **cfg,
        "resolved": {
            "optimizer": optimizer_name,
            "run_name": run_name,
            "out_dir": str(out_dir),
            "seed": seed,
            "device": device,
            "dtype": dtype_str,
        },
    }

    (out_dir / "config_resolved.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps({"config": resolved, "run_name": run_name, "optimizer": optimizer_name, "seed": seed}, indent=2),
        encoding="utf-8",
    )

    step_log = JsonlWriter(out_dir / "step.jsonl")
    diag_log = JsonlWriter(out_dir / "diag.jsonl") if diag_enabled else None
    sel_log = JsonlWriter(out_dir / "sel.jsonl") if optimizer_name == "selector" else None

    model_config = GPTConfig(
        block_size=block_size,
        vocab_size=50304,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
    )
    model = GPT(model_config)
    model.to(device)

    decay_groups = split_decay_params(model)

    selector: Optional[Selector] = None
    optim: Optional[torch.optim.Optimizer] = None
    optim_map: Dict[str, torch.optim.Optimizer] = {}

    if optimizer_name == "selector":
        for name in selector_candidates:
            optim_map[name] = make_optimizer(
                name,
                model,
                lr=learning_rate,
                betas=betas,
                eps=eps,
                momentum=momentum,
                rho=rho,
                muon_momentum=muon_momentum,
            )
        selector = Selector(
            SelectorConfig(
                candidates=selector_candidates,
                sel_every=sel_every,
                sel_min_ep=sel_min_ep,
                sel_patience=sel_patience,
                dc_ema_rho=dc_ema_rho,
                dc_delta=dc_delta,
                dc_min=dc_min,
                dc_probes=diag_probes,
            ),
            optims=optim_map,
            log_writer=sel_log,
        )
    else:
        optim = make_optimizer(
            optimizer_name,
            model,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            momentum=momentum,
            rho=rho,
            muon_momentum=muon_momentum,
        )

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and dtype in (torch.float16,)))
    dc_diag = None
    if diag_enabled:
        dc_diag = DCDiagnostics(
            cfg=DCDiagConfig(
                diag_every=diag_every,
                probes=diag_probes,
                tail_samples=diag_tail_samples,
                fp32=True,
                tau_mult=diag_tau_mult,
                tau_ref_path=(tau_ref_path if isinstance(tau_ref_path, str) else None),
            ),
            log_writer=diag_log,
            run_dir=out_dir,
        )

    iter_num = 0
    best_val = 1e9
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        iter_num = int(ckpt.get("iter_num", 0))
        best_val = float(ckpt.get("best_val", 1e9))
        if optimizer_name == "selector":
            for name, opt in optim_map.items():
                if name in ckpt.get("optimizers", {}):
                    opt.load_state_dict(ckpt["optimizers"][name])
            if selector is not None and ckpt.get("selector") is not None:
                selector.load_state_dict(ckpt["selector"])
        else:
            assert optim is not None
            optim.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])

    t0 = time.time()
    for it in range(iter_num, max_iters):
        lr = get_lr(it, learning_rate=learning_rate, warmup_iters=warmup_iters, lr_decay_iters=lr_decay_iters, min_lr=min_lr)

        if optimizer_name == "selector":
            for opt in optim_map.values():
                opt.zero_grad(set_to_none=True)
        else:
            assert optim is not None
            optim.zero_grad(set_to_none=True)

        lossf = 0.0
        for _micro in range(grad_accum):
            X, Y = openwebtext_get_batch(data_dir, "train", batch_size, block_size, device)
            with torch.autocast(device_type=device, dtype=dtype, enabled=(dtype != torch.float32)):
                _, loss = model(X, Y)
                loss = loss / grad_accum
            lossf += float(loss.detach())
            scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()

        if scaler.is_enabled():
            if selector is not None:
                scaler.unscale_(selector.active_optimizer)
            else:
                assert optim is not None
                scaler.unscale_(optim)

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        bs_tokens = batch_size * block_size

        if optimizer_name == "selector":
            assert selector is not None
            for _opt in selector.optimizers.values():
                _set_optimizer_lr(_opt, lr)
                for _g in _opt.param_groups:
                    _g["bs"] = bs_tokens

            selector.update_inactive_states_on_trajectory(bs_tokens=bs_tokens)

            active_opt = selector.active_optimizer
            _set_optimizer_lr(active_opt, lr)
            for g in active_opt.param_groups:
                g["bs"] = bs_tokens

            if scaler.is_enabled():
                scaler.step(active_opt)
                scaler.update()
            else:
                active_opt.step()
        else:
            assert optim is not None
            _set_optimizer_lr(optim, lr)
            for g in optim.param_groups:
                g["bs"] = bs_tokens
            if scaler.is_enabled():
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()

        _apply_decoupled_weight_decay(decay_groups, weight_decay=weight_decay, lr=lr)

        dt = time.time() - t0
        if it % log_interval == 0:
            rec = {
                "iter": it,
                "loss": float(lossf),
                "lr": float(lr),
                "time_s": float(dt),
                "optimizer": selector.active_name if selector is not None else optimizer_name,
            }
            step_log.write(rec)
            print(json.dumps(rec))

        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, data_dir, batch_size, block_size, device, eval_iters)
            rec = {"iter": it, "train_loss": losses["train"], "val_loss": losses["val"]}
            step_log.write({"event": "eval", **rec})
            print(json.dumps({"event": "eval", **rec}))
            best_val = min(best_val, float(losses["val"]))

        if it % save_interval == 0 or it == max_iters - 1:
            ckpt_path = out_dir / "checkpoints" / f"ckpt_iter{it:07d}.pt"
            ckpt: Dict[str, Any] = {
                "model": model.state_dict(),
                "iter_num": it,
                "best_val": best_val,
                "scaler": scaler.state_dict() if scaler is not None else None,
            }
            if optimizer_name == "selector":
                ckpt["optimizers"] = {name: opt.state_dict() for name, opt in optim_map.items()}
                ckpt["selector"] = selector.state_dict() if selector is not None else None
            else:
                ckpt["optimizer"] = optim.state_dict() if optim is not None else None
            torch.save(ckpt, ckpt_path)

        if dc_diag is not None:
            if optimizer_name == "selector":
                assert selector is not None
                dc_diag.maybe_run(
                    it=it,
                    model=model,
                    get_batch=lambda split: openwebtext_get_batch(data_dir, split, batch_size, block_size, device),
                    optimizer_name=selector.active_name,
                    optimizer=selector.active_optimizer,
                    bs_tokens=bs_tokens,
                )
            else:
                assert optim is not None
                dc_diag.maybe_run(
                    it=it,
                    model=model,
                    get_batch=lambda split: openwebtext_get_batch(data_dir, split, batch_size, block_size, device),
                    optimizer_name=optimizer_name,
                    optimizer=optim,
                    bs_tokens=bs_tokens,
                )

        if selector is not None:
            selector.maybe_select(
                step=it,
                model=model,
                get_batch=lambda split: openwebtext_get_batch(data_dir, split, batch_size, block_size, device),
                bs_tokens=bs_tokens,
            )

    if dc_diag is not None:
        dc_diag.finalize()

    step_log.close()
    if diag_log is not None:
        diag_log.close()
    if sel_log is not None:
        sel_log.close()


if __name__ == "__main__":
    main()
