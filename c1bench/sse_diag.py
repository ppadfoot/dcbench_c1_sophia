# c1bench/sse_diag.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from c1bench.utils import JsonlWriter, mkdir_p


@dataclass
class SSEDiagConfig:
    enabled: bool = False
    sse_every: int = 10
    samples_per_group: int = 50_000
    groups: Tuple[str, ...] = ("all",)
    grouping_mode: str = "basic"  # "basic" or "layer_bucket"
    save_every: int = 10
    seed: int = 1337
    fp32: bool = True


class _Sampler:
    """
    Very small helper: fixed indices into flattened tensor for reproducible sampling.
    """
    def __init__(self, numel: int, m: int, seed: int):
        self.numel = int(numel)
        self.m = int(min(m, numel))
        rng = np.random.default_rng(seed)
        self.idx = rng.choice(self.numel, size=self.m, replace=False) if self.m > 0 else np.zeros((0,), dtype=np.int64)

    def take(self, flat: torch.Tensor) -> torch.Tensor:
        # flat: [numel]
        if self.m == 0:
            return flat.new_empty((0,))
        return flat[self.idx]


class SSEDiagnostics:
    """
    SSE logging:
      - capture_pre(): snapshot weights (only when needed)
      - capture_post(): compute per-coordinate |g| and |U| (U approx = |Δw|/lr), and log binned-median/fits later.

    We log raw samples to:
      <run_dir>/sse/sse_iterXXXXXXX.npz

    and a small JSONL summary to:
      <run_dir>/sse.jsonl
    """

    def __init__(self, cfg: SSEDiagConfig, run_dir: Path, log_writer: Optional[JsonlWriter] = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.log_writer = log_writer
        self.sse_dir = mkdir_p(self.run_dir / "sse")
        self._pre_w: Dict[str, torch.Tensor] = {}

        # lazily built samplers per param name (or grouped)
        self._samplers: Dict[str, _Sampler] = {}
        self._seed_base = int(cfg.seed)

        # For grouping, we need a name->group mapping for parameters.
        # We'll infer it from parameter names on first use.
        self._name_to_group: Dict[str, str] = {}
        self._initialized = False

    def _infer_group(self, name: str, n_layer: Optional[int] = None) -> str:
        """
        Grouping rules for nanoGPT-like naming:
          - embed: wte/wpe/lm_head
          - norm: ln_*/ln_f
          - layer_bucket: attn/mlp early/mid/late based on transformer.h.<idx>
          - otherwise: all
        """
        nm = name.lower()

        # embed
        if any(k in nm for k in ["wte", "wpe", "lm_head"]):
            return "embed"

        # norm
        if "ln_" in nm or "lnf" in nm or "ln_f" in nm or ".ln" in nm:
            return "norm"

        if self.cfg.grouping_mode != "layer_bucket":
            return "all"

        # layer_bucket
        m = None
        # common nanoGPT: transformer.h.<i>.
        import re
        m = re.search(r"\.h\.(\d+)\.", nm)
        if m is None:
            return "all"
        li = int(m.group(1))

        # determine n_layer if unknown: caller passes if available; else guess 12.
        L = int(n_layer) if n_layer is not None else 12
        # bucket: early/mid/late thirds
        b0 = L // 3
        b1 = 2 * L // 3
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

    def _maybe_init(self, model: torch.nn.Module) -> None:
        if self._initialized:
            return

        # Try to fetch n_layer if GPTConfig-like
        n_layer = None
        if hasattr(model, "config") and hasattr(model.config, "n_layer"):
            try:
                n_layer = int(model.config.n_layer)
            except Exception:
                n_layer = None

        # build mapping for all params
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            g = self._infer_group(name, n_layer=n_layer)
            self._name_to_group[name] = g

        self._initialized = True

    def _should_run(self, it: int) -> bool:
        return bool(self.cfg.enabled) and (it % int(self.cfg.sse_every) == 0)

    @torch.no_grad()
    def capture_pre(self, it: int, model: torch.nn.Module, lr: float, device: str) -> None:
        """
        Save weights for computing Δw later. Only called each iter, but will early-exit unless it%sse_every==0.
        """
        if not self._should_run(it):
            return
        self._maybe_init(model)

        self._pre_w = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # store fp32 for stability; keep on CPU to reduce VRAM
            w = p.detach()
            if self.cfg.fp32:
                w = w.float()
            self._pre_w[name] = w.cpu().clone()

    @torch.no_grad()
    def capture_post(self, it: int, model: torch.nn.Module, lr: float, device: str) -> None:
        """
        After optimizer.step() and BEFORE decoupled weight decay:
          - collect |grad| and |Δw|/lr samples by group
          - save to npz
          - write JSONL summary
        """
        if not self._should_run(it):
            return
        self._maybe_init(model)

        # collect per group arrays
        out: Dict[str, Dict[str, List[np.ndarray]]] = {}
        groups_set = set(self.cfg.groups)

        def _push(group: str, key: str, arr: np.ndarray) -> None:
            if group not in out:
                out[group] = {"g": [], "u": []}
            out[group][key].append(arr)

        # build samplers per parameter tensor
        seed0 = self._seed_base + it * 1009

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            group = self._name_to_group.get(name, "all")
            if group not in groups_set and "all" not in groups_set:
                continue
            # group key: if group not selected but all selected, map everything to all
            grp_key = group if group in groups_set else "all"

            g = p.grad
            if g is None:
                continue

            # flatten
            g_flat = g.detach().flatten()
            w_now = p.detach()
            if self.cfg.fp32:
                g_flat = g_flat.float()
                w_now = w_now.float()

            # Δw = w_pre - w_now  (we stored pre before step)
            if name not in self._pre_w:
                continue
            w_pre = self._pre_w[name].to(w_now.device)
            dw = (w_pre - w_now).flatten()

            # effective U ~ Δw / lr (avoid lr=0 at it=0)
            denom = float(lr) if float(lr) != 0.0 else 1e-12
            u_flat = (dw / denom)

            # sampling
            key_sampler = name  # sampler per param name
            if key_sampler not in self._samplers:
                self._samplers[key_sampler] = _Sampler(numel=int(g_flat.numel()), m=self.cfg.samples_per_group, seed=seed0 + hash(name) % 100000)
            sampler = self._samplers[key_sampler]

            # take same indices for g and u
            g_s = sampler.take(g_flat).abs().cpu().numpy()
            u_s = sampler.take(u_flat).abs().cpu().numpy()

            _push(grp_key, "g", g_s)
            _push(grp_key, "u", u_s)

        # concatenate + optionally subsample to cap size
        npz: Dict[str, np.ndarray] = {}
        summary: Dict[str, Any] = {"iter": it, "lr": float(lr), "groups": {}}

        for grp, d in out.items():
            g_arr = np.concatenate(d["g"], axis=0) if d["g"] else np.array([], dtype=np.float64)
            u_arr = np.concatenate(d["u"], axis=0) if d["u"] else np.array([], dtype=np.float64)

            # Keep only positive finite
            g_arr = g_arr[np.isfinite(g_arr)]
            u_arr = u_arr[np.isfinite(u_arr)]
            g_arr = g_arr[g_arr > 0]
            u_arr = u_arr[u_arr > 0]

            npz[f"g_abs__{grp}"] = g_arr.astype(np.float32, copy=False)
            npz[f"u_abs__{grp}"] = u_arr.astype(np.float32, copy=False)

            # simple summary quantiles
            def qstats(a: np.ndarray):
                if a.size == 0:
                    return {}
                return {
                    "q50": float(np.quantile(a, 0.50)),
                    "q90": float(np.quantile(a, 0.90)),
                    "q99": float(np.quantile(a, 0.99)),
                    "q999": float(np.quantile(a, 0.999)) if a.size >= 1000 else float(np.max(a)),
                    "max": float(np.max(a)),
                    "n": int(a.size),
                }

            summary["groups"][grp] = {"g_abs": qstats(g_arr), "u_abs": qstats(u_arr)}

        # save npz
        if (it % int(self.cfg.save_every) == 0) or (it == 0):
            path = self.sse_dir / f"sse_iter{it:07d}.npz"
            np.savez_compressed(path, **npz)
            summary["npz_path"] = str(path)

        # write jsonl
        if self.log_writer is not None:
            self.log_writer.write(summary)
        else:
            # fallback: write to a default jsonl
            pass
