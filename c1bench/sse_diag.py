# c1bench/sse_diag.py
from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from c1bench.utils import JsonlWriter, mkdir_p


@dataclass
class SSEDiagConfig:
    enabled: bool = False
    sse_every: int = 10
    samples_per_group: int = 50_000  # cap PER GROUP (paired (g,u) samples)
    groups: Tuple[str, ...] = ("all",)
    grouping_mode: str = "basic"  # "basic" or "layer_bucket"
    save_every: int = 10          # save npz every N iters (independent from sse_every)
    seed: int = 1337
    fp32: bool = True


def _stable_hash(s: str) -> int:
    """Deterministic hash across runs (unlike Python's built-in hash)."""
    return int(zlib.adler32(s.encode("utf-8")) & 0xFFFFFFFF)


class _Sampler:
    """
    Fixed coordinate indices into a flattened tensor.
    Used ONLY to make g and u use the same coordinates for a given tensor.
    """
    def __init__(self, numel: int, m: int, seed: int):
        self.numel = int(numel)
        self.m = int(min(m, numel))
        rng = np.random.default_rng(int(seed))
        if self.m > 0:
            self.idx = rng.choice(self.numel, size=self.m, replace=False).astype(np.int64)
        else:
            self.idx = np.zeros((0,), dtype=np.int64)

    def take(self, flat: torch.Tensor) -> torch.Tensor:
        if self.m == 0:
            return flat.new_empty((0,))
        return flat[self.idx]


class SSEDiagnostics:
    """
    SSE logging: collect paired samples (|g_i|, |U_i|) to audit sublinear/plateau behavior.

    - capture_pre(): snapshot weights (only on it % sse_every == 0)
    - capture_post(): after optimizer.step() and BEFORE decoupled weight decay:
         U_i ~ |Δw_i| / lr

    Saves raw paired samples per group to:
      <run_dir>/sse/sse_iterXXXXXXX.npz
        keys: g_abs__<group>, u_abs__<group>

    Also writes a small JSONL summary to:
      <run_dir>/sse.jsonl  (if log_writer provided)
    """

    def __init__(self, cfg: SSEDiagConfig, run_dir: Path, log_writer: Optional[JsonlWriter] = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.log_writer = log_writer
        self.sse_dir = mkdir_p(self.run_dir / "sse")

        self._pre_w: Dict[str, torch.Tensor] = {}
        self._samplers: Dict[str, _Sampler] = {}
        self._seed_base = int(cfg.seed)

        self._name_to_group: Dict[str, str] = {}
        self._initialized = False

    def _infer_group(self, name: str, n_layer: Optional[int] = None) -> str:
        nm = name.lower()

        # embed
        if any(k in nm for k in ["wte", "wpe", "lm_head"]):
            return "embed"
        # norm
        if "ln_" in nm or "lnf" in nm or "ln_f" in nm or ".ln" in nm:
            return "norm"

        if self.cfg.grouping_mode != "layer_bucket":
            return "all"

        import re
        m = re.search(r"\.h\.(\d+)\.", nm)  # transformer.h.<i>.
        if m is None:
            return "all"
        li = int(m.group(1))
        L = int(n_layer) if n_layer is not None else 12
        b0 = L // 3
        b1 = 2 * L // 3
        stage = "early" if li < b0 else ("mid" if li < b1 else "late")

        if ".attn." in nm:
            return f"attn_{stage}"
        if ".mlp." in nm:
            return f"mlp_{stage}"
        return "all"

    def _maybe_init(self, model: torch.nn.Module) -> None:
        if self._initialized:
            return
        n_layer = None
        if hasattr(model, "config") and hasattr(model.config, "n_layer"):
            try:
                n_layer = int(model.config.n_layer)
            except Exception:
                n_layer = None

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self._name_to_group[name] = self._infer_group(name, n_layer=n_layer)
        self._initialized = True

    def _should_run(self, it: int) -> bool:
        return bool(self.cfg.enabled) and (it % int(self.cfg.sse_every) == 0)

    @torch.no_grad()
    def capture_pre(self, it: int, model: torch.nn.Module, lr: float, device: str) -> None:
        if not self._should_run(it):
            return
        self._maybe_init(model)

        self._pre_w = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            w = p.detach()
            if self.cfg.fp32:
                w = w.float()
            self._pre_w[name] = w.cpu().clone()

    @torch.no_grad()
    def capture_post(self, it: int, model: torch.nn.Module, lr: float, device: str) -> None:
        if not self._should_run(it):
            return
        self._maybe_init(model)

        groups_set = set(self.cfg.groups)
        want_all = ("all" in groups_set)

        # accumulate paired samples per group
        acc: Dict[str, Dict[str, List[np.ndarray]]] = {}

        def _push(grp: str, g_arr: np.ndarray, u_arr: np.ndarray) -> None:
            if grp not in acc:
                acc[grp] = {"g": [], "u": []}
            acc[grp]["g"].append(g_arr)
            acc[grp]["u"].append(u_arr)

        seed0 = self._seed_base + it * 1009
        denom = float(lr) if float(lr) != 0.0 else 1e-12

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            g = p.grad
            if g is None:
                continue
            if name not in self._pre_w:
                continue

            group = self._name_to_group.get(name, "all")

            record_specific = (group in groups_set) and (group != "all")
            record_all = want_all
            if not record_specific and not record_all:
                continue

            g_flat = g.detach().flatten()
            w_now = p.detach()
            if self.cfg.fp32:
                g_flat = g_flat.float()
                w_now = w_now.float()

            w_pre = self._pre_w[name].to(w_now.device)
            dw = (w_pre - w_now).flatten()
            u_flat = (dw / denom)

            # sampler per param name (stable seed); per-param m limited to avoid huge blow-up
            if name not in self._samplers:
                s = seed0 + (_stable_hash(name) % 1_000_003)
                m_param = int(min(self.cfg.samples_per_group, 4096))
                self._samplers[name] = _Sampler(numel=int(g_flat.numel()), m=m_param, seed=s)
            sampler = self._samplers[name]

            g_s = sampler.take(g_flat).abs().cpu().numpy().astype(np.float32, copy=False)
            u_s = sampler.take(u_flat).abs().cpu().numpy().astype(np.float32, copy=False)

            if record_specific:
                _push(group, g_s, u_s)
            if record_all:
                _push("all", g_s, u_s)

        # finalize per group (paired cap)
        npz: Dict[str, np.ndarray] = {}
        summary: Dict[str, Any] = {"iter": int(it), "lr": float(lr), "groups": {}}

        rng_grp = np.random.default_rng(self._seed_base + 17 * it)

        for grp, d in acc.items():
            g_arr = np.concatenate(d["g"], axis=0) if d["g"] else np.array([], dtype=np.float32)
            u_arr = np.concatenate(d["u"], axis=0) if d["u"] else np.array([], dtype=np.float32)

            g_arr = g_arr[np.isfinite(g_arr)]
            u_arr = u_arr[np.isfinite(u_arr)]
            n = int(min(g_arr.size, u_arr.size))
            g_arr = g_arr[:n]; u_arr = u_arr[:n]
            m = (g_arr > 0) & (u_arr > 0)
            g_arr = g_arr[m]; u_arr = u_arr[m]
            n = int(min(g_arr.size, u_arr.size))
            g_arr = g_arr[:n]; u_arr = u_arr[:n]

            cap = int(self.cfg.samples_per_group)
            if cap > 0 and n > cap:
                idx = rng_grp.choice(n, size=cap, replace=False)
                g_arr = g_arr[idx]
                u_arr = u_arr[idx]

            npz[f"g_abs__{grp}"] = g_arr.astype(np.float32, copy=False)
            npz[f"u_abs__{grp}"] = u_arr.astype(np.float32, copy=False)

            def qstats(a: np.ndarray) -> Dict[str, Any]:
                if a.size == 0:
                    return {"n": 0}
                return {
                    "n": int(a.size),
                    "q50": float(np.quantile(a, 0.50)),
                    "q90": float(np.quantile(a, 0.90)),
                    "q99": float(np.quantile(a, 0.99)),
                    "max": float(np.max(a)),
                }

            summary["groups"][grp] = {"g_abs": qstats(g_arr), "u_abs": qstats(u_arr)}

        if (it % int(self.cfg.save_every) == 0) or (it == 0):
            path = self.sse_dir / f"sse_iter{it:07d}.npz"
            np.savez_compressed(path, **npz)
            summary["npz_path"] = str(path)

        if self.log_writer is not None:
            self.log_writer.write(summary)
            