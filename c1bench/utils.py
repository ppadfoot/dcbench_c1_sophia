from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set python/numpy/torch seeds (single-process)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_iso() -> str:
    """UTC timestamp like 20260129T123456Z."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def mkdir_p(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class JsonlWriter:
    """Small JSONL writer with immediate flush (safe for long runs)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", encoding="utf-8")

    def write(self, obj: Dict[str, Any]) -> None:
        self._f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def global_grad_norm(parameters: Sequence[torch.nn.Parameter], norm_type: float = 2.0) -> float:
    """Compute global grad norm (like torch.nn.utils.clip_grad_norm_ does)."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0

    # Compute on FP32 for stability
    if norm_type == 2.0:
        total_sq = 0.0
        for g in grads:
            gn = g.detach().float().norm(2).item()
            total_sq += gn * gn
        return float(total_sq ** 0.5)

    # Generic p-norm
    total = 0.0
    for g in grads:
        total += float(g.detach().float().norm(norm_type).item() ** norm_type)
    return float(total ** (1.0 / norm_type))


def clip_grad_global_norm_(
    parameters: Sequence[torch.nn.Parameter],
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    """Clip gradients by global norm. Returns the pre-clip norm."""
    # torch's clip_grad_norm_ already does the right thing (in-place)
    pre = global_grad_norm(parameters, norm_type=norm_type)
    if max_norm is None or max_norm <= 0:
        return pre
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm, norm_type=norm_type)
    return pre


def sample_abs_values(
    tensors: Sequence[torch.Tensor],
    k: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> np.ndarray:
    """Sample |values| from a list of tensors (uniform over flattened entries).

    This is used for heavy-tail CCDF/Hill diagnostics without storing full tensors.
    """
    if k <= 0:
        return np.empty((0,), dtype=np.float32)

    flats: list[torch.Tensor] = []
    for t in tensors:
        if t is None:
            continue
        # Keep on CPU for cheap numpy conversion
        flats.append(t.detach().reshape(-1).abs().cpu())

    if not flats:
        return np.empty((0,), dtype=np.float32)

    flat = torch.cat(flats, dim=0)
    n = int(flat.numel())
    if n == 0:
        return np.empty((0,), dtype=np.float32)

    kk = min(k, n)
    if generator is None:
        idx = torch.randint(0, n, (kk,))
    else:
        idx = torch.randint(0, n, (kk,), generator=generator)

    out = flat[idx].numpy().astype(np.float32, copy=False)
    return out
