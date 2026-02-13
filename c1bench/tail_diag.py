"""
Heavy-tail diagnostics for gradient and gradient noise.

Goal: capture empirical distributions of |g| and |g - E[g]| (component-wise) in a way that:
- is compatible with very large models (sampling),
- is stable enough for tail plots (use K>=32 mini-batches),
- can be stratified by parameter types (decay/no_decay/norm/bias/embed/...).

This module intentionally logs *raw sampled values* (as npz) + small JSONL summaries.
Plotting is handled by tools/plot_tail_metrics.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .optim_factory import split_decay_params
from .utils import JsonlWriter


@dataclass
class TailDiagConfig:
    # Master switch
    enabled: bool = True

    # How often to run (training iterations)
    tail_every: int = 200

    # Number of independent mini-batch gradients used to estimate mean grad (and hence noise)
    k_batches: int = 32

    # Number of sampled coordinates per group (per diagnostic run)
    samples_per_group: int = 200_000

    # Which parameter groups to log (subset is fine)
    # Recommended default keeps size moderate but still answers "do norms behave differently?"
    groups: Tuple[str, ...] = ("all", "decay", "no_decay", "norm", "bias", "embed")

    # How to form parameter groups.
    # - "basic": only {all, decay, no_decay, norm, bias, embed}
    # - "layer_bucket": additionally form attention/MLP groups by layer thirds:
    #       attn_early, attn_mid, attn_late, mlp_early, mlp_mid, mlp_late
    grouping_mode: str = "basic"

    # Save sampled arrays every N diag runs (in addition to JSON summaries). Use 1 to save every time.
    save_every: int = 1

    # Sampling seed (deterministic sampling of coordinates)
    seed: int = 1337

    # How to estimate mean gradient across K batches:
    # - "mean": simple average
    # - "mom": median-of-means (more robust under heavy tails)
    mean_estimator: str = "mean"
    mom_chunks: int = 8  # only used if mean_estimator == "mom"

    # Dtype control for diagnostics (does not change model params); gradients are always accumulated in fp32.
    fp32: bool = True

    # Additionally log a simple two-batch difference noise proxy:
    #   delta := g_a - g_b  (two independent mini-batches)
    # This is useful because it does not depend on estimating mean(g) and is easy to reason about.
    # (The tail exponent is invariant to constant rescaling, so we log raw |delta|.)
    log_delta: bool = True


def _is_bias_param(name: str, p: torch.nn.Parameter) -> bool:
    return name.endswith(".bias") or name == "bias"


def _is_norm_param(name: str, p: torch.nn.Parameter) -> bool:
    # Works for this repo (LayerNorm params are named like "...ln_1.weight", "...ln_f.weight")
    n = name.lower()
    if "ln_" in n or ".ln" in n or "layernorm" in n or "norm" in n:
        return True
    return False


def _is_embed_param(name: str, p: torch.nn.Parameter) -> bool:
    n = name.lower()
    return (".wte." in n) or (".wpe." in n) or ("embedding" in n) or ("embed" in n)


def _extract_layer_idx(name: str) -> Optional[int]:
    """Extract transformer block index from parameter name.

    This repo uses names like: "transformer.h.0.attn.c_attn.weight".
    """
    m = re.search(r"\.h\.(\d+)\.", name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _bucket(layer_idx: int, n_layer: int) -> str:
    """Return one of {'early','mid','late'} based on layer thirds."""
    if n_layer <= 0:
        return "mid"
    a = n_layer // 3
    b = (2 * n_layer) // 3
    if layer_idx < a:
        return "early"
    if layer_idx < b:
        return "mid"
    return "late"


def _is_attn_param(name: str) -> bool:
    n = name.lower()
    return ".attn." in n


def _is_mlp_param(name: str) -> bool:
    n = name.lower()
    return ".mlp." in n


class _GroupSampler:
    """
    Uniform coordinate sampler over a set of parameters, without materializing a full flattened vector.
    We use multinomial allocation to keep sampling O(#params) instead of O(#samples).

    For each parameter tensor p, we sample c_p indices uniformly from [0, p.numel()).
    We then pack all samples into a single vector of length M (=samples_per_group),
    using contiguous slices per parameter.
    """

    def __init__(self, params: Sequence[torch.nn.Parameter], n_samples: int, *, rng: np.random.Generator):
        self.params: List[torch.nn.Parameter] = [p for p in params if p.requires_grad]
        self.n_samples = int(n_samples)

        if self.n_samples <= 0:
            raise ValueError("n_samples must be > 0")

        sizes = np.array([int(p.numel()) for p in self.params], dtype=np.int64)
        total = int(sizes.sum())
        if total <= 0:
            raise ValueError("No parameters with numel > 0 in this group")

        probs = sizes / total
        counts = rng.multinomial(self.n_samples, probs)

        self._slices: List[Tuple[torch.nn.Parameter, slice, torch.Tensor]] = []
        off = 0
        for p, c in zip(self.params, counts):
            c = int(c)
            if c <= 0:
                continue
            idx = rng.integers(0, int(p.numel()), size=c, dtype=np.int64)
            idx_t = torch.from_numpy(idx)  # keep on CPU; moved to device lazily
            self._slices.append((p, slice(off, off + c), idx_t))
            off += c

        # multinomial can (rarely) allocate fewer due to rounding; pad by sampling last param
        if off < self.n_samples:
            p = self.params[-1]
            c = self.n_samples - off
            idx = rng.integers(0, int(p.numel()), size=c, dtype=np.int64)
            idx_t = torch.from_numpy(idx)
            self._slices.append((p, slice(off, off + c), idx_t))
            off += c

        assert off == self.n_samples

    def accumulate(
        self,
        *,
        sum_buf: torch.Tensor,
        chosen_specs: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        batch_i: int,
    ) -> None:
        """Accumulate sampled grad values into ``sum_buf`` and write one-batch samples.

        ``chosen_specs`` is a sequence of (chosen_buf, chosen_batch_idx) pairs.
        For each coordinate j, we write the value from batch ``chosen_batch_idx[j]`` into ``chosen_buf[j]``.

        This lets us record multiple independent one-batch samples (e.g., for a two-batch delta)
        without re-gathering values.
        """
        device = sum_buf.device
        for p, sl, idx in self._slices:
            g = p.grad
            if g is None:
                # No grad -> treat as zeros
                continue
            g_flat = g.detach().reshape(-1)

            if idx.device != device:
                idx = idx.to(device=device, non_blocking=True)
                # store back to avoid repeated transfers
                # (we mutate the tuple by rebuilding it)
                # NOTE: we cannot directly mutate a tuple element; rebuild list lazily in-place:
                # handled below by local reassignment only.

            vals = torch.index_select(g_flat, 0, idx).float()

            sum_buf[sl] += vals

            # write chosen values for the coordinates whose chosen_batch == batch_i
            for chosen_buf, chosen_batch in chosen_specs:
                cb = chosen_batch[sl]
                if cb.dtype != torch.int64:
                    cb = cb.to(torch.int64)
                sel = torch.nonzero(cb == int(batch_i), as_tuple=False).flatten()
                if sel.numel() > 0:
                    chosen_slice = chosen_buf[sl]
                    chosen_slice.index_copy_(0, sel, vals.index_select(0, sel))

    def materialize_for_device(self, device: torch.device) -> None:
        """Move all stored index tensors to the given device (one-time cost)."""
        new_slices: List[Tuple[torch.nn.Parameter, slice, torch.Tensor]] = []
        for p, sl, idx in self._slices:
            if idx.device != device:
                idx = idx.to(device=device, non_blocking=True)
            new_slices.append((p, sl, idx))
        self._slices = new_slices


class TailDiagnostics:
    def __init__(self, cfg: TailDiagConfig, *, run_dir: Path, log_writer: Optional[JsonlWriter]):
        self.cfg = cfg
        self.run_dir = run_dir
        self.log_writer = log_writer

        self._tails_dir = run_dir / "tails"
        self._tails_dir.mkdir(parents=True, exist_ok=True)

        self._prepared = False
        self._rng = np.random.default_rng(int(cfg.seed))

        # Prepared state
        self._samplers: Dict[str, _GroupSampler] = {}
        self._param_groups: Dict[str, List[torch.nn.Parameter]] = {}

        # Count how many times we ran (to apply save_every)
        self._runs = 0

    def _build_param_groups(self, model: torch.nn.Module) -> Dict[str, List[torch.nn.Parameter]]:
        named = [(n, p) for (n, p) in model.named_parameters() if p.requires_grad]
        all_params = [p for _, p in named]

        # decay/no_decay split (same logic as optimizer)
        dg = split_decay_params(model)
        decay_params = set(dg.decay)
        no_decay_params = set(dg.no_decay)

        groups: Dict[str, List[torch.nn.Parameter]] = {}

        # Always-available base groups
        groups["all"] = all_params
        groups["decay"] = [p for p in all_params if p in decay_params]
        groups["no_decay"] = [p for p in all_params if p in no_decay_params]
        groups["bias"] = [p for n, p in named if _is_bias_param(n, p)]
        groups["norm"] = [p for n, p in named if _is_norm_param(n, p)]
        groups["embed"] = [p for n, p in named if _is_embed_param(n, p)]

        # Optional richer grouping
        mode = (self.cfg.grouping_mode or "basic").lower()
        if mode == "layer_bucket":
            # Determine number of layers
            n_layer = None
            if hasattr(model, "config") and hasattr(model.config, "n_layer"):
                try:
                    n_layer = int(model.config.n_layer)
                except Exception:
                    n_layer = None
            if n_layer is None:
                # Fallback: infer from parameter names
                idxs = [i for (n, _) in named for i in [_extract_layer_idx(n)] if i is not None]
                n_layer = (max(idxs) + 1) if idxs else 0

            # Build bucketed groups by scanning names
            attn_buckets = {"attn_early": [], "attn_mid": [], "attn_late": []}
            mlp_buckets = {"mlp_early": [], "mlp_mid": [], "mlp_late": []}

            for n, p in named:
                li = _extract_layer_idx(n)
                if li is None:
                    continue
                b = _bucket(li, n_layer)

                # Don't include norms/embeddings in attn/mlp buckets; keep those separate.
                if _is_norm_param(n, p) or _is_embed_param(n, p):
                    continue

                if _is_attn_param(n):
                    attn_buckets[f"attn_{b}"].append(p)
                elif _is_mlp_param(n):
                    mlp_buckets[f"mlp_{b}"].append(p)

            groups.update(attn_buckets)
            groups.update(mlp_buckets)

        # Filter to requested groups and drop empties
        out: Dict[str, List[torch.nn.Parameter]] = {}
        # cfg.groups can be list/tuple; normalize
        requested = list(self.cfg.groups)
        for g in requested:
            if g in groups and len(groups[g]) > 0:
                out[g] = groups[g]
        return out

    def _prepare(self, model: torch.nn.Module, device: torch.device) -> None:
        self._param_groups = self._build_param_groups(model)
        self._samplers = {}
        for gname, params in self._param_groups.items():
            self._samplers[gname] = _GroupSampler(params, self.cfg.samples_per_group, rng=self._rng)
            self._samplers[gname].materialize_for_device(device)
        self._prepared = True

    @torch.no_grad()
    def finalize(self) -> None:
        # currently nothing to flush besides JSONL writer handled in train.py
        return

    def maybe_run(
        self,
        *,
        it: int,
        model: torch.nn.Module,
        get_batch: Callable[[str], Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        if (not self.cfg.enabled) or (self.cfg.tail_every <= 0):
            return
        if it % int(self.cfg.tail_every) != 0:
            return
        self.run(it=it, model=model, get_batch=get_batch)

    def run(
        self,
        *,
        it: int,
        model: torch.nn.Module,
        get_batch: Callable[[str], Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        device = next(model.parameters()).device
        if not self._prepared:
            self._prepare(model, device)

        k = int(self.cfg.k_batches)
        if k <= 0:
            return

        # Allocate buffers per group
        sum_bufs: Dict[str, torch.Tensor] = {}
        chosen_bufs: Dict[str, torch.Tensor] = {}
        chosen_batch: Dict[str, torch.Tensor] = {}

        # Optional: two-batch delta buffers
        chosen_bufs_a: Dict[str, torch.Tensor] = {}
        chosen_bufs_b: Dict[str, torch.Tensor] = {}
        chosen_batch_a: Dict[str, torch.Tensor] = {}
        chosen_batch_b: Dict[str, torch.Tensor] = {}

        for gname, sampler in self._samplers.items():
            M = sampler.n_samples
            sum_bufs[gname] = torch.zeros(M, device=device, dtype=torch.float32)
            chosen_bufs[gname] = torch.empty(M, device=device, dtype=torch.float32)
            chosen_batch[gname] = torch.randint(0, k, (M,), device=device, dtype=torch.int64)

            if self.cfg.log_delta:
                chosen_bufs_a[gname] = torch.empty(M, device=device, dtype=torch.float32)
                chosen_bufs_b[gname] = torch.empty(M, device=device, dtype=torch.float32)
                a = torch.randint(0, k, (M,), device=device, dtype=torch.int64)
                b = torch.randint(0, max(1, k - 1), (M,), device=device, dtype=torch.int64)
                if k > 1:
                    # ensure b != a elementwise
                    b = b + (b >= a).to(torch.int64)
                else:
                    b = a.clone()
                chosen_batch_a[gname] = a
                chosen_batch_b[gname] = b

        # If using median-of-means, we accumulate per chunk
        use_mom = (self.cfg.mean_estimator.lower() == "mom")
        if use_mom:
            n_chunks = max(1, int(self.cfg.mom_chunks))
            if n_chunks > k:
                n_chunks = k
            # chunk size (last chunk may be slightly larger)
            chunk_edges = np.linspace(0, k, n_chunks + 1, dtype=np.int64).tolist()
            chunk_sums: Dict[str, torch.Tensor] = {}
            for gname, sampler in self._samplers.items():
                M = sampler.n_samples
                chunk_sums[gname] = torch.zeros((n_chunks, M), device=device, dtype=torch.float32)
        else:
            chunk_edges = []

        # Compute K independent gradients and update buffers
        was_training = model.training
        model.train()  # match training mode (dropout etc.)

        for bi in range(k):
            X, Y = get_batch("train")
            model.zero_grad(set_to_none=True)

            # We intentionally do not use AMP/scaler here; gradients are for diagnostics only.
            _, loss = model(X, Y)
            loss.backward()

            # accumulate into each group's buffers
            for gname, sampler in self._samplers.items():
                chosen_specs = [(chosen_bufs[gname], chosen_batch[gname])]
                if self.cfg.log_delta:
                    chosen_specs.append((chosen_bufs_a[gname], chosen_batch_a[gname]))
                    chosen_specs.append((chosen_bufs_b[gname], chosen_batch_b[gname]))

                sampler.accumulate(sum_buf=sum_bufs[gname], chosen_specs=chosen_specs, batch_i=bi)

                if use_mom:
                    # find which chunk this batch belongs to
                    # (cheap: linear scan over chunk_edges, since n_chunks is small)
                    chunk_id = None
                    for ci in range(len(chunk_edges) - 1):
                        if chunk_edges[ci] <= bi < chunk_edges[ci + 1]:
                            chunk_id = ci
                            break
                    assert chunk_id is not None
                    # Re-accumulate into chunk sums using the same sampled values.
                    # For correctness, we need the per-batch sampled values again. Re-gather:
                    # (Yes, this doubles gather cost when mean_estimator=="mom"; ok for diagnostics.)
                    # We do this by subtracting the previous sum_buf contribution? Too complex.
                    # Instead, do a second pass over slices here for each group.
                    # To keep code simple, we compute per-batch sampled vector by gathering into a temp tensor.
                    # NOTE: This is only used when mean_estimator=="mom".
                    # ----------------------------------------------------------------
                    # Build tmp vector
                    M = sampler.n_samples
                    tmp = torch.empty(M, device=device, dtype=torch.float32)
                    for p, sl, idx in sampler._slices:  # internal
                        g = p.grad
                        if g is None:
                            tmp[sl] = 0.0
                            continue
                        g_flat = g.detach().reshape(-1)
                        if idx.device != device:
                            idx = idx.to(device=device, non_blocking=True)
                        tmp[sl] = torch.index_select(g_flat, 0, idx).float()
                    chunk_sums[gname][chunk_id] += tmp

        if not was_training:
            model.eval()

        # Compute mean + noise samples, move to CPU, save
        save_arrays = (self.cfg.save_every > 0) and (self._runs % int(self.cfg.save_every) == 0)
        npz_payload: Dict[str, np.ndarray] = {}
        summary_rows: List[Dict[str, object]] = []

        for gname, sampler in self._samplers.items():
            if use_mom:
                means = chunk_sums[gname]
                # Convert chunk sums -> chunk means
                # (each chunk can have different size; handle by dividing with per-chunk counts)
                n_chunks = means.shape[0]
                chunk_counts = np.diff(chunk_edges)
                chunk_counts_t = torch.tensor(chunk_counts, device=device, dtype=torch.float32).reshape(-1, 1)
                chunk_means = means / chunk_counts_t
                # median across chunks (robust mean)
                mean_vec = torch.median(chunk_means, dim=0).values
            else:
                mean_vec = sum_bufs[gname] / float(k)

            grad_vec = chosen_bufs[gname]
            noise_vec = grad_vec - mean_vec

            delta_abs = None
            if self.cfg.log_delta:
                # two independent one-batch gradients (coordinate-wise) -> simple noise proxy
                delta_vec = chosen_bufs_a[gname] - chosen_bufs_b[gname]
                delta_abs = delta_vec.abs().detach().cpu().numpy()

            grad_abs = grad_vec.abs().detach().cpu().numpy()
            noise_abs = noise_vec.abs().detach().cpu().numpy()
            mean_abs = mean_vec.abs().detach().cpu().numpy()

            if save_arrays:
                npz_payload[f"grad_abs__{gname}"] = grad_abs
                npz_payload[f"noise_abs__{gname}"] = noise_abs
                npz_payload[f"mean_abs__{gname}"] = mean_abs
                if delta_abs is not None:
                    npz_payload[f"delta_abs__{gname}"] = delta_abs

            # Small summaries for JSONL
            def _q(a: np.ndarray, q: float) -> float:
                return float(np.quantile(a, q))

            row = {
                "event": "tail",
                "it": int(it),
                "group": gname,
                "k_batches": int(k),
                "samples": int(sampler.n_samples),
                "mean_estimator": self.cfg.mean_estimator,
                "grad_q50": _q(grad_abs, 0.50),
                "grad_q90": _q(grad_abs, 0.90),
                "grad_q99": _q(grad_abs, 0.99),
                "grad_q999": _q(grad_abs, 0.999),
                "grad_max": float(np.max(grad_abs)),
                "noise_q50": _q(noise_abs, 0.50),
                "noise_q90": _q(noise_abs, 0.90),
                "noise_q99": _q(noise_abs, 0.99),
                "noise_q999": _q(noise_abs, 0.999),
                "noise_max": float(np.max(noise_abs)),
                "delta_q50": _q(delta_abs, 0.50) if delta_abs is not None else None,
                "delta_q90": _q(delta_abs, 0.90) if delta_abs is not None else None,
                "delta_q99": _q(delta_abs, 0.99) if delta_abs is not None else None,
                "delta_q999": _q(delta_abs, 0.999) if delta_abs is not None else None,
                "delta_max": float(np.max(delta_abs)) if delta_abs is not None else None,
                "mean_q90": _q(mean_abs, 0.90),
                "mean_q99": _q(mean_abs, 0.99),
                "mean_max": float(np.max(mean_abs)),
            }
            summary_rows.append(row)

        # Save arrays once per diag run
        if save_arrays and len(npz_payload) > 0:
            npz_payload["it"] = np.array([int(it)], dtype=np.int64)
            npz_payload["k_batches"] = np.array([int(k)], dtype=np.int64)
            npz_payload["samples_per_group"] = np.array([int(self.cfg.samples_per_group)], dtype=np.int64)
            out_path = self._tails_dir / f"tails_iter{int(it):07d}.npz"
            np.savez_compressed(out_path, **npz_payload)
            for row in summary_rows:
                row["npz_file"] = str(out_path.relative_to(self.run_dir))

        # Write JSONL summaries
        if self.log_writer is not None:
            for row in summary_rows:
                self.log_writer.write(row)

        self._runs += 1
