from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .utils import JsonlWriter, sample_abs_values


@dataclass
class DCDiagConfig:
    diag_every: int = 100
    probes: int = 4  # K
    tail_samples: int = 80000
    fp32: bool = True
    eps: float = 1e-12
    tau_mult: float = 10.0
    tau_min: float = 1e-8


def _flatten_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.reshape(-1) * b.reshape(-1)).sum()


def _maybe_get_state(optimizer: torch.optim.Optimizer, p: torch.Tensor) -> Dict[str, Any]:
    st = optimizer.state.get(p, None)
    if st is None:
        return {}
    return st


def _adam_like_direction(
    *,
    p: torch.Tensor,
    g: torch.Tensor,
    state: Dict[str, Any],
    beta1: float,
    beta2: float,
    eps: float,
    bias_correction: bool,
    trust_ratio: bool,
) -> torch.Tensor:
    # exp_avg / exp_avg_sq may not exist yet
    exp_avg = state.get("exp_avg")
    exp_avg_sq = state.get("exp_avg_sq")
    if exp_avg is None:
        exp_avg = torch.zeros_like(p, memory_format=torch.preserve_format)
    if exp_avg_sq is None:
        exp_avg_sq = torch.zeros_like(p, memory_format=torch.preserve_format)

    step = int(state.get("step", 0)) + 1

    m = exp_avg.mul(beta1).add(g, alpha=1.0 - beta1)
    v = exp_avg_sq.mul(beta2).addcmul(g, g, value=1.0 - beta2)

    if bias_correction:
        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step
        m_hat = m / bc1
        v_hat = v / bc2
    else:
        m_hat = m
        v_hat = v

    update = m_hat / (v_hat.sqrt() + eps)

    if trust_ratio:
        w_norm = p.detach().float().norm(2)
        u_norm = update.detach().float().norm(2)
        if w_norm > 0 and u_norm > 0:
            tr = (w_norm / u_norm).to(update.dtype)
            update = update * tr

    return update


def _sgd_direction(*, p: torch.Tensor, g: torch.Tensor, state: Dict[str, Any], momentum: float, nesterov: bool) -> torch.Tensor:
    if momentum is None or momentum <= 0:
        return g
    buf = state.get("momentum_buffer")
    if buf is None:
        buf = torch.zeros_like(p, memory_format=torch.preserve_format)
    buf_new = buf.mul(momentum).add(g)
    if nesterov:
        return g.add(buf_new, alpha=momentum)
    return buf_new


def _lion_direction(*, p: torch.Tensor, g: torch.Tensor, state: Dict[str, Any], beta1: float, beta2: float) -> torch.Tensor:
    exp_avg = state.get("exp_avg")
    if exp_avg is None:
        exp_avg = torch.zeros_like(p, memory_format=torch.preserve_format)
    update = exp_avg.mul(beta1).add(g, alpha=1.0 - beta1)
    return update.sign()


def _sophia_direction(
    *,
    p: torch.Tensor,
    g: torch.Tensor,
    state: Dict[str, Any],
    beta1: float,
    rho: float,
    eps: float,
    bs: float,
) -> torch.Tensor:
    exp_avg = state.get("exp_avg")
    hess = state.get("hessian")
    if exp_avg is None:
        exp_avg = torch.zeros_like(p, memory_format=torch.preserve_format)
    if hess is None:
        hess = torch.zeros_like(p, memory_format=torch.preserve_format)

    exp_avg_new = exp_avg.mul(beta1).add(g, alpha=1.0 - beta1)
    denom = (rho * bs * hess).add(1e-15)
    ratio = (exp_avg_new.abs() / denom).clamp(max=1.0)
    return exp_avg_new.sign() * ratio


def _orthogonalize_matrix(x: torch.Tensor) -> torch.Tensor:
    # Polar factor via SVD: x = U S V^T => UV^T is closest orthogonal matrix (Frobenius)
    # Use float32 for stability
    x32 = x.detach().float()
    try:
        u, _, vT = torch.linalg.svd(x32, full_matrices=False)
        q = (u @ vT).to(x.dtype)
        return q
    except Exception:
        # fallback: normalize
        return x / (x.norm() + 1e-12)


def _muon_direction_best_effort(
    *,
    p: torch.Tensor,
    g: torch.Tensor,
    state: Dict[str, Any],
    momentum: float,
    betas: Tuple[float, float],
    eps: float,
) -> torch.Tensor:
    """Best-effort direction for Muon.

    The official Muon implementation may use a hybrid:
    - matrices: momentum + orthogonalization
    - vectors: Adam-like

    We detect state keys to choose a reasonable surrogate.
    """
    # If it looks Adam-like, use that
    if "exp_avg" in state and "exp_avg_sq" in state:
        return _adam_like_direction(
            p=p,
            g=g,
            state=state,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            bias_correction=True,
            trust_ratio=False,
        )

    # If it looks momentum-based, use orthogonalization for matrices
    if "momentum_buffer" in state or p.ndim >= 2:
        buf = state.get("momentum_buffer")
        if buf is None:
            buf = torch.zeros_like(p, memory_format=torch.preserve_format)
        buf_new = buf.mul(momentum).add(g)
        if p.ndim >= 2:
            return _orthogonalize_matrix(buf_new)
        return buf_new

    # Fallback: raw grad
    return g


def direction_for_optimizer(
    optimizer_name: str,
    optimizer: torch.optim.Optimizer,
    p: torch.Tensor,
    g: torch.Tensor,
    *,
    bs: float,
) -> torch.Tensor:
    """Compute the *effective direction* U for a single parameter tensor.

    This is used ONLY for diagnostics/scoring. The real training step uses the
    official optimizer implementation.
    """
    name = optimizer_name.lower()
    state = _maybe_get_state(optimizer, p)
    group = None
    for gr in optimizer.param_groups:
        if p in gr["params"]:
            group = gr
            break
    if group is None:
        raise RuntimeError("Parameter not found in optimizer.param_groups")

    if name == "adamw":
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = group.get("eps", 1e-8)
        return _adam_like_direction(p=p, g=g, state=state, beta1=beta1, beta2=beta2, eps=eps, bias_correction=True, trust_ratio=False)

    if name == "lamb":
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = group.get("eps", 1e-6)
        return _adam_like_direction(p=p, g=g, state=state, beta1=beta1, beta2=beta2, eps=eps, bias_correction=True, trust_ratio=True)

    if name == "sgd":
        mom = float(group.get("momentum", 0.0))
        nesterov = bool(group.get("nesterov", False))
        return _sgd_direction(p=p, g=g, state=state, momentum=mom, nesterov=nesterov)

    if name == "lion":
        beta1, beta2 = group.get("betas", (0.9, 0.99))
        return _lion_direction(p=p, g=g, state=state, beta1=beta1, beta2=beta2)

    if name == "sophiag":
        beta1, _ = group.get("betas", (0.965, 0.99))
        rho = float(group.get("rho", 0.1))
        eps = float(group.get("eps", 1e-8))
        return _sophia_direction(p=p, g=g, state=state, beta1=beta1, rho=rho, eps=eps, bs=bs)

    if name == "muon":
        mom = float(group.get("momentum", 0.95))
        betas = group.get("betas", (0.9, 0.999))
        eps = float(group.get("eps", 1e-8))
        return _muon_direction_best_effort(p=p, g=g, state=state, momentum=mom, betas=betas, eps=eps)

    raise ValueError(f"Unknown optimizer for direction: {optimizer_name}")


@torch.no_grad()
def estimate_dc_single(
    *,
    model: torch.nn.Module,
    get_batch,
    optimizer_name: str,
    optimizer: torch.optim.Optimizer,
    cfg: DCDiagConfig,
    device: str,
    bs_for_sophia: float,
    writer: JsonlWriter,
    step: int,
) -> Dict[str, float]:
    """Estimate DC on current (w,s) for a single optimizer.

    Uses the two-sample / copy-state idea from the paper (we do NOT mutate
    optimizer state here; we only read it).
    """
    P_acc = 0.0
    G_acc = 0.0
    E_acc = 0.0

    # We compute gradients in FP32 for robustness by default.
    # NOTE: this does not change model parameters; it only does extra backward passes.
    for k in range(cfg.probes):
        # --- sample independent mini-batches ---
        X_a, Y_a = get_batch("train")
        X_b, Y_b = get_batch("train")

        # --- compute g_a ---
        model.zero_grad(set_to_none=True)
        logits, loss = model(X_a, Y_a)
        loss.backward()
        grads_a = [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]

        # --- compute g_b ---
        model.zero_grad(set_to_none=True)
        logits, loss = model(X_b, Y_b)
        loss.backward()
        grads_b = [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]

        # --- compute U direction + accumulate dot products ---
        for p, ga, gb in zip(model.parameters(), grads_a, grads_b):
            if ga is None or gb is None:
                continue
            ga = ga.to(device)
            gb = gb.to(device)
            u = direction_for_optimizer(optimizer_name, optimizer, p, ga, bs=bs_for_sophia)

            # Accumulate in float64 for numerical stability
            P_acc += float(_flatten_dot(gb.float(), u.float()).item())
            G_acc += float(_flatten_dot(ga.float(), gb.float()).item())
            E_acc += float(_flatten_dot(u.float(), u.float()).item())

    K = float(cfg.probes)
    P_hat = P_acc / K
    G_hat = G_acc / K
    E_hat = E_acc / K

    # winsorize E_hat (estimator-only) for stability
    u_norm = math.sqrt(max(E_hat, 0.0))
    tau = max(cfg.tau_min, cfg.tau_mult * u_norm)
    dc_hat = (max(P_hat, 0.0) ** 2) / (max(G_hat, 0.0) * min(E_hat, tau * tau) + cfg.eps)

    # alignment proxies
    c_aln_hat = P_hat / (G_hat + cfg.eps)
    cos_hat = P_hat / (math.sqrt(max(G_hat, 0.0)) * math.sqrt(max(E_hat, 0.0)) + cfg.eps)

    out = {
        "step": int(step),
        "P_hat": float(P_hat),
        "G_hat": float(G_hat),
        "E_hat": float(E_hat),
        "U_norm": float(u_norm),
        "tau": float(tau),
        "DC_hat": float(dc_hat),
        "c_aln_hat": float(c_aln_hat),
        "cos_hat": float(cos_hat),
    }

    writer.write(out)
    return out


@torch.no_grad()
def sample_tails_single(
    *,
    model: torch.nn.Module,
    get_batch,
    optimizer_name: str,
    optimizer: torch.optim.Optimizer,
    cfg: DCDiagConfig,
    device: str,
    bs_for_sophia: float,
) -> Dict[str, np.ndarray]:
    """One tail-sample snapshot for gradients and step-directions."""
    X, Y = get_batch("train")
    model.zero_grad(set_to_none=True)
    logits, loss = model(X, Y)
    loss.backward()

    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    g_samples = sample_abs_values(grads, cfg.tail_samples)

    # step-direction samples use U computed from current grad
    u_tensors: List[torch.Tensor] = []
    for p in model.parameters():
        if p.grad is None:
            continue
        u = direction_for_optimizer(optimizer_name, optimizer, p, p.grad.detach(), bs=bs_for_sophia)
        u_tensors.append(u)
    u_samples = sample_abs_values(u_tensors, cfg.tail_samples)

    return {"g_abs": g_samples, "u_abs": u_samples}

class DCDiagnostics:
    """Periodic diagnostics helper.

    The training loop calls maybe_run() each step.
    We log scalar diagnostics into diag.jsonl and (optionally) save
    a single heavy-tail snapshot into an .npz file for CCDF/Hill plots.
    """

    def __init__(self, *, cfg: DCDiagConfig, log_writer: JsonlWriter, run_dir, tail_once: bool = True) -> None:
        self.cfg = cfg
        self.log_writer = log_writer
        self.run_dir = run_dir
        self.tail_once = tail_once
        self._tail_saved = False

    def maybe_run(
        self,
        *,
        it: int,
        model: torch.nn.Module,
        get_batch,
        optimizer_name: str,
        optimizer: torch.optim.Optimizer,
        bs_tokens: int,
        device: Optional[str] = None,
    ) -> Optional[Dict[str, float]]:
        if it % max(1, self.cfg.diag_every) != 0:
            return None

        dev = device or ("cuda" if next(model.parameters()).is_cuda else "cpu")

        # Sophia expects a batch-size scaling; we pass tokens-per-step as a best-effort proxy.
        bs_for_sophia = float(bs_tokens)

        out = estimate_dc_single(
            model=model,
            get_batch=get_batch,
            optimizer_name=optimizer_name,
            optimizer=optimizer,
            cfg=self.cfg,
            device=dev,
            bs_for_sophia=bs_for_sophia,
            writer=self.log_writer,
            step=it,
        )

        # Save one tail snapshot early (the first time diagnostics run)
        if self.tail_once and (not self._tail_saved):
            try:
                tails = sample_tails_single(
                    model=model,
                    get_batch=get_batch,
                    optimizer_name=optimizer_name,
                    optimizer=optimizer,
                    cfg=self.cfg,
                    device=dev,
                    bs_for_sophia=bs_for_sophia,
                )
                out_path = (self.run_dir / "out" / "tail_samples_first_diag.npz")
                np.savez_compressed(out_path, **tails)
                self._tail_saved = True
            except Exception:
                pass

        return out

    def finalize(self) -> None:
        return
