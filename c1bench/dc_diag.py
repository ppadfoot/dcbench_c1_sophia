from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .utils import JsonlWriter, sample_abs_values


@dataclass
class DCDiagConfig:
    diag_every: int = 100
    probes: int = 4
    tail_samples: int = 80000
    fp32: bool = True
    eps: float = 1e-12

    # tau_ref scheme:
    # We want tau to be a FIXED reference based on AdamW, not proportional to u_norm of current optimizer.
    # tau_ref is interpreted as a cap on ||U||, i.e. we cap E_hat by tau_ref^2.
    tau_mult: float = 10.0
    tau_min: float = 1e-8

    # If True, use tau_ref loaded from env/file; if missing, fallback to dynamic (warn via tau_source field).
    tau_mode: str = "adamw_ref"  # "adamw_ref" | "dynamic"
    tau_ref_path: Optional[str] = None  # optional; can also come from env DCBENCH_TAU_REF_PATH
    tau_ref_value: Optional[float] = None  # optional; can also come from env DCBENCH_TAU_REF
    tau_ref_filename: str = "tau_ref.json"  # stored in run_dir for AdamW runs


def _flatten_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.dot(a.reshape(-1), b.reshape(-1))


def _maybe_get_state(optimizer: torch.optim.Optimizer, p: torch.Tensor) -> Dict[str, Any]:
    return optimizer.state.get(p, {})  # type: ignore[arg-type]


def _read_tau_ref_json(path: Path) -> Optional[float]:
    try:
        j = json.loads(path.read_text(encoding="utf-8"))
        v = j.get("tau_ref", None)
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _resolve_tau_ref(cfg: DCDiagConfig, run_dir: Optional[Path]) -> Tuple[Optional[float], str]:
    """Resolve tau_ref value and its source.

    Priority:
      1) cfg.tau_ref_value
      2) env DCBENCH_TAU_REF
      3) cfg.tau_ref_path
      4) env DCBENCH_TAU_REF_PATH
      5) run_dir/tau_ref.json (if exists)
    """
    if cfg.tau_ref_value is not None:
        return float(cfg.tau_ref_value), "cfg_value"

    env_v = os.environ.get("DCBENCH_TAU_REF", "").strip()
    if env_v:
        try:
            return float(env_v), "env_value"
        except Exception:
            pass

    path = cfg.tau_ref_path
    if not path:
        path = os.environ.get("DCBENCH_TAU_REF_PATH", "").strip()
    if path:
        p = Path(path)
        v = _read_tau_ref_json(p)
        if v is not None:
            return v, "path"

    if run_dir is not None:
        p = run_dir / cfg.tau_ref_filename
        if p.exists():
            v = _read_tau_ref_json(p)
            if v is not None:
                return v, "run_dir"

    return None, "none"


@torch.no_grad()
def _sgd_direction(p: torch.Tensor, g: torch.Tensor, state: Dict[str, Any], momentum: float, nesterov: bool) -> torch.Tensor:
    if momentum <= 0:
        return g
    buf = state.get("momentum_buffer", None)
    if buf is None:
        buf = torch.zeros_like(p, memory_format=torch.preserve_format)
    buf_new = buf.mul(momentum).add(g)
    if nesterov:
        return g.add(buf_new, alpha=momentum)
    return buf_new


@torch.no_grad()
def _adam_like_direction(
    p: torch.Tensor,
    g: torch.Tensor,
    state: Dict[str, Any],
    *,
    beta1: float,
    beta2: float,
    eps: float,
    bias_correction: bool,
    trust_ratio: bool,
) -> torch.Tensor:
    exp_avg = state.get("exp_avg", None)
    exp_avg_sq = state.get("exp_avg_sq", None)
    step = int(state.get("step", 0))

    if exp_avg is None:
        exp_avg = torch.zeros_like(p, memory_format=torch.preserve_format)
    if exp_avg_sq is None:
        exp_avg_sq = torch.zeros_like(p, memory_format=torch.preserve_format)

    exp_avg_new = exp_avg.mul(beta1).add(g, alpha=1.0 - beta1)
    exp_avg_sq_new = exp_avg_sq.mul(beta2).addcmul(g, g, value=1.0 - beta2)
    step_new = step + 1

    if bias_correction:
        bc1 = 1.0 - beta1 ** step_new
        bc2 = 1.0 - beta2 ** step_new
        m_hat = exp_avg_new / max(bc1, 1e-16)
        v_hat = exp_avg_sq_new / max(bc2, 1e-16)
    else:
        m_hat = exp_avg_new
        v_hat = exp_avg_sq_new

    u = m_hat / (v_hat.sqrt().add(eps))

    if trust_ratio:
        w_norm = p.detach().float().norm(2)
        u_norm = u.detach().float().norm(2)
        if w_norm > 0 and u_norm > 0:
            u = u * (w_norm / u_norm)

    return u


@torch.no_grad()
def _lion_direction(p: torch.Tensor, g: torch.Tensor, state: Dict[str, Any], beta1: float, beta2: float) -> torch.Tensor:
    exp_avg = state.get("exp_avg", None)
    if exp_avg is None:
        exp_avg = torch.zeros_like(p, memory_format=torch.preserve_format)
    exp_avg_new = exp_avg.mul(beta1).add(g, alpha=1.0 - beta1)
    return exp_avg_new.sign()


@torch.no_grad()
def _sophia_direction(
    p: torch.Tensor,
    g: torch.Tensor,
    state: Dict[str, Any],
    *,
    beta1: float,
    rho: float,
    eps: float,
    bs: float,
) -> torch.Tensor:
    exp_avg = state.get("exp_avg", None)
    hess = state.get("hessian", None)
    if exp_avg is None:
        exp_avg = torch.zeros_like(p, memory_format=torch.preserve_format)
    if hess is None:
        hess = torch.zeros_like(p, memory_format=torch.preserve_format)
    exp_avg_new = exp_avg.mul(beta1).add(g, alpha=1.0 - beta1)
    denom = (hess * float(bs)).add(eps)
    u = exp_avg_new / denom
    return u.clamp(min=-rho, max=rho)


@torch.no_grad()
def direction_for_optimizer(
    optimizer_name: str,
    optimizer: torch.optim.Optimizer,
    p: torch.Tensor,
    g: torch.Tensor,
    *,
    bs: float,
) -> torch.Tensor:
    name = optimizer_name.lower()
    state = _maybe_get_state(optimizer, p)

    # identity match (avoid Tensor.__contains__)
    group = None
    for gr in optimizer.param_groups:
        for q in gr["params"]:
            if q is p:
                group = gr
                break
        if group is not None:
            break
    if group is None:
        raise RuntimeError("Parameter not found in optimizer.param_groups (identity check).")

    if name == "adamw":
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = group.get("eps", 1e-8)
        return _adam_like_direction(p, g, state, beta1=float(beta1), beta2=float(beta2), eps=float(eps), bias_correction=True, trust_ratio=False)

    if name == "lamb":
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = group.get("eps", 1e-6)
        return _adam_like_direction(p, g, state, beta1=float(beta1), beta2=float(beta2), eps=float(eps), bias_correction=True, trust_ratio=True)

    if name == "sgd":
        mom = float(group.get("momentum", 0.0))
        nesterov = bool(group.get("nesterov", False))
        return _sgd_direction(p, g, state, momentum=mom, nesterov=nesterov)

    if name == "lion":
        beta1, beta2 = group.get("betas", (0.9, 0.99))
        return _lion_direction(p, g, state, beta1=float(beta1), beta2=float(beta2))

    if name == "sophiag":
        beta1, _ = group.get("betas", (0.965, 0.99))
        rho = float(group.get("rho", 0.1))
        eps = float(group.get("eps", 1e-8))
        return _sophia_direction(p, g, state, beta1=float(beta1), rho=rho, eps=eps, bs=bs)

    if name == "muon":
        use_muon = bool(group.get("use_muon", False))
        if use_muon:
            # use official muon_update if available
            from muon import muon_update  # official
            beta = float(group.get("momentum", 0.95))
            buf = state.get("momentum_buffer", None)
            if buf is None:
                buf = torch.zeros_like(p, memory_format=torch.preserve_format)

            grad_tmp = g.detach().clone()
            buf_tmp = buf.detach().clone()
            upd = muon_update(grad_tmp, buf_tmp, beta=beta, ns_steps=5, nesterov=True)
            return upd.reshape(p.shape)

        # aux Adam group (use_muon=False)
        betas = group.get("betas", (0.9, 0.95))
        eps = float(group.get("eps", 1e-10))
        return _adam_like_direction(p, g, state, beta1=float(betas[0]), beta2=float(betas[1]), eps=eps, bias_correction=True, trust_ratio=False)

    raise ValueError(f"Unknown optimizer for direction: {optimizer_name}")


def _estimate_dc_core(
    *,
    model: torch.nn.Module,
    get_batch,
    optimizer_name: str,
    optimizer: torch.optim.Optimizer,
    cfg: DCDiagConfig,
    device: str,
    bs_for_sophia: float,
    tau_ref: Optional[float],
    tau_source: str,
) -> Dict[str, float]:
    P_acc = 0.0
    G_acc = 0.0
    E_acc = 0.0
    g2_acc = 0.0

    for _k in range(cfg.probes):
        X_a, Y_a = get_batch("train")
        X_b, Y_b = get_batch("train")

        model.zero_grad(set_to_none=True)
        _, loss = model(X_a, Y_a)
        loss.backward()
        grads_a = [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]

        model.zero_grad(set_to_none=True)
        _, loss = model(X_b, Y_b)
        loss.backward()
        grads_b = [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]

        for ga in grads_a:
            if ga is None:
                continue
            g2_acc += float(_flatten_dot(ga.float(), ga.float()).item())

        for p, ga, gb in zip(model.parameters(), grads_a, grads_b):
            if ga is None or gb is None:
                continue
            ga = ga.to(device)
            gb = gb.to(device)
            u = direction_for_optimizer(optimizer_name, optimizer, p, ga, bs=bs_for_sophia)
            P_acc += float(_flatten_dot(gb.float(), u.float()).item())
            G_acc += float(_flatten_dot(ga.float(), gb.float()).item())
            E_acc += float(_flatten_dot(u.float(), u.float()).item())

    model.zero_grad(set_to_none=True)

    K = float(cfg.probes)
    P_hat = P_acc / K
    G_hat = G_acc / K
    E_hat = E_acc / K
    g2_hat = g2_acc / K

    u_norm = math.sqrt(max(E_hat, 0.0))
    g_norm = math.sqrt(max(g2_hat, 0.0))

    P_pos = max(P_hat, 0.0)
    G_pos = max(G_hat, 0.0)

    # raw (no clipping)
    dc_hat_raw = (P_pos ** 2) / (G_pos * max(E_hat, 0.0) + cfg.eps)

    # reference clipping
    tau_used = None
    E_cap = E_hat
    if cfg.tau_mode == "adamw_ref":
        if tau_ref is not None:
            tau_used = max(cfg.tau_min, float(tau_ref))
            E_cap = min(E_hat, tau_used * tau_used)
        else:
            # fallback to dynamic (but record that we couldn't use ref)
            tau_used = max(cfg.tau_min, cfg.tau_mult * u_norm)
            tau_source = "no_ref_fallback_dynamic"
            E_cap = min(E_hat, tau_used * tau_used)  # still clips (now meaningful vs old)
    else:
        # dynamic (but now we actually clip by tau^2 even in dynamic mode)
        tau_used = max(cfg.tau_min, cfg.tau_mult * u_norm)
        tau_source = "dynamic"
        E_cap = min(E_hat, tau_used * tau_used)

    dc_hat = (P_pos ** 2) / (G_pos * max(E_cap, 0.0) + cfg.eps)

    c_aln_hat = P_hat / (G_hat + cfg.eps)
    cos_hat = P_hat / (math.sqrt(max(G_hat, 0.0)) * math.sqrt(max(E_hat, 0.0)) + cfg.eps)

    return {
        "P_hat": float(P_hat),
        "G_hat": float(G_hat),
        "E_hat": float(E_hat),
        "E_cap": float(E_cap),
        "g_norm": float(g_norm),
        "u_norm": float(u_norm),
        "tau_used": float(tau_used if tau_used is not None else 0.0),
        "tau_source": str(tau_source),
        "tau_ref": float(tau_ref) if tau_ref is not None else None,
        "dc_hat_raw": float(dc_hat_raw),
        "dc_hat": float(dc_hat),
        "c_aln_hat": float(c_aln_hat),
        "cos_hat": float(cos_hat),
    }


def estimate_dc_single(
    *,
    model: torch.nn.Module,
    get_batch,
    optimizer_name: str,
    optimizer: torch.optim.Optimizer,
    cfg: DCDiagConfig,
    device: str,
    bs_for_sophia: float,
    writer: Optional[JsonlWriter],
    step: int,
    run_dir: Optional[Path] = None,
) -> Dict[str, float]:
    tau_ref, tau_source = _resolve_tau_ref(cfg, run_dir)

    with torch.enable_grad():
        out = _estimate_dc_core(
            model=model,
            get_batch=get_batch,
            optimizer_name=optimizer_name,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            bs_for_sophia=bs_for_sophia,
            tau_ref=tau_ref,
            tau_source=tau_source,
        )

    rec: Dict[str, Any] = {"it": int(step), **out, "DC_hat": float(out["dc_hat"])}
    if writer is not None:
        writer.write(rec)
    return rec


def estimate_dc(
    *,
    model: torch.nn.Module,
    get_batch,
    optimizer: torch.optim.Optimizer,
    optimizer_name: str,
    device: str,
    probes: int,
    bs_tokens: float,
    fp32: bool = True,
) -> Dict[str, float]:
    cfg = DCDiagConfig(probes=int(probes), fp32=bool(fp32))
    tau_ref, tau_source = _resolve_tau_ref(cfg, run_dir=None)

    with torch.enable_grad():
        out = _estimate_dc_core(
            model=model,
            get_batch=get_batch,
            optimizer_name=optimizer_name,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            bs_for_sophia=float(bs_tokens),
            tau_ref=tau_ref,
            tau_source=tau_source,
        )

    # minimal return for selector
    return {
        "dc_hat": float(out["dc_hat"]),
        "P_hat": float(out["P_hat"]),
        "G_hat": float(out["G_hat"]),
        "E_hat": float(out["E_hat"]),
        "E_cap": float(out["E_cap"]),
        "dc_hat_raw": float(out["dc_hat_raw"]),
        "tau_used": float(out["tau_used"]),
        "tau_source": str(out["tau_source"]),
    }


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
    with torch.enable_grad():
        X, Y = get_batch("train")
        model.zero_grad(set_to_none=True)
        _, loss = model(X, Y)
        loss.backward()

        grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        g_samples = sample_abs_values(grads, cfg.tail_samples)

        u_tensors: List[torch.Tensor] = []
        for p in model.parameters():
            if p.grad is None:
                continue
            u = direction_for_optimizer(optimizer_name, optimizer, p, p.grad.detach(), bs=bs_for_sophia)
            u_tensors.append(u)
        u_samples = sample_abs_values(u_tensors, cfg.tail_samples)
        model.zero_grad(set_to_none=True)

    return {"g_abs": g_samples, "u_abs": u_samples}


class DCDiagnostics:
    def __init__(self, *, cfg: DCDiagConfig, log_writer: Optional[JsonlWriter], run_dir, tail_once: bool = True) -> None:
        self.cfg = cfg
        self.log_writer = log_writer
        self.run_dir = Path(run_dir)
        self.tail_once = tail_once
        self._tail_saved = False
        self._tau_saved = False

    def _maybe_save_tau_ref_from_adamw(self, *, optimizer_name: str, u_norm: float, step: int) -> None:
        if self.cfg.tau_mode != "adamw_ref":
            return
        if optimizer_name.lower() != "adamw":
            return
        if self._tau_saved:
            return

        tau_ref = max(self.cfg.tau_min, float(self.cfg.tau_mult) * float(u_norm))
        p = self.run_dir / self.cfg.tau_ref_filename
        payload = {"tau_ref": float(tau_ref), "it": int(step), "u_norm": float(u_norm), "note": "tau_ref = tau_mult * u_norm(adamw) @ first diag"}
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._tau_saved = True

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

        out = estimate_dc_single(
            model=model,
            get_batch=get_batch,
            optimizer_name=optimizer_name,
            optimizer=optimizer,
            cfg=self.cfg,
            device=dev,
            bs_for_sophia=float(bs_tokens),
            writer=self.log_writer,
            step=it,
            run_dir=self.run_dir,
        )

        # if this is AdamW and we are in ref mode, save tau_ref once
        self._maybe_save_tau_ref_from_adamw(optimizer_name=optimizer_name, u_norm=float(out["u_norm"]), step=it)

        if self.tail_once and (not self._tail_saved):
            try:
                tails = sample_tails_single(
                    model=model,
                    get_batch=get_batch,
                    optimizer_name=optimizer_name,
                    optimizer=optimizer,
                    cfg=self.cfg,
                    device=dev,
                    bs_for_sophia=float(bs_tokens),
                )
                out_path = self.run_dir / "tail_samples.npz"
                np.savez_compressed(out_path, **tails)
                self._tail_saved = True
            except Exception:
                pass

        return out

    def finalize(self) -> None:
        return
