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

    # numerical stability
    eps: float = 1e-12
    g_floor_abs: float = 1e-6  # floor for G_used

    # tau_ref scheme (AdamW reference)
    tau_mult: float = 10.0
    tau_min: float = 1e-8
    tau_mode: str = "adamw_ref"  # "adamw_ref" | "dynamic"
    tau_ref_path: Optional[str] = None  # can also come from env DCBENCH_TAU_REF_PATH
    tau_ref_value: Optional[float] = None  # can also come from env DCBENCH_TAU_REF
    tau_ref_filename: str = "tau_ref.json"

    # better tau_ref: median of first N AdamW diag points (reduces noise)
    tau_ref_n_diags: int = 3


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
      5) run_dir/tau_ref.json
    """
    if cfg.tau_ref_value is not None:
        return float(cfg.tau_ref_value), "cfg_value"

    env_v = os.environ.get("DCBENCH_TAU_REF", "").strip()
    if env_v:
        try:
            return float(env_v), "env_value"
        except Exception:
            pass

    path = cfg.tau_ref_path or os.environ.get("DCBENCH_TAU_REF_PATH", "").strip()
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
        bc1 = 1.0 - beta1**step_new
        bc2 = 1.0 - beta2**step_new
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
        return _adam_like_direction(
            p, g, state, beta1=float(beta1), beta2=float(beta2), eps=float(eps), bias_correction=True, trust_ratio=False
        )

    if name == "lamb":
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = group.get("eps", 1e-6)
        return _adam_like_direction(
            p, g, state, beta1=float(beta1), beta2=float(beta2), eps=float(eps), bias_correction=True, trust_ratio=True
        )

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
            from muon import muon_update  # official
            beta = float(group.get("momentum", 0.95))
            buf = state.get("momentum_buffer", None)
            if buf is None:
                buf = torch.zeros_like(p, memory_format=torch.preserve_format)
            grad_tmp = g.detach().clone()
            buf_tmp = buf.detach().clone()
            upd = muon_update(grad_tmp, buf_tmp, beta=beta, ns_steps=5, nesterov=True)
            return upd.reshape(p.shape)

        betas = group.get("betas", (0.9, 0.95))
        eps = float(group.get("eps", 1e-10))
        return _adam_like_direction(
            p, g, state, beta1=float(betas[0]), beta2=float(betas[1]), eps=eps, bias_correction=True, trust_ratio=False
        )

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
    # We compute per-probe totals to robustify G (heavy tails).
    P_list: List[float] = []
    G_list: List[float] = []
    E_list: List[float] = []
    g2a_list: List[float] = []
    g2b_list: List[float] = []

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

        Pk = 0.0
        Gk = 0.0
        Ek = 0.0
        g2ak = 0.0
        g2bk = 0.0

        for p, ga, gb in zip(model.parameters(), grads_a, grads_b):
            if ga is None or gb is None:
                continue

            # Use float32 reductions for stability
            ga_f = ga.float()
            gb_f = gb.float()

            g2ak += float(_flatten_dot(ga_f, ga_f).item())
            g2bk += float(_flatten_dot(gb_f, gb_f).item())
            Gk += float(_flatten_dot(ga_f, gb_f).item())

            ga_t = ga.to(device)
            gb_t = gb.to(device)
            u = direction_for_optimizer(optimizer_name, optimizer, p, ga_t, bs=bs_for_sophia)

            u_f = u.float()
            Pk += float(_flatten_dot(gb_t.float(), u_f).item())
            Ek += float(_flatten_dot(u_f, u_f).item())

        P_list.append(Pk)
        G_list.append(Gk)
        E_list.append(Ek)
        g2a_list.append(g2ak)
        g2b_list.append(g2bk)

    model.zero_grad(set_to_none=True)

    K = float(cfg.probes)
    P_hat = float(sum(P_list) / K)
    G_hat = float(sum(G_list) / K)
    E_hat = float(sum(E_list) / K)
    g2a_hat = float(sum(g2a_list) / K)
    g2b_hat = float(sum(g2b_list) / K)

    u_norm = math.sqrt(max(E_hat, 0.0))
    g_norm = math.sqrt(max(g2a_hat, 0.0))

    # Stable G_used: mean of positive parts across probes
    G_pos_list = [max(x, 0.0) for x in G_list]
    G_pos_mean = float(sum(G_pos_list) / K)
    frac_G_nonpos = float(sum(1 for x in G_list if x <= 0.0) / max(1, len(G_list)))

    if G_pos_mean > cfg.g_floor_abs:
        G_used = G_pos_mean
        G_source = "pos_mean"
    else:
        G_used = float(cfg.g_floor_abs)
        G_source = "pos_mean_floor"

    # Positive progress (paper definition uses (P_hat)_+)
    P_pos = max(P_hat, 0.0)

    # E clipping using tau_ref
    tau_used = None
    E_cap = float(E_hat)
    if cfg.tau_mode == "adamw_ref":
        if tau_ref is not None:
            tau_used = max(cfg.tau_min, float(tau_ref))
            E_cap = min(E_cap, tau_used * tau_used)
        else:
            tau_used = max(cfg.tau_min, cfg.tau_mult * u_norm)
            tau_source = "no_ref_fallback_dynamic"
            E_cap = min(E_cap, tau_used * tau_used)
    else:
        tau_used = max(cfg.tau_min, cfg.tau_mult * u_norm)
        tau_source = "dynamic"
        E_cap = min(E_cap, tau_used * tau_used)

    # Core quantities:
    # - dc_hat_raw: no E clipping
    # - dc_hat: with E clipping
    dc_hat_raw_uncapped = (P_pos**2) / (G_used * max(E_hat, 0.0) + cfg.eps)
    dc_hat_uncapped = (P_pos**2) / (G_used * max(E_cap, 0.0) + cfg.eps)

    # Best-paper: DC is cos^2 proxy => clamp to [0,1]
    dc_hat_raw = float(max(0.0, min(1.0, dc_hat_raw_uncapped)))
    dc_hat = float(max(0.0, min(1.0, dc_hat_uncapped)))

    # Also log "certified descent proxy" without G normalization: score = P^2 / E
    score_hat_raw = float((P_pos**2) / (max(E_hat, 0.0) + cfg.eps))
    score_hat = float((P_pos**2) / (max(E_cap, 0.0) + cfg.eps))

    # alignment proxies
    c_aln_hat = float(P_hat / (G_hat + cfg.eps))
    cos_denom = (math.sqrt(max(G_hat, 0.0)) * math.sqrt(max(E_hat, 0.0)) + cfg.eps)
    cos_hat = float(P_hat / cos_denom) if cos_denom > 0 else 0.0
    cos_hat = float(max(-1.0, min(1.0, cos_hat)))

    return {
        "P_hat": float(P_hat),
        "P_pos": float(P_pos),
        "G_hat": float(G_hat),
        "G_pos_mean": float(G_pos_mean),
        "G_used": float(G_used),
        "G_source": str(G_source),
        "frac_G_nonpos": float(frac_G_nonpos),
        "E_hat": float(E_hat),
        "E_cap": float(E_cap),
        "g2a_hat": float(g2a_hat),
        "g2b_hat": float(g2b_hat),
        "g_norm": float(g_norm),
        "u_norm": float(u_norm),
        "tau_used": float(tau_used if tau_used is not None else 0.0),
        "tau_source": str(tau_source),
        "tau_ref": float(tau_ref) if tau_ref is not None else None,
        "dc_hat_raw_uncapped": float(dc_hat_raw_uncapped),
        "dc_hat_uncapped": float(dc_hat_uncapped),
        "dc_hat_raw": float(dc_hat_raw),
        "dc_hat": float(dc_hat),
        "score_hat_raw": float(score_hat_raw),
        "score_hat": float(score_hat),
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

    # what selector needs + some debugging
    return {
        "dc_hat": float(out["dc_hat"]),
        "dc_hat_raw": float(out["dc_hat_raw"]),
        "P_hat": float(out["P_hat"]),
        "G_hat": float(out["G_hat"]),
        "G_used": float(out["G_used"]),
        "frac_G_nonpos": float(out["frac_G_nonpos"]),
        "E_hat": float(out["E_hat"]),
        "E_cap": float(out["E_cap"]),
        "tau_used": float(out["tau_used"]),
        "score_hat": float(out["score_hat"]),
        "score_hat_raw": float(out["score_hat_raw"]),
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
        self._adamw_u_norm_hist: List[float] = []

    def _maybe_save_tau_ref_from_adamw(self, *, optimizer_name: str, u_norm: float, step: int) -> None:
        """Best-paper: compute tau_ref as median of first N diag u_norm values for AdamW."""
        if self.cfg.tau_mode != "adamw_ref":
            return
        if optimizer_name.lower() != "adamw":
            return
        if self._tau_saved:
            return

        self._adamw_u_norm_hist.append(float(u_norm))
        if len(self._adamw_u_norm_hist) < max(1, int(self.cfg.tau_ref_n_diags)):
            return

        u_med = float(np.median(np.array(self._adamw_u_norm_hist, dtype=np.float64)))
        tau_ref = max(self.cfg.tau_min, float(self.cfg.tau_mult) * u_med)

        p = self.run_dir / self.cfg.tau_ref_filename
        payload = {
            "tau_ref": float(tau_ref),
            "it": int(step),
            "u_norm_median": float(u_med),
            "u_norm_hist": list(self._adamw_u_norm_hist),
            "note": f"tau_ref = tau_mult * median(u_norm(adamw) over first {self.cfg.tau_ref_n_diags} diags)",
        }
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
