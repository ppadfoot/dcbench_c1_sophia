from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch


OptimizerName = Literal["sgd", "adamw", "lion", "sophiag", "lamb", "muon"]


@dataclass
class DecayGroups:
    decay: List[torch.nn.Parameter]
    no_decay: List[torch.nn.Parameter]


def split_decay_params(model: torch.nn.Module) -> DecayGroups:
    """Split parameters into (decay, no_decay) like nanoGPT.

    We do NOT rely on optimizer-internal `weight_decay`. Instead, we apply
    decoupled weight decay externally, and we typically don't decay:
    biases, LayerNorm/BatchNorm parameters, and similar scale/bias terms.
    """
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # common exclusions
        if name.endswith(".bias"):
            no_decay.append(p)
        elif name.endswith(".weight") and ("ln" in name.lower() or "norm" in name.lower()):
            no_decay.append(p)
        elif "bias" in name.lower() and p.ndim == 1:
            no_decay.append(p)
        else:
            decay.append(p)

    return DecayGroups(decay=decay, no_decay=no_decay)


def apply_decoupled_weight_decay(groups: DecayGroups, weight_decay: float, lr: float) -> None:
    """Apply decoupled weight decay (AdamW-style) to *decay* parameters only."""
    if weight_decay is None or weight_decay <= 0:
        return
    if lr is None or lr <= 0:
        return
    with torch.no_grad():
        for p in groups.decay:
            p.mul_(1.0 - lr * weight_decay)


def _import_symbol(module: str, symbol: str):
    mod = __import__(module, fromlist=[symbol])
    return getattr(mod, symbol)


def make_optimizer(
    name: OptimizerName,
    model: torch.nn.Module,
    *,
    lr: float,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    momentum: float = 0.9,
    rho: float = 0.1,
    interval: int = 10,
    variant: int = 4,
    muon_momentum: float = 0.95,
) -> Tuple[torch.optim.Optimizer, DecayGroups, Dict[str, Any]]:
    """Create an optimizer instance.

    Notes
    -----
    - We keep weight_decay=0 in optimizers and apply decoupled WD externally.
    - This function returns (optimizer, decay_groups, extra_dict).
      `extra_dict` is used for things like Sophia's per-step batch-size.
    """

    decay_groups = split_decay_params(model)

    # param groups for optimizer (no internal decay)
    param_groups = [
        {"params": decay_groups.decay, "weight_decay": 0.0},
        {"params": decay_groups.no_decay, "weight_decay": 0.0},
    ]

    extra: Dict[str, Any] = {}

    name_l = name.lower()

    if name_l == "adamw":
        opt = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=0.0)
        return opt, decay_groups, extra

    if name_l == "sgd":
        # vanilla SGD + momentum (no nesterov by default)
        opt = torch.optim.SGD(param_groups, lr=lr, momentum=momentum, dampening=0.0, weight_decay=0.0, nesterov=False)
        return opt, decay_groups, extra

    if name_l == "lion":
        # PyPI: lion-pytorch
        Lion = _import_symbol("lion_pytorch", "Lion")
        opt = Lion(param_groups, lr=lr, betas=betas, weight_decay=0.0)
        return opt, decay_groups, extra

    if name_l == "lamb":
        # Most common: pytorch-lamb repo
        Lamb = None
        try:
            Lamb = _import_symbol("pytorch_lamb", "Lamb")
        except Exception:
            try:
                Lamb = _import_symbol("pytorch_lamb.lamb", "Lamb")
            except Exception as e:
                raise ImportError(
                    "Could not import Lamb. Install dependency from requirements.txt"
                ) from e
        opt = Lamb(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=0.0)
        return opt, decay_groups, extra

    if name_l == "sophiag":
        # local file sophia.py (from Sophia repo)
        SophiaG = _import_symbol("sophia", "SophiaG")
        opt = SophiaG(param_groups, lr=lr, betas=betas, rho=rho, weight_decay=0.0, eps=eps, interval=interval, variant=variant)
        # Sophia step uses `bs` (batch size) for scaling hessian estimate
        extra.update({"sophia_interval": interval})
        return opt, decay_groups, extra

    if name_l == "muon":
        # Official Muon repo is expected to provide MuonWithAuxAdam
        MuonWithAuxAdam = None
        last_err = None
        for mod, sym in [
            ("muon", "MuonWithAuxAdam"),
            ("muon.muon", "MuonWithAuxAdam"),
        ]:
            try:
                MuonWithAuxAdam = _import_symbol(mod, sym)
                break
            except Exception as e:
                last_err = e
                continue
        if MuonWithAuxAdam is None:
            raise ImportError(
                "Could not import MuonWithAuxAdam from Muon. Install dependency from requirements.txt"
            ) from last_err

        # We pass param_groups directly; Muon implementation decides how to treat matrices.
        opt = MuonWithAuxAdam(
            param_groups,
            lr=lr,
            momentum=muon_momentum,
            weight_decay=0.0,
            betas=betas,
            eps=eps,
        )
        return opt, decay_groups, extra

    raise ValueError(f"Unknown optimizer name: {name}")
