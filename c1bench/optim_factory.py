from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

import importlib
import inspect
import torch


OptimizerName = Literal["sgd", "adamw", "lion", "sophiag", "lamb", "muon"]


@dataclass
class DecayGroups:
    decay: List[torch.nn.Parameter]
    no_decay: List[torch.nn.Parameter]


def split_decay_params(model: torch.nn.Module) -> DecayGroups:
    """nanoGPT-style decay split (we keep WD external in train.py)."""
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias"):
            no_decay.append(p)
        elif name.endswith(".weight") and ("ln" in name.lower() or "norm" in name.lower()):
            no_decay.append(p)
        elif "bias" in name.lower() and p.ndim == 1:
            no_decay.append(p)
        else:
            decay.append(p)

    return DecayGroups(decay=decay, no_decay=no_decay)


def _import_symbol(module: str, symbol: str):
    mod = importlib.import_module(module)
    return getattr(mod, symbol)


def _find_lamb_class():
    candidates = [
        ("pytorch_lamb", "Lamb"),
        ("pytorch_lamb.lamb", "Lamb"),
        ("torch_optimizer", "Lamb"),
        ("torch_optimizer.lamb", "Lamb"),
    ]
    last = None
    for mod, sym in candidates:
        try:
            return _import_symbol(mod, sym)
        except Exception as e:
            last = e
    raise ImportError(
        "Could not import LAMB optimizer.\n"
        "Install one of:\n"
        "  python -m pip install -U git+https://github.com/cybertronai/pytorch-lamb.git\n"
        "  python -m pip install -U torch-optimizer\n"
    ) from last


def _muon_param_partition_for_gpt(model: torch.nn.Module):
    """Partition params for Muon like official guidance:
    - Muon only for hidden 2D weights inside transformer blocks
    - Adam (aux) for embeddings/head and all 1D params
    """
    muon_params: List[torch.nn.Parameter] = []
    aux_params: List[torch.nn.Parameter] = []

    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)

        # GPT in this repo names blocks as "transformer.h.<i>...."
        if name.startswith("transformer.h.") and p.ndim >= 2:
            muon_params.append(p)
        else:
            aux_params.append(p)

    return muon_params, aux_params


def make_optimizer(
    name: OptimizerName,
    model: torch.nn.Module,
    *,
    lr: float,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    momentum: float = 0.9,
    rho: float = 0.1,
    muon_momentum: float = 0.95,
) -> torch.optim.Optimizer:
    """Create optimizer. Weight decay is kept at 0 inside optimizers."""
    dg = split_decay_params(model)
    param_groups_std = [
        {"params": dg.decay, "weight_decay": 0.0},
        {"params": dg.no_decay, "weight_decay": 0.0},
    ]

    name_l = name.lower()

    if name_l == "adamw":
        return torch.optim.AdamW(param_groups_std, lr=lr, betas=betas, eps=eps, weight_decay=0.0)

    if name_l == "sgd":
        return torch.optim.SGD(param_groups_std, lr=lr, momentum=momentum, dampening=0.0, weight_decay=0.0, nesterov=False)

    if name_l == "lion":
        Lion = _import_symbol("lion_pytorch", "Lion")
        return Lion(param_groups_std, lr=lr, betas=betas, weight_decay=0.0)

    if name_l == "lamb":
        Lamb = _find_lamb_class()
        try:
            return Lamb(param_groups_std, lr=lr, betas=betas, eps=eps, weight_decay=0.0)
        except TypeError:
            # filter kwargs by signature for odd implementations
            sig = inspect.signature(Lamb.__init__)
            names = set(sig.parameters.keys()) - {"self"}
            kwargs: Dict[str, Any] = {}
            if "lr" in names:
                kwargs["lr"] = lr
            if "betas" in names:
                kwargs["betas"] = betas
            if "eps" in names:
                kwargs["eps"] = eps
            if "weight_decay" in names:
                kwargs["weight_decay"] = 0.0
            return Lamb(param_groups_std, **kwargs)

    if name_l == "sophiag":
        SophiaG = _import_symbol("sophia", "SophiaG")
        return SophiaG(param_groups_std, lr=lr, betas=betas, rho=rho, weight_decay=0.0)

    if name_l == "muon":
        # Official Muon: use SingleDeviceMuonWithAuxAdam when not using torch.distributed
        try:
            MuonCls = _import_symbol("muon", "SingleDeviceMuonWithAuxAdam")
        except Exception:
            MuonCls = _import_symbol("muon", "MuonWithAuxAdam")

        muon_params, aux_params = _muon_param_partition_for_gpt(model)

        # IMPORTANT: group keys must match EXACTLY what muon.py asserts.
        muon_group = {
            "params": muon_params,
            "lr": lr,
            "momentum": muon_momentum,
            "weight_decay": 0.0,
            "use_muon": True,
        }
        aux_group = {
            "params": aux_params,
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": 0.0,
            "use_muon": False,
        }
        return MuonCls([aux_group, muon_group])

    raise ValueError(f"Unknown optimizer name: {name}")
