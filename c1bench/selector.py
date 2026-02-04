from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch

from .dc_diag import estimate_dc
from .utils import JsonlWriter


@dataclass
class SelectorConfig:
    candidates: Sequence[str]
    sel_every: int = 100
    sel_min_ep: int = 100
    sel_patience: int = 100
    dc_ema_rho: float = 0.2
    dc_delta: float = 0.02
    dc_min: float = 0.0
    dc_probes: int = 4


class Selector:
    """DC-based optimizer selector with on-trajectory state tracking.

    - Maintains a separate optimizer instance per candidate.
    - Uses DC_hat to score candidates at fixed intervals.
    - Switches with EMA smoothing + hysteresis.
    - Updates inactive optimizers' state on-trajectory via lr=0 steps.
    """

    def __init__(
        self,
        cfg: SelectorConfig,
        *,
        optims: Dict[str, torch.optim.Optimizer],
        log_writer: Optional[JsonlWriter] = None,
    ) -> None:
        self.cfg = cfg
        self.optimizers: Dict[str, torch.optim.Optimizer] = dict(optims)
        self.log_writer = log_writer

        if not cfg.candidates:
            raise ValueError("SelectorConfig.candidates must be non-empty")

        self.active_name: str = str(list(cfg.candidates)[0])
        self.last_switch_step: int = 0
        self.dc_ema: Dict[str, float] = {str(c): 0.0 for c in cfg.candidates}

    @property
    def active_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizers[self.active_name]

    def state_dict(self) -> Dict[str, object]:
        return {
            "active_name": self.active_name,
            "last_switch_step": self.last_switch_step,
            "dc_ema": dict(self.dc_ema),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.active_name = str(state.get("active_name", self.active_name))
        self.last_switch_step = int(state.get("last_switch_step", self.last_switch_step))
        dc_ema = state.get("dc_ema", None)
        if isinstance(dc_ema, dict):
            self.dc_ema = {str(k): float(v) for k, v in dc_ema.items()}

    @torch.no_grad()
    def update_inactive_states_on_trajectory(self, *, bs_tokens: Optional[int] = None) -> None:
        """Update internal states of all inactive optimizers using current grads.

        Implementation: temporarily set lr=0 and call .step() so params don't change,
        but moment buffers / EMA stats are updated.
        """
        for name, opt in self.optimizers.items():
            if name == self.active_name:
                continue
            old_lrs = [g.get("lr", 0.0) for g in opt.param_groups]
            for g in opt.param_groups:
                g["lr"] = 0.0
                if bs_tokens is not None:
                    g["bs"] = float(bs_tokens)
            try:
                opt.step()
            finally:
                for g, lr in zip(opt.param_groups, old_lrs):
                    g["lr"] = lr

    @torch.no_grad()
    def maybe_select(
        self,
        *,
        step: int,
        model: torch.nn.Module,
        get_batch,
        bs_tokens: float,
    ) -> Optional[str]:
        """Score candidates and maybe switch active optimizer.

        Returns:
            The new active optimizer name if a switch happened, otherwise None.
        """
        if step < self.cfg.sel_min_ep:
            return None
        if (step % max(1, self.cfg.sel_every)) != 0:
            return None
        if (step - self.last_switch_step) < self.cfg.sel_patience:
            return None

        device = "cuda" if next(model.parameters()).is_cuda else "cpu"

        dc_hat: Dict[str, float] = {}
        P_hat: Dict[str, float] = {}
        G_hat: Dict[str, float] = {}
        E_hat: Dict[str, float] = {}

        for name in self.cfg.candidates:
            opt = self.optimizers[str(name)]
            res = estimate_dc(
                model=model,
                get_batch=get_batch,
                optimizer=opt,
                optimizer_name=str(name),
                device=device,
                probes=self.cfg.dc_probes,
                bs_tokens=bs_tokens,
                fp32=True,
            )
            dc_hat[str(name)] = float(res["dc_hat"])
            P_hat[str(name)] = float(res["P_hat"])
            G_hat[str(name)] = float(res["G_hat"])
            E_hat[str(name)] = float(res["E_hat"])

        # EMA smoothing
        for name in self.cfg.candidates:
            n = str(name)
            self.dc_ema[n] = (1.0 - self.cfg.dc_ema_rho) * self.dc_ema[n] + self.cfg.dc_ema_rho * dc_hat[n]

        best = max(self.cfg.candidates, key=lambda n: self.dc_ema[str(n)])
        best = str(best)
        cur = self.active_name

        switched_to: Optional[str] = None
        if best != cur:
            if self.dc_ema[best] >= max(self.cfg.dc_min, (1.0 + self.cfg.dc_delta) * self.dc_ema[cur]):
                self.active_name = best
                self.last_switch_step = int(step)
                switched_to = best

        if self.log_writer is not None:
            self.log_writer.write(
                {
                    "it": int(step),
                    "active": cur,
                    "best": best,
                    "switched_to": switched_to,
                    "dc_hat": dc_hat,
                    "dc_ema": dict(self.dc_ema),
                    "P_hat": P_hat,
                    "G_hat": G_hat,
                    "E_hat": E_hat,
                }
            )

        return switched_to
        