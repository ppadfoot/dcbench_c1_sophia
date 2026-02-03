from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .dc_diag import estimate_dc
from .utils import JsonlWriter


@dataclass
class SelectorConfig:
    candidates: Sequence[str]
    sel_every: int = 100
    sel_min_ep: int = 100
    sel_patience: int = 100
    dc_ema_rho: float = 0.1
    delta: float = 0.05
    dc_min: float = 0.0
    dc_probes: int = 4
    log_interval: int = 1


class DCBenchSelector:
    """A practical DC-based optimizer selector.

    Key practical choice:
    - keep a *separate optimizer instance per candidate*.
    - update the *inactive* optimizers' internal states on-trajectory by calling
      their `.step()` with lr=0 (so weights don't change but momentums do).

    This avoids the classic failure mode where switching resets moments.
    """

    def __init__(
        self,
        cfg: SelectorConfig,
        optimizers: Dict[str, torch.optim.Optimizer],
        log_path: str,
    ) -> None:
        self.cfg = cfg
        self.optimizers = optimizers
        self.sel_writer = JsonlWriter(log_path)

        self.active_name = list(cfg.candidates)[0]
        self.last_switch_step = 0

        self.dc_ema: Dict[str, float] = {c: 0.0 for c in cfg.candidates}

    def close(self) -> None:
        self.sel_writer.close()

    @property
    def active_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizers[self.active_name]

    @torch.no_grad()
    def update_inactive_states_on_trajectory(self, *, bs_tokens: Optional[int] = None) -> None:
        """Update the optimizers' internal states using the current gradients.

        We temporarily set lr=0 so `.step()` does not change parameters.
        """
        for name, opt in self.optimizers.items():
            if name == self.active_name:
                continue
            # set lr=0 for all param groups
            old_lrs = [g.get("lr", 0.0) for g in opt.param_groups]
            for g in opt.param_groups:
                g["lr"] = 0.0
            try:
                if bs_tokens is not None:
                    for g in opt.param_groups:
                        g["bs"] = bs_tokens
                opt.step()
            finally:
                for g, lr in zip(opt.param_groups, old_lrs):
                    g["lr"] = lr

    def maybe_select(
        self,
        *,
        step: int,
        model: torch.nn.Module,
        get_batch,
        device: str,
        bs_tokens: float,
        verbose: bool = False,
    ) -> Optional[str]:
        """Run DC probes and maybe switch. Returns the new active name if switched."""
        if step < self.cfg.sel_min_ep:
            return None
        if (step % self.cfg.sel_every) != 0:
            return None
        if (step - self.last_switch_step) < self.cfg.sel_patience:
            return None

        # Estimate DC for each candidate.
        dc_hat: Dict[str, float] = {}
        p_hat: Dict[str, float] = {}
        g_hat: Dict[str, float] = {}
        e_hat: Dict[str, float] = {}

        for name in self.cfg.candidates:
            opt = self.optimizers[name]
            res = estimate_dc(
                model=model,
                get_batch=get_batch,
                optimizer=opt,
                optimizer_name=name,
                device=device,
                probes=self.cfg.dc_probes,
                bs_tokens=bs_tokens,
                fp32=True,
            )
            dc_hat[name] = float(res["dc_hat"])
            p_hat[name] = float(res["P_hat"])
            g_hat[name] = float(res["G_hat"])
            e_hat[name] = float(res["E_hat"])

        # EMA smoothing
        for name in self.cfg.candidates:
            self.dc_ema[name] = (1.0 - self.cfg.dc_ema_rho) * self.dc_ema[name] + self.cfg.dc_ema_rho * dc_hat[name]

        # pick best
        best = max(self.cfg.candidates, key=lambda n: self.dc_ema[n])
        cur = self.active_name

        switched = None
        if best != cur:
            if self.dc_ema[best] >= max(self.cfg.dc_min, (1.0 + self.cfg.delta) * self.dc_ema[cur]):
                switched = best
                self.active_name = best
                self.last_switch_step = step

        # log
        row = {
            "step": step,
            "active": cur,
            "best": best,
            "switched_to": switched,
            "dc_hat": dc_hat,
            "dc_ema": dict(self.dc_ema),
            "P_hat": p_hat,
            "G_hat": g_hat,
            "E_hat": e_hat,
        }
        self.sel_writer.write(row)

        if verbose:
            msg = f"[selector] step={step} active={cur} best={best}"
            if switched is not None:
                msg += f" SWITCH->{switched}"
            print(msg)

        return switched
