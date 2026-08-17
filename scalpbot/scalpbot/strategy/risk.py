"""Cortacircuitos de riesgo. Se aplican igual en backtest y en vivo."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import RiskConfig


@dataclass
class RiskState:
    """Estado mutable del gestor de riesgo dentro de una sesion de trading."""
    equity: float
    day: object = None
    day_start_equity: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    halted_reason: str | None = None
    halted_days: set = field(default_factory=set)


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.state = RiskState(equity=cfg.initial_equity,
                               day_start_equity=cfg.initial_equity)

    def roll_day(self, ts: pd.Timestamp) -> None:
        """Reinicia contadores al cambiar de dia UTC.

        La racha de perdidas DEBE reiniciarse aqui. Si no, al alcanzar el limite
        el bot no puede abrir, y como no abre nunca cierra en ganancia: el
        contador jamas baja y el bot queda bloqueado para siempre.
        """
        day = ts.date()
        if self.state.day != day:
            self.state.day = day
            self.state.day_start_equity = self.state.equity
            self.state.trades_today = 0
            self.state.consecutive_losses = 0
            self.state.halted_reason = None

    def can_open(self) -> tuple[bool, str]:
        s, c = self.state, self.cfg
        if s.equity <= 0:
            return False, "equity agotado"
        if s.trades_today >= c.max_trades_per_day:
            return False, "limite diario de operaciones"
        if s.consecutive_losses >= c.max_consecutive_losses:
            return False, "racha de perdidas"
        dd = (s.day_start_equity - s.equity) / max(s.day_start_equity, 1e-9)
        if dd >= c.max_daily_loss_pct / 100.0:
            return False, "stop diario de perdidas"
        return True, ""

    def register_open(self) -> None:
        self.state.trades_today += 1

    def register_close(self, pnl: float) -> None:
        self.state.equity += pnl
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

    def notional(self, size_frac: float) -> float:
        """Nocional de la posicion, aplicando apalancamiento configurado."""
        return max(0.0, self.state.equity * size_frac * self.cfg.leverage)
