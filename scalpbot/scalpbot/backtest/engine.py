"""Motor de backtest barra a barra con costes, latencia y limites de riesgo.

Supuestos explicitos (todos conservadores):
  * La senal se calcula al CIERRE de la barra t y se ejecuta a la APERTURA de
    t+1. Nunca se opera al precio que genero la senal.
  * El deslizamiento se aplica siempre en contra.
  * Si en la misma vela se tocan take-profit y stop-loss, se asume el stop.
  * Las comisiones se cobran en ambos lados sobre el nocional.
  * Una sola posicion abierta a la vez (scalping direccional simple).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..strategy.risk import RiskManager
from ..utils import get_logger

log = get_logger("backtest")

EXIT_REASONS = ("take_profit", "stop_loss", "timeout", "flip", "end_of_data")


@dataclass
class Trade:
    entry_ts: Any
    exit_ts: Any
    side: int
    entry_px: float
    exit_px: float
    qty: float
    notional: float
    fees: float
    pnl: float
    ret_bps: float
    bars_held: int
    reason: str
    edge_bps: float
    p_dir: float


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    stats: dict[str, Any] = field(default_factory=dict)


def run_backtest(cfg: Config, bars: pd.DataFrame, signals: pd.DataFrame) -> BacktestResult:
    """Ejecuta el backtest.

    bars: OHLC indexado por ts (debe cubrir el indice de `signals`).
    signals: salida de `strategy.policy.decide_batch` (side, size_frac, tp/sl...).
    """
    sig = signals.reindex(bars.index)
    has_signal = sig["side"].notna()
    if not has_signal.any():
        raise ValueError("no hay senales solapadas con las barras")

    open_ = bars["open"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    index = bars.index

    side_arr = sig["side"].fillna(0).to_numpy(dtype=float)
    size_arr = sig["size_frac"].fillna(0.0).to_numpy(dtype=float)
    tp_arr = sig["tp_bps"].fillna(0.0).to_numpy(dtype=float)
    sl_arr = sig["sl_bps"].fillna(0.0).to_numpy(dtype=float)
    edge_arr = sig["edge_bps"].fillna(0.0).to_numpy(dtype=float)
    pup_arr = sig["p_up"].fillna(0.5).to_numpy(dtype=float)
    pdn_arr = sig["p_dn"].fillna(0.5).to_numpy(dtype=float)

    costs, scfg = cfg.costs, cfg.strategy
    slip = costs.slippage_bps * 1e-4
    entry_fee_rate = (costs.maker_fee_bps if costs.entry_is_maker else costs.taker_fee_bps) * 1e-4
    exit_fee_rate = (costs.maker_fee_bps if costs.exit_is_maker else costs.taker_fee_bps) * 1e-4
    horizon = cfg.labels.horizon

    risk = RiskManager(cfg.risk)
    n = len(bars)
    equity_curve = np.empty(n, dtype=float)

    trades: list[Trade] = []
    pos_side = 0
    entry_px = tp_px = sl_px = 0.0
    qty = notional = entry_fee = 0.0
    entry_i = 0
    entry_edge = entry_pdir = 0.0
    cooldown_until = -1
    blocked_reasons: dict[str, int] = {}

    def close_position(i: int, px: float, reason: str) -> None:
        nonlocal pos_side, qty, notional, entry_fee
        exit_notional = px * qty
        fee = exit_notional * exit_fee_rate
        pnl = pos_side * (px - entry_px) * qty - entry_fee - fee
        risk.register_close(pnl)
        trades.append(Trade(
            entry_ts=index[entry_i], exit_ts=index[i], side=pos_side,
            entry_px=entry_px, exit_px=px, qty=qty, notional=notional,
            fees=entry_fee + fee, pnl=pnl,
            ret_bps=(pnl / notional * 1e4) if notional > 0 else 0.0,
            bars_held=i - entry_i, reason=reason,
            edge_bps=entry_edge, p_dir=entry_pdir,
        ))
        pos_side = 0
        qty = notional = entry_fee = 0.0

    for i in range(n):
        risk.roll_day(index[i])

        # ---------- 1. Gestion de la posicion abierta (dentro de la barra i)
        if pos_side != 0:
            hit_tp = high[i] >= tp_px if pos_side > 0 else low[i] <= tp_px
            hit_sl = low[i] <= sl_px if pos_side > 0 else high[i] >= sl_px

            if hit_tp and hit_sl:
                close_position(i, sl_px * (1 - pos_side * slip), "stop_loss")
            elif hit_sl:
                close_position(i, sl_px * (1 - pos_side * slip), "stop_loss")
            elif hit_tp:
                close_position(i, tp_px * (1 - pos_side * slip), "take_profit")
            elif i - entry_i >= horizon:
                close_position(i, close[i] * (1 - pos_side * slip), "timeout")
            elif scfg.exit_on_flip and side_arr[i] != 0 and side_arr[i] != pos_side:
                close_position(i, close[i] * (1 - pos_side * slip), "flip")

            if pos_side == 0:
                cooldown_until = i + scfg.cooldown_bars

        # ---------- 2. Apertura: senal en la barra i, ejecucion en i+1 open
        if pos_side == 0 and i + 1 < n and i > cooldown_until:
            want = int(side_arr[i])
            if want != 0 and size_arr[i] > 0:
                ok, reason = risk.can_open()
                if ok:
                    px = open_[i + 1] * (1 + want * slip)
                    notional = risk.notional(size_arr[i])
                    if notional > 0 and px > 0:
                        qty = notional / px
                        entry_px = px
                        entry_fee = notional * entry_fee_rate
                        pos_side = want
                        entry_i = i + 1
                        entry_edge = edge_arr[i]
                        entry_pdir = pup_arr[i] if want > 0 else pdn_arr[i]
                        tp_px = entry_px * (1 + want * tp_arr[i] * 1e-4)
                        sl_px = entry_px * (1 - want * sl_arr[i] * 1e-4)
                        risk.register_open()
                else:
                    blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1

        # ---------- 3. Marca a mercado (solo si la posicion ya esta viva)
        if pos_side != 0 and i >= entry_i:
            unrealized = pos_side * (close[i] - entry_px) * qty - entry_fee
            equity_curve[i] = risk.state.equity + unrealized
        else:
            equity_curve[i] = risk.state.equity

    if pos_side != 0:
        close_position(n - 1, close[-1] * (1 - pos_side * slip), "end_of_data")
        equity_curve[-1] = risk.state.equity

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    equity = pd.Series(equity_curve, index=index, name="equity")

    if blocked_reasons:
        log.info("aperturas bloqueadas por riesgo: %s", blocked_reasons)
    log.info("backtest: %d operaciones | equity %.2f -> %.2f",
             len(trades_df), cfg.risk.initial_equity, equity.iloc[-1])

    return BacktestResult(trades=trades_df, equity=equity,
                          stats={"blocked": blocked_reasons})
