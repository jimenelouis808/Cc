"""Traduce probabilidades del modelo en decisiones de trading.

El paso critico no es "predecir si sube": es comprobar que el valor esperado
NETO DE COSTES es positivo. En micro-scalping los costes suelen ser mayores que
la senal, y ahi es donde mueren la mayoria de los bots.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import CostConfig, StrategyConfig


@dataclass
class Decision:
    side: int           # -1 short, 0 fuera, +1 long
    size_frac: float    # fraccion del equity a arriesgar (0..max_position_pct)
    edge_bps: float     # valor esperado neto en bps
    p_up: float
    p_dn: float


def expected_edge(p_up: np.ndarray, p_dn: np.ndarray, tp_bps: np.ndarray,
                  sl_bps: np.ndarray, cost_bps: float) -> tuple[np.ndarray, np.ndarray]:
    """EV en bps de ir largo y de ir corto, ya descontados los costes.

    Para un largo: gana tp si la barrera superior se toca antes, pierde sl si
    se toca la inferior, y ~0 si expira por tiempo. Los costes se pagan siempre.
    """
    ev_long = p_up * tp_bps - p_dn * sl_bps - cost_bps
    ev_short = p_dn * tp_bps - p_up * sl_bps - cost_bps
    return ev_long, ev_short


def kelly_size(p_win: float, win_bps: float, loss_bps: float, fraction: float,
               cap: float) -> float:
    """Kelly fraccional. Devuelve la fraccion de equity a comprometer."""
    if loss_bps <= 0 or win_bps <= 0:
        return 0.0
    b = win_bps / loss_bps
    q = 1.0 - p_win
    f = (b * p_win - q) / b
    if f <= 0:
        return 0.0
    return float(np.clip(f * fraction, 0.0, cap))


def decide_batch(proba: pd.DataFrame, tp_bps: pd.Series, sl_bps: pd.Series,
                 scfg: StrategyConfig, ccfg: CostConfig) -> pd.DataFrame:
    """Aplica la politica a un lote de predicciones (vectorizado, para backtest).

    proba: columnas p_dn/p_flat/p_up.
    tp_bps / sl_bps: tamano de las barreras de cada barra en bps.
    """
    p_up = proba["p_up"].to_numpy()
    p_dn = proba["p_dn"].to_numpy()
    tp = tp_bps.reindex(proba.index).to_numpy()
    sl = sl_bps.reindex(proba.index).to_numpy()
    cost = ccfg.round_trip_bps()

    ev_long, ev_short = expected_edge(p_up, p_dn, tp, sl, cost)

    side = np.zeros(len(proba), dtype=np.int8)
    long_ok = (ev_long >= scfg.min_edge_bps) & (p_up >= scfg.min_prob) & (ev_long >= ev_short)
    short_ok = (ev_short >= scfg.min_edge_bps) & (p_dn >= scfg.min_prob) & (ev_short > ev_long)
    side[long_ok] = 1
    if scfg.allow_short:
        side[short_ok] = -1

    edge = np.where(side > 0, ev_long, np.where(side < 0, ev_short, 0.0))

    # Kelly sobre la probabilidad condicional de acertar la direccion.
    p_dir = np.where(side > 0, p_up, p_dn)
    p_adv = np.where(side > 0, p_dn, p_up)
    denom = p_dir + p_adv
    p_win = np.divide(p_dir, denom, out=np.full_like(p_dir, 0.5), where=denom > 1e-9)
    b = np.divide(tp, sl, out=np.ones_like(tp), where=sl > 1e-9)
    f = np.divide(b * p_win - (1 - p_win), b, out=np.zeros_like(b), where=b > 1e-9)
    size = np.clip(f * scfg.kelly_fraction, 0.0, scfg.max_position_pct)
    size = np.where(side != 0, size, 0.0)

    return pd.DataFrame({
        "side": side, "size_frac": size, "edge_bps": edge,
        "tp_bps": tp, "sl_bps": sl, "p_up": p_up, "p_dn": p_dn,
    }, index=proba.index)


def decide_one(p_dn: float, p_flat: float, p_up: float, tp_bps: float, sl_bps: float,
               scfg: StrategyConfig, ccfg: CostConfig) -> Decision:
    """Version escalar para el bucle en vivo."""
    frame = decide_batch(
        pd.DataFrame({"p_dn": [p_dn], "p_flat": [p_flat], "p_up": [p_up]}, index=[0]),
        pd.Series([tp_bps], index=[0]), pd.Series([sl_bps], index=[0]), scfg, ccfg)
    r = frame.iloc[0]
    return Decision(side=int(r["side"]), size_frac=float(r["size_frac"]),
                    edge_bps=float(r["edge_bps"]), p_up=p_up, p_dn=p_dn)
