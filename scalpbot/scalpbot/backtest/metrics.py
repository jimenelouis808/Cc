"""Metricas de rendimiento y test de significancia."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import bars_per_year


def compute_stats(cfg: Config, equity: pd.Series, trades: pd.DataFrame) -> dict[str, Any]:
    """Resumen honesto: rendimiento, riesgo, costes y significancia estadistica."""
    e0 = float(cfg.risk.initial_equity)
    eq = equity.astype(float)
    rets = eq.pct_change().fillna(0.0)
    bpy = bars_per_year(cfg.data.interval)

    total_return = eq.iloc[-1] / e0 - 1.0
    n_bars = len(eq)
    years = n_bars / bpy
    cagr = (eq.iloc[-1] / e0) ** (1 / years) - 1.0 if years > 0 and eq.iloc[-1] > 0 else float("nan")

    sd = rets.std(ddof=0)
    sharpe = float(rets.mean() / sd * np.sqrt(bpy)) if sd > 1e-12 else float("nan")
    downside = rets[rets < 0].std(ddof=0)
    sortino = float(rets.mean() / downside * np.sqrt(bpy)) if downside > 1e-12 else float("nan")

    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < -1e-9 and np.isfinite(cagr) else float("nan")

    stats: dict[str, Any] = {
        "bars": n_bars,
        "days": round(years * 365, 1),
        "initial_equity": e0,
        "final_equity": float(eq.iloc[-1]),
        "total_return_pct": 100 * float(total_return),
        "cagr_pct": 100 * float(cagr) if np.isfinite(cagr) else float("nan"),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": 100 * max_dd,
        "calmar": calmar,
    }

    if trades is None or trades.empty:
        stats.update({"n_trades": 0, "note": "el modelo no genero ninguna operacion"})
        return stats

    pnl = trades["pnl"].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()

    stats.update({
        "n_trades": int(len(trades)),
        "trades_per_day": round(len(trades) / max(years * 365, 1e-9), 1),
        "win_rate_pct": 100 * float((pnl > 0).mean()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 1e-12 else float("inf"),
        "avg_trade_bps": float(trades["ret_bps"].mean()),
        "median_trade_bps": float(trades["ret_bps"].median()),
        "avg_win_usd": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_usd": float(losses.mean()) if len(losses) else 0.0,
        "avg_bars_held": float(trades["bars_held"].mean()),
        "total_fees_usd": float(trades["fees"].sum()),
        "fees_over_gross_pct": 100 * float(
            trades["fees"].sum() / max(abs(pnl).sum() + trades["fees"].sum(), 1e-9)),
        "total_volume_usd": float(trades["notional"].sum()),
        "exit_mix": {k: int(v) for k, v in trades["reason"].value_counts().items()},
    })

    # t-stat del retorno medio por operacion: cuantifica si el edge es ruido.
    r = trades["ret_bps"].astype(float)
    if len(r) > 2 and r.std(ddof=1) > 1e-12:
        t_stat = float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))
        stats["t_stat"] = t_stat
        stats["edge_significant"] = bool(abs(t_stat) > 2.0)
    else:
        stats["t_stat"] = float("nan")
        stats["edge_significant"] = False

    # Punto de equilibrio: cuanto edge bruto hace falta solo para pagar costes.
    stats["round_trip_cost_bps"] = cfg.costs.round_trip_bps()
    stats["gross_edge_bps"] = float(r.mean() + cfg.costs.round_trip_bps())

    return stats


def cost_sensitivity(cfg: Config, trades: pd.DataFrame,
                     extra_bps: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0)) -> pd.DataFrame:
    """Cuanto sobrevive la estrategia si los costes reales son peores.

    Es la prueba mas informativa de un backtest de scalping: si el PnL se
    evapora con 2 bps extra, no hay estrategia, hay una estimacion de costes
    optimista.
    """
    if trades is None or trades.empty:
        return pd.DataFrame()
    rows = []
    for extra in extra_bps:
        adj = trades["ret_bps"].astype(float) - extra
        pnl = adj / 1e4 * trades["notional"].astype(float)
        rows.append({
            "extra_cost_bps": extra,
            "avg_trade_bps": float(adj.mean()),
            "total_pnl_usd": float(pnl.sum()),
            "win_rate_pct": 100 * float((adj > 0).mean()),
            "profitable": bool(pnl.sum() > 0),
        })
    return pd.DataFrame(rows)


def format_stats(stats: dict[str, Any]) -> str:
    """Render legible en terminal."""
    order = [
        ("bars", "Barras", "{:,.0f}"), ("days", "Dias", "{:.1f}"),
        ("n_trades", "Operaciones", "{:,.0f}"),
        ("trades_per_day", "Operaciones/dia", "{:.1f}"),
        ("final_equity", "Equity final", "${:,.2f}"),
        ("total_return_pct", "Retorno total", "{:+.2f}%"),
        ("cagr_pct", "CAGR", "{:+.1f}%"),
        ("sharpe", "Sharpe", "{:.2f}"), ("sortino", "Sortino", "{:.2f}"),
        ("max_drawdown_pct", "Max drawdown", "{:.2f}%"),
        ("calmar", "Calmar", "{:.2f}"),
        ("win_rate_pct", "Tasa de acierto", "{:.1f}%"),
        ("profit_factor", "Profit factor", "{:.2f}"),
        ("avg_trade_bps", "Media por trade", "{:+.2f} bps"),
        ("gross_edge_bps", "Edge bruto", "{:+.2f} bps"),
        ("round_trip_cost_bps", "Coste ida y vuelta", "{:.2f} bps"),
        ("avg_bars_held", "Barras en posicion", "{:.1f}"),
        ("total_fees_usd", "Comisiones pagadas", "${:,.2f}"),
        ("t_stat", "t-stat del edge", "{:.2f}"),
    ]
    lines = []
    for key, label, fmt in order:
        if key not in stats:
            continue
        v = stats[key]
        try:
            lines.append(f"  {label:<24} {fmt.format(v)}")
        except (ValueError, TypeError):
            lines.append(f"  {label:<24} {v}")
    if "exit_mix" in stats:
        lines.append(f"  {'Motivo de salida':<24} {stats['exit_mix']}")
    if "edge_significant" in stats:
        verdict = "SI (|t| > 2)" if stats["edge_significant"] else "NO (indistinguible de ruido)"
        lines.append(f"  {'Edge significativo':<24} {verdict}")
    if "note" in stats:
        lines.append(f"  {'Nota':<24} {stats['note']}")
    return "\n".join(lines)
