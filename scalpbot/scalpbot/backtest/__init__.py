"""Backtest event-driven y metricas."""
from .engine import BacktestResult, run_backtest
from .metrics import compute_stats, cost_sensitivity, format_stats

__all__ = ["BacktestResult", "run_backtest", "compute_stats", "cost_sensitivity",
           "format_stats"]
