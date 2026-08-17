"""Utilidades compartidas: logging, tiempo y helpers numericos."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

INTERVAL_MS = {
    "1s": 1_000, "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}

BPS = 1e-4


def get_logger(name: str = "scalpbot", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                                         datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def bars_per_year(interval: str) -> float:
    ms = INTERVAL_MS.get(interval)
    if ms is None:
        raise ValueError(f"intervalo no soportado: {interval}")
    return 365.0 * 24 * 3600 * 1000 / ms


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_div(a, b, fill: float = 0.0):
    """Division elemento a elemento que devuelve `fill` donde el denominador es ~0."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full(np.broadcast(a_arr, b_arr).shape, float(fill))
    mask = np.abs(b_arr) > 1e-12
    np.divide(a_arr, b_arr, out=out, where=mask)
    return out


def zscore(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Z-score rodante causal (solo usa pasado)."""
    mp = min_periods or max(5, window // 4)
    mean = s.rolling(window, min_periods=mp).mean()
    std = s.rolling(window, min_periods=mp).std(ddof=0)
    return ((s - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def rank_pct(s: pd.Series, window: int) -> pd.Series:
    """Percentil rodante del ultimo valor dentro de su ventana."""
    mp = max(5, window // 4)
    return s.rolling(window, min_periods=mp).rank(pct=True)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Sustituye infinitos por NaN sin rellenar hacia adelante (evita fugas sutiles)."""
    return df.replace([np.inf, -np.inf], np.nan)


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Escribe parquet si hay engine disponible; si no, CSV."""
    p = Path(path)
    ensure_dir(p.parent)
    try:
        df.to_parquet(p.with_suffix(".parquet"))
        return p.with_suffix(".parquet")
    except Exception:
        df.to_csv(p.with_suffix(".csv"))
        return p.with_suffix(".csv")


def read_table(path: str | Path) -> pd.DataFrame:
    """Lee la tabla escrita por `write_table`, probando parquet y luego CSV."""
    p = Path(path)
    for cand in (p, p.with_suffix(".parquet"), p.with_suffix(".csv")):
        if cand.exists():
            if cand.suffix == ".parquet":
                return pd.read_parquet(cand)
            return pd.read_csv(cand, index_col=0, parse_dates=True)
    raise FileNotFoundError(f"no existe tabla en {p} (.parquet/.csv)")
