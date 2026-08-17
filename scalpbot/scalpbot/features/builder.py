"""Ensambla la matriz de features a partir de las klines crudas."""
from __future__ import annotations

import pandas as pd

from ..utils import clean_frame, get_logger
from . import context, microstructure, technical

log = get_logger("features")

# Columnas del kline crudo que se conservan para backtest/etiquetado,
# NUNCA se usan como features (serian nivel absoluto no estacionario).
PASSTHROUGH = ["open", "high", "low", "close", "volume"]


def build_features(raw: pd.DataFrame, warmup: int = 300) -> pd.DataFrame:
    """Devuelve un DataFrame con OHLCV + todas las features, sin NaN iniciales.

    `warmup` descarta las primeras barras donde los indicadores largos aun no
    tienen historia suficiente.
    """
    required = {"open", "high", "low", "close", "volume", "taker_buy_base", "quote_volume", "trades"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"faltan columnas en el dataset crudo: {sorted(missing)}")

    blocks = [raw[PASSTHROUGH], technical.build(raw), microstructure.build(raw)]
    ctx = context.build(raw)
    if not ctx.empty:
        blocks.append(ctx)
    else:
        log.warning("sin features de contexto (no hay funding/OI/sentimiento en el dataset)")

    df = clean_frame(pd.concat(blocks, axis=1))
    if warmup:
        df = df.iloc[warmup:]

    n_before = len(df)
    df = df.dropna()
    if len(df) < n_before:
        log.info("descartadas %d filas con NaN residuales", n_before - len(df))
    if df.empty:
        raise RuntimeError("la matriz de features quedo vacia; necesitas mas historico")

    log.info("features: %d filas x %d columnas", len(df), len(feature_columns(df)))
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Nombres de columnas usables como input del modelo."""
    return [c for c in df.columns if c.startswith(("ta_", "ms_", "ctx_", "book_"))]
