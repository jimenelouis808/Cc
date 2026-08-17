"""Indicadores tecnicos en pandas puro (sin TA-Lib).

Todos son causales: el valor en la barra t solo usa informacion hasta t
inclusive. Se devuelven en forma normalizada (z-scores, ratios, distancias
relativas) porque los niveles absolutos no son estacionarios y destruyen la
capacidad de generalizacion del modelo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import rank_pct, safe_div, zscore


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=max(2, span // 3)).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = pd.Series(safe_div(avg_gain, avg_loss, fill=np.nan), index=close.index)
    return 100.0 - 100.0 / (1.0 + rs)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Fuerza de tendencia. Alto = tendencia; bajo = rango (clave para el regimen)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=1 / window, adjust=False, min_periods=window).mean() / atr_.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=1 / window, adjust=False, min_periods=window).mean() / atr_.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               window: int = 14, smooth: int = 3) -> pd.Series:
    hh = high.rolling(window, min_periods=window // 2).max()
    ll = low.rolling(window, min_periods=window // 2).min()
    k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    return k.rolling(smooth, min_periods=1).mean()


def bollinger_position(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """Posicion dentro de las bandas: -1 banda inferior, +1 banda superior."""
    ma = close.rolling(window, min_periods=window // 2).mean()
    sd = close.rolling(window, min_periods=window // 2).std(ddof=0)
    return (close - ma) / (n_std * sd).replace(0, np.nan)


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    line = ema(close, fast) - ema(close, slow)
    return line - ema(line, signal)


def realized_vol(ret: pd.Series, window: int) -> pd.Series:
    return ret.rolling(window, min_periods=max(3, window // 3)).std(ddof=0)


def hurst_proxy(ret: pd.Series, window: int = 100) -> pd.Series:
    """Ratio de varianza: <1 reversion a la media, >1 tendencia, ~1 aleatorio."""
    var1 = ret.rolling(window, min_periods=window // 2).var(ddof=0)
    var5 = ret.rolling(5).sum().rolling(window, min_periods=window // 2).var(ddof=0)
    return pd.Series(safe_div(var5, 5.0 * var1, fill=np.nan), index=ret.index)


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Construye el bloque de features tecnicas."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    ret = np.log(c).diff()
    out: dict[str, pd.Series] = {}

    # --- Momento multi-escala, escalado por volatilidad para ser comparable
    vol60 = realized_vol(ret, 60)
    for w in (1, 3, 5, 10, 20, 60, 120):
        out[f"ret_{w}"] = ret.rolling(w, min_periods=1).sum() / (vol60 * np.sqrt(w)).replace(0, np.nan)

    # --- Aceleracion del momento (cambio de la pendiente)
    out["mom_accel"] = out["ret_5"] - out["ret_20"]

    # --- Distancia a medias moviles, en unidades de ATR
    atr14 = atr(h, l, c, 14)
    atr_rel = (atr14 / c).replace(0, np.nan)
    for span in (9, 21, 55, 200):
        out[f"dist_ema{span}"] = (c - ema(c, span)) / (atr14).replace(0, np.nan)
    out["ema_slope_21"] = ema(c, 21).diff(5) / atr14.replace(0, np.nan)
    out["ema_cross_9_21"] = (ema(c, 9) - ema(c, 21)) / atr14.replace(0, np.nan)

    # --- Osciladores, centrados en 0
    out["rsi_14"] = (rsi(c, 14) - 50.0) / 50.0
    out["rsi_7"] = (rsi(c, 7) - 50.0) / 50.0
    out["stoch_14"] = (stochastic(h, l, c, 14) - 50.0) / 50.0
    out["macd_hist"] = macd_hist(c) / atr14.replace(0, np.nan)
    out["bb_pos_20"] = bollinger_position(c, 20)
    out["bb_pos_60"] = bollinger_position(c, 60)

    # --- Volatilidad y regimen
    out["atr_rel"] = atr_rel
    out["atr_rel_z"] = zscore(atr_rel, 480)
    out["vol_ratio_5_60"] = safe_div(realized_vol(ret, 5), vol60, fill=np.nan)
    out["vol_ratio_20_120"] = safe_div(realized_vol(ret, 20), realized_vol(ret, 120), fill=np.nan)
    out["adx_14"] = adx(h, l, c, 14) / 100.0
    out["hurst_100"] = hurst_proxy(ret, 100)
    out["vol_rank_1d"] = rank_pct(atr_rel, 1440)

    # --- Estructura de precio
    hh = h.rolling(60, min_periods=20).max()
    ll = l.rolling(60, min_periods=20).min()
    out["donchian_pos_60"] = 2 * (c - ll) / (hh - ll).replace(0, np.nan) - 1
    out["range_rel"] = (h - l) / c
    out["body_rel"] = (c - o).abs() / (h - l).replace(0, np.nan)
    out["upper_wick"] = (h - np.maximum(o, c)) / (h - l).replace(0, np.nan)
    out["lower_wick"] = (np.minimum(o, c) - l) / (h - l).replace(0, np.nan)
    out["close_loc"] = 2 * (c - l) / (h - l).replace(0, np.nan) - 1
    out["gap_rel"] = (o - c.shift(1)) / atr14.replace(0, np.nan)

    # --- Autocorrelacion reciente de retornos (persistencia vs reversion)
    out["autocorr_20"] = ret.rolling(20, min_periods=10).corr(ret.shift(1))

    # --- Estacionalidad intradia (el flujo cambia por sesion)
    minutes = df.index.hour * 60 + df.index.minute
    out["tod_sin"] = pd.Series(np.sin(2 * np.pi * minutes / 1440), index=df.index)
    out["tod_cos"] = pd.Series(np.cos(2 * np.pi * minutes / 1440), index=df.index)
    out["dow"] = pd.Series(df.index.dayofweek.astype(float), index=df.index)

    return pd.DataFrame(out, index=df.index).add_prefix("ta_")
