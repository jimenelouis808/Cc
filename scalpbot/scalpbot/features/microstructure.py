"""Features de microestructura derivadas de klines de Binance.

Las klines traen `taker_buy_base`: el volumen ejecutado por compradores
agresores. Con eso se reconstruye el desequilibrio de flujo de ordenes (OFI),
que es la senal con mas contenido predictivo a horizontes de segundos-minutos.
Tambien se estiman impacto de precio (Kyle lambda, Amihud) y toxicidad del
flujo (VPIN simplificado).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import safe_div, zscore


def build(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    ret = np.log(c).diff()
    vol = df["volume"].replace(0, np.nan)
    out: dict[str, pd.Series] = {}

    # --- Desequilibrio de agresores: (compras - ventas) / total, en [-1, 1]
    buy = df["taker_buy_base"]
    sell = df["volume"] - buy
    imb = pd.Series(safe_div(buy - sell, df["volume"], fill=np.nan), index=df.index)
    out["ofi"] = imb
    for w in (3, 5, 15, 60):
        out[f"ofi_ma{w}"] = imb.rolling(w, min_periods=max(2, w // 3)).mean()
    out["ofi_z_240"] = zscore(imb, 240)

    # --- OFI ponderado por volumen: un desequilibrio con volumen pesa mas
    signed_vol = (buy - sell)
    out["signed_vol_z"] = zscore(signed_vol, 240)
    cum = signed_vol.rolling(30, min_periods=10).sum()
    out["cvd_30"] = pd.Series(
        safe_div(cum, df["volume"].rolling(30, min_periods=10).sum(), fill=np.nan),
        index=df.index)
    out["cvd_slope"] = out["cvd_30"].diff(5)

    # --- Divergencia flujo/precio: el flujo empuja pero el precio no cede
    ret_z = zscore(ret, 240)
    out["flow_price_div"] = out["ofi_z_240"] - ret_z

    # --- Actividad: numero de trades y tamano medio
    trades = df["trades"].replace(0, np.nan)
    out["trades_z"] = zscore(np.log1p(df["trades"]), 240)
    out["avg_trade_size"] = pd.Series(safe_div(vol, trades, fill=np.nan), index=df.index)
    out["avg_trade_size_z"] = zscore(np.log1p(out["avg_trade_size"]), 240)
    out["vol_z"] = zscore(np.log1p(df["volume"]), 240)
    out["vol_burst"] = pd.Series(
        safe_div(df["volume"], df["volume"].rolling(60, min_periods=20).mean(), fill=np.nan),
        index=df.index)

    # --- Impacto de precio: cuanto mueve el precio una unidad de volumen
    out["amihud"] = pd.Series(safe_div(ret.abs(), df["quote_volume"], fill=np.nan),
                              index=df.index).rolling(30, min_periods=10).mean() * 1e9
    out["kyle_lambda"] = pd.Series(
        safe_div(ret.abs().rolling(30, min_periods=10).mean(),
                 signed_vol.abs().rolling(30, min_periods=10).mean(), fill=np.nan),
        index=df.index) * 1e4

    # --- VPIN simplificado: fraccion de volumen desequilibrado (toxicidad)
    out["vpin_50"] = pd.Series(
        safe_div(signed_vol.abs().rolling(50, min_periods=20).sum(),
                 df["volume"].rolling(50, min_periods=20).sum(), fill=np.nan),
        index=df.index)

    # --- Spread efectivo (Corwin-Schultz simplificado sobre high/low)
    hl = np.log(df["high"] / df["low"].replace(0, np.nan))
    out["hl_spread_proxy"] = hl.rolling(5, min_periods=2).mean()
    out["spread_z"] = zscore(out["hl_spread_proxy"], 240)

    # --- Desviacion del VWAP rodante: presion de reversion intradia
    vwap = (df["quote_volume"].rolling(60, min_periods=20).sum()
            / df["volume"].rolling(60, min_periods=20).sum().replace(0, np.nan))
    out["vwap_dev"] = (c - vwap) / c
    out["vwap_dev_z"] = zscore(out["vwap_dev"], 240)

    # --- Eficiencia del movimiento: |retorno neto| / suma de |retornos|
    out["efficiency_20"] = pd.Series(
        safe_div(ret.rolling(20, min_periods=10).sum().abs(),
                 ret.abs().rolling(20, min_periods=10).sum(), fill=np.nan),
        index=df.index)

    return pd.DataFrame(out, index=df.index).add_prefix("ms_")


def book_features(bids: list[list], asks: list[list], mid: float | None = None) -> dict[str, float]:
    """Features de un snapshot del libro (solo disponible en vivo).

    bids/asks: listas [[precio, cantidad], ...] ordenadas de mejor a peor.
    """
    if not bids or not asks:
        return {}
    bid_p = np.array([float(b[0]) for b in bids])
    bid_q = np.array([float(b[1]) for b in bids])
    ask_p = np.array([float(a[0]) for a in asks])
    ask_q = np.array([float(a[1]) for a in asks])

    best_bid, best_ask = bid_p[0], ask_p[0]
    mid_px = mid if mid is not None else (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid

    out = {
        "book_spread_bps": spread / mid_px * 1e4,
        "book_microprice_dev_bps": (
            (best_bid * ask_q[0] + best_ask * bid_q[0]) / (bid_q[0] + ask_q[0]) - mid_px
        ) / mid_px * 1e4,
    }
    for depth in (5, 10, 20):
        bq, aq = bid_q[:depth].sum(), ask_q[:depth].sum()
        out[f"book_imb_{depth}"] = (bq - aq) / (bq + aq) if (bq + aq) > 0 else 0.0
    # Pendiente del libro: cuanta liquidez por bps de distancia al mid
    out["book_slope_bid"] = float(bid_q[:20].sum() / max((mid_px - bid_p[:20].min()) / mid_px * 1e4, 1e-6))
    out["book_slope_ask"] = float(ask_q[:20].sum() / max((ask_p[:20].max() - mid_px) / mid_px * 1e4, 1e-6))
    return out
