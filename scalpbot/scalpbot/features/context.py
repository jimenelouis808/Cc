"""Features de contexto/analisis general: funding, open interest, sentimiento.

Estas series llegan con baja frecuencia (5 min, 8 h, diaria). Ya vienen
retrasadas una barra desde el loader; aqui solo se transforman en variaciones
y z-scores, nunca en niveles crudos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import rank_pct, zscore

CONTEXT_SOURCES = ["funding_rate", "open_interest", "ls_ratio", "taker_ratio", "fear_greed"]


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el bloque de contexto. Columnas ausentes se omiten sin fallar."""
    out: dict[str, pd.Series] = {}
    c = df["close"]

    if "funding_rate" in df:
        f = df["funding_rate"].astype(float)
        out["funding"] = f * 1e4                      # en bps
        out["funding_z"] = zscore(f, 2880)            # ~2 dias en barras de 1m
        out["funding_chg"] = f.diff(480) * 1e4

    if "open_interest" in df:
        oi = df["open_interest"].astype(float)
        out["oi_chg_60"] = np.log(oi).diff(60)
        out["oi_chg_240"] = np.log(oi).diff(240)
        out["oi_z"] = zscore(np.log(oi), 2880)
        # OI subiendo + precio subiendo = posiciones nuevas long (continuacion).
        # OI bajando + precio subiendo = cierre de shorts (menos sostenible).
        price_chg = np.log(c).diff(60)
        out["oi_price_agree"] = np.sign(out["oi_chg_60"]) * np.sign(price_chg)

    if "ls_ratio" in df:
        ls = df["ls_ratio"].astype(float)
        out["ls_ratio_z"] = zscore(ls, 2880)
        out["ls_ratio_chg"] = ls.pct_change(60)
        out["ls_rank"] = rank_pct(ls, 2880)

    if "taker_ratio" in df:
        tr = df["taker_ratio"].astype(float)
        out["taker_ratio_z"] = zscore(tr, 1440)
        out["taker_ratio_chg"] = tr.diff(30)

    if "fear_greed" in df:
        fg = df["fear_greed"].astype(float)
        out["fear_greed"] = (fg - 50.0) / 50.0
        out["fear_greed_chg"] = fg.diff(1440) / 50.0

    if not out:
        return pd.DataFrame(index=df.index)
    return pd.DataFrame(out, index=df.index).add_prefix("ctx_")
