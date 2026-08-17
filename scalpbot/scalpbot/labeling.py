"""Etiquetado por triple barrera (Lopez de Prado, *Advances in Financial ML*).

Por que no etiquetar con "el retorno de la proxima barra":
  1. Un scalper no mantiene la posicion un tiempo fijo: sale por take-profit,
     por stop o por tiempo. La etiqueta debe reflejar esa regla de salida.
  2. Las barreras escaladas por volatilidad hacen que la etiqueta signifique lo
     mismo en regimen tranquilo y en regimen agitado.
  3. El signo del retorno a 1 barra es casi todo ruido de microestructura.

Se generan tres clases: +1 (toca la barrera superior primero), -1 (inferior
primero), 0 (expira por tiempo sin tocar ninguna).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import LabelConfig
from .utils import get_logger

log = get_logger("labeling")


@dataclass
class Labels:
    y: pd.Series              # clase en {-1, 0, 1}
    ret: pd.Series            # retorno logaritmico realizado hasta la salida
    t1: pd.Series             # indice entero de la barra de salida
    holding: pd.Series        # barras mantenidas
    barrier_up: pd.Series     # barrera de take-profit, fraccion del precio
    barrier_dn: pd.Series     # barrera de stop-loss, fraccion del precio
    ambiguous: pd.Series      # True si TP y SL se tocaron en la misma barra
    weights: pd.Series        # pesos de muestra para el entrenamiento

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "y": self.y, "ret": self.ret, "t1": self.t1, "holding": self.holding,
            "barrier_up": self.barrier_up, "barrier_dn": self.barrier_dn,
            "ambiguous": self.ambiguous, "weight": self.weights,
        })


def volatility(close: pd.Series, window: int = 60) -> pd.Series:
    """Sigma EWMA de retornos log POR BARRA. Causal."""
    ret = np.log(close).diff()
    return ret.ewm(span=window, adjust=False, min_periods=window // 2).std()


def barrier_fracs(close: pd.Series, cfg: LabelConfig) -> tuple[pd.Series, pd.Series]:
    """Tamano de las barreras TP y SL como fraccion del precio.

    La sigma por barra se escala por sqrt(horizonte): asi la barrera es del
    orden del movimiento tipico DURANTE la vida del trade. Sin este escalado
    las barreras son tan estrechas que el ruido las toca en 2-3 barras y el
    coste de transaccion se come cualquier edge.

    Ademas se impone `min_barrier_bps`, que debe ser holgadamente mayor que el
    coste de ida y vuelta: una barrera menor que los costes garantiza perdidas.
    """
    sigma = volatility(close, cfg.vol_window)
    scale = np.sqrt(cfg.horizon) if cfg.scale_by_horizon else 1.0
    floor = cfg.min_barrier_bps * 1e-4
    up = np.maximum(cfg.tp_sigma * sigma * scale, floor)
    dn = np.maximum(cfg.sl_sigma * sigma * scale, floor)
    return up, dn


def triple_barrier(df: pd.DataFrame, cfg: LabelConfig) -> Labels:
    """Aplica la triple barrera sobre OHLC.

    La entrada de la etiqueta se asume al CIERRE de la barra t (mismo instante
    en que se calculan las features). El recorrido empieza en t+1, de modo que
    la etiqueta nunca contiene informacion disponible antes de decidir.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(df)

    up_s, dn_s = barrier_fracs(df["close"], cfg)
    up_frac = up_s.to_numpy(dtype=float)
    dn_frac = dn_s.to_numpy(dtype=float)

    y = np.zeros(n, dtype=np.int8)
    ret = np.zeros(n, dtype=float)
    t1 = np.full(n, -1, dtype=np.int64)
    ambiguous = np.zeros(n, dtype=bool)
    valid = np.ones(n, dtype=bool)

    h = cfg.horizon
    for i in range(n):
        if not np.isfinite(up_frac[i]) or not np.isfinite(dn_frac[i]) or i + 1 >= n:
            valid[i] = False
            t1[i] = min(i + h, n - 1)
            continue
        entry = close[i]
        tp = entry * (1.0 + up_frac[i])
        sl = entry * (1.0 - dn_frac[i])
        end = min(i + h, n - 1)

        hit = 0
        exit_idx = end
        for j in range(i + 1, end + 1):
            touch_up = high[j] >= tp
            touch_dn = low[j] <= sl
            if touch_up and touch_dn:
                # Ambas en la misma vela: no sabemos el orden intrabar.
                # Asumimos el peor caso (stop primero): es la hipotesis
                # conservadora y evita inflar el backtest.
                hit, exit_idx, ambiguous[i] = -1, j, True
                break
            if touch_up:
                hit, exit_idx = 1, j
                break
            if touch_dn:
                hit, exit_idx = -1, j
                break

        y[i] = hit
        t1[i] = exit_idx
        if hit == 1:
            ret[i] = np.log(tp / entry)
        elif hit == -1:
            ret[i] = np.log(sl / entry)
        else:
            ret[i] = np.log(close[exit_idx] / entry)

    idx = df.index
    holding = pd.Series(t1 - np.arange(n), index=idx, name="holding")
    labels = Labels(
        y=pd.Series(y, index=idx, name="y"),
        ret=pd.Series(ret, index=idx, name="ret"),
        t1=pd.Series(t1, index=idx, name="t1"),
        holding=holding,
        barrier_up=pd.Series(up_frac, index=idx, name="barrier_up"),
        barrier_dn=pd.Series(dn_frac, index=idx, name="barrier_dn"),
        ambiguous=pd.Series(ambiguous, index=idx, name="ambiguous"),
        weights=pd.Series(1.0, index=idx, name="weight"),
    )
    labels.weights = sample_weights(labels, valid)

    dist = pd.Series(y).value_counts(normalize=True).sort_index()
    log.info("etiquetas: %s | ambiguas %.1f%% | holding medio %.1f barras",
             {int(k): f"{v:.1%}" for k, v in dist.items()},
             100 * ambiguous.mean(), holding.mean())
    return labels


def sample_weights(labels: Labels, valid: np.ndarray) -> pd.Series:
    """Pesos por unicidad y por magnitud del retorno.

    Etiquetas solapadas (la de t y la de t+1 comparten barras futuras) violan
    la independencia. Se penalizan por el numero de etiquetas concurrentes.
    Ademas se pondera por |retorno|: los eventos grandes importan mas.
    """
    n = len(labels.y)
    t1 = labels.t1.to_numpy()
    # Concurrencia: cuantas etiquetas estan "vivas" en cada barra.
    counts = np.zeros(n + 1, dtype=float)
    for i in range(n):
        end = min(int(t1[i]), n - 1)
        counts[i] += 1.0
        counts[end + 1] -= 1.0
    concurrency = np.cumsum(counts[:n])
    concurrency = np.maximum(concurrency, 1.0)

    uniqueness = np.zeros(n)
    for i in range(n):
        end = min(int(t1[i]), n - 1)
        uniqueness[i] = np.mean(1.0 / concurrency[i:end + 1])

    magnitude = np.abs(labels.ret.to_numpy())
    magnitude = magnitude / (np.nanmean(magnitude) + 1e-12)
    w = uniqueness * np.clip(magnitude, 0.1, 5.0)
    w = np.where(valid, w, 0.0)
    mean_w = w[w > 0].mean() if (w > 0).any() else 1.0
    return pd.Series(w / mean_w, index=labels.y.index, name="weight")
