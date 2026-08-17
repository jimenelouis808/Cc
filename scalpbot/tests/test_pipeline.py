"""Tests del pipeline. Los criticos son los de fuga de informacion."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scalpbot.backtest.engine import run_backtest
from scalpbot.backtest.metrics import compute_stats
from scalpbot.config import Config, CostConfig, LabelConfig, StrategyConfig
from scalpbot.data.synthetic import generate_context, generate_klines
from scalpbot.features.builder import build_features, feature_columns
from scalpbot.labeling import triple_barrier
from scalpbot.model.cv import PurgedWalkForward
from scalpbot.strategy.policy import decide_batch, expected_edge
from scalpbot.strategy.risk import RiskManager


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    kl = generate_klines(n_bars=3000, seed=3)
    return kl.join(generate_context(kl, seed=5).shift(1))


@pytest.fixture(scope="module")
def feats(raw: pd.DataFrame) -> pd.DataFrame:
    return build_features(raw, warmup=300)


# --------------------------------------------------------------- sin lookahead

def test_features_are_causal(raw: pd.DataFrame):
    """Alterar el futuro no debe cambiar ninguna feature del pasado.

    Es el test mas importante del repositorio: una fuga aqui invalida
    absolutamente todos los resultados posteriores.
    """
    cut = 2000
    full = build_features(raw, warmup=300)

    tampered = raw.copy()
    for col in ("open", "high", "low", "close"):
        tampered.iloc[cut:, tampered.columns.get_loc(col)] *= 1.25
    tampered.iloc[cut:, tampered.columns.get_loc("volume")] *= 4.0
    partial = build_features(tampered, warmup=300)

    common = full.index.intersection(partial.index)
    common = common[common < raw.index[cut]]
    assert len(common) > 500, "muy pocas filas comunes para que el test valga"

    cols = feature_columns(full)
    a = full.loc[common, cols].to_numpy()
    b = partial.loc[common, cols].to_numpy()
    finite = np.isfinite(a) & np.isfinite(b)
    assert np.allclose(a[finite], b[finite], rtol=1e-9, atol=1e-12), \
        "una feature del pasado cambio al modificar el futuro: hay lookahead"


def test_labels_use_only_future(feats: pd.DataFrame):
    """La etiqueta de t debe expirar en (t, t+horizon]."""
    cfg = LabelConfig(horizon=10)
    lab = triple_barrier(feats, cfg)
    pos = np.arange(len(feats))
    holding = lab.t1.to_numpy() - pos
    interior = holding[:-cfg.horizon - 1]
    assert (interior >= 1).all(), "alguna etiqueta expira en el pasado o en la propia barra"
    assert (interior <= cfg.horizon).all(), "alguna etiqueta excede el horizonte"


# ------------------------------------------------------------- triple barrera

def test_triple_barrier_hits_take_profit():
    """Serie que sube en linea recta: todas las etiquetas iniciales deben ser +1."""
    n = 60
    close = pd.Series(np.linspace(100, 130, n))
    df = pd.DataFrame({"open": close, "high": close * 1.001,
                       "low": close * 0.999, "close": close})
    lab = triple_barrier(df, LabelConfig(horizon=10, vol_window=10, min_barrier_bps=10.0))
    mid = lab.y.iloc[20:40]
    assert (mid == 1).all(), f"esperaba todo +1 en tendencia alcista, obtuve {mid.unique()}"


def test_triple_barrier_ambiguous_assumes_stop():
    """Si TP y SL caen en la misma vela se asume el stop (conservador)."""
    n = 30
    close = pd.Series([100.0] * n)
    high = pd.Series([100.0] * n)
    low = pd.Series([100.0] * n)
    high.iloc[5] = 200.0   # toca ambas barreras en la misma vela
    low.iloc[5] = 50.0
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close})
    # sigma sera ~0, asi que manda el suelo min_barrier_bps
    lab = triple_barrier(df, LabelConfig(horizon=10, vol_window=5, min_barrier_bps=50.0))
    assert lab.y.iloc[4] == -1, "la vela ambigua deberia etiquetarse como stop"
    assert bool(lab.ambiguous.iloc[4]) is True


# --------------------------------------------------------------- CV purgada

def test_purged_walk_forward_is_temporal():
    """Ningun indice de train puede caer dentro o despues del test."""
    n, embargo = 10_000, 50
    t1 = np.minimum(np.arange(n) + 20, n - 1)
    cv = PurgedWalkForward(n_splits=4, embargo=embargo, min_train=2000)
    splits = list(cv.split(n, t1))
    assert len(splits) >= 3

    for sp in splits:
        assert sp.train_idx.max() < sp.test_idx.min(), "train invade el test"
        # Purga: ninguna etiqueta de train puede expirar cerca del test.
        assert (t1[sp.train_idx] < sp.test_idx.min() - embargo).all(), \
            "la purga no elimino etiquetas solapadas"


def test_purged_walk_forward_rejects_short_history():
    cv = PurgedWalkForward(n_splits=6, embargo=10, min_train=5000)
    with pytest.raises(ValueError, match="historico insuficiente"):
        list(cv.split(5003))


# ------------------------------------------------------------------- costes

def test_round_trip_cost():
    c = CostConfig(taker_fee_bps=5.0, maker_fee_bps=2.0, slippage_bps=1.0)
    assert c.round_trip_bps() == pytest.approx(12.0)
    c.entry_is_maker = c.exit_is_maker = True
    assert c.round_trip_bps() == pytest.approx(6.0)


def test_expected_edge_subtracts_cost():
    p_up = np.array([0.6]); p_dn = np.array([0.2])
    tp = np.array([30.0]); sl = np.array([30.0])
    ev_long, ev_short = expected_edge(p_up, p_dn, tp, sl, cost_bps=12.0)
    # 0.6*30 - 0.2*30 - 12 = 18 - 6 - 12 = 0
    assert ev_long[0] == pytest.approx(0.0)
    assert ev_short[0] == pytest.approx(0.2 * 30 - 0.6 * 30 - 12)


def test_policy_rejects_when_cost_exceeds_edge():
    """Con barreras mas pequenas que los costes no debe abrirse nada."""
    proba = pd.DataFrame({"p_dn": [0.05], "p_flat": [0.15], "p_up": [0.80]}, index=[0])
    tp = pd.Series([8.0], index=[0])   # barrera 8 bps
    sl = pd.Series([8.0], index=[0])
    costs = CostConfig(taker_fee_bps=5.0, slippage_bps=1.0)  # 12 bps ida y vuelta
    out = decide_batch(proba, tp, sl, StrategyConfig(min_edge_bps=2.0), costs)
    assert out["side"].iloc[0] == 0, "abrio con una barrera menor que el coste"


def test_policy_opens_with_real_edge():
    proba = pd.DataFrame({"p_dn": [0.10], "p_flat": [0.20], "p_up": [0.70]}, index=[0])
    tp = sl = pd.Series([40.0], index=[0])
    out = decide_batch(proba, tp, sl, StrategyConfig(min_edge_bps=2.0),
                       CostConfig(taker_fee_bps=5.0, slippage_bps=1.0))
    assert out["side"].iloc[0] == 1
    assert 0 < out["size_frac"].iloc[0] <= StrategyConfig().max_position_pct


# ------------------------------------------------------------------- riesgo

def test_consecutive_loss_breaker_resets_daily():
    """Regresion: el bloqueo por racha debe levantarse al cambiar de dia."""
    cfg = Config()
    cfg.risk.max_consecutive_losses = 3
    rm = RiskManager(cfg.risk)
    rm.roll_day(pd.Timestamp("2025-01-01", tz="UTC"))
    for _ in range(3):
        rm.register_close(-10.0)
    assert rm.can_open()[0] is False
    rm.roll_day(pd.Timestamp("2025-01-02", tz="UTC"))
    assert rm.can_open()[0] is True, "el bot quedaria bloqueado para siempre"


def test_daily_loss_limit_blocks():
    cfg = Config()
    cfg.risk.max_daily_loss_pct = 2.0
    rm = RiskManager(cfg.risk)
    rm.roll_day(pd.Timestamp("2025-01-01", tz="UTC"))
    rm.register_close(-0.021 * cfg.risk.initial_equity)
    ok, reason = rm.can_open()
    assert ok is False and "diario" in reason


# ------------------------------------------------------------------ backtest

def _flat_bars(n: int, price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open": price, "high": price, "low": price, "close": price},
                        index=idx)


def test_backtest_charges_costs_on_flat_market():
    """Mercado plano: cada operacion debe perder exactamente los costes."""
    n = 40
    bars = _flat_bars(n)
    sig = pd.DataFrame({
        "side": [0] * n, "size_frac": [0.0] * n, "edge_bps": [0.0] * n,
        "tp_bps": [50.0] * n, "sl_bps": [50.0] * n,
        "p_up": [0.5] * n, "p_dn": [0.5] * n,
    }, index=bars.index)
    sig.iloc[0, sig.columns.get_loc("side")] = 1
    sig.iloc[0, sig.columns.get_loc("size_frac")] = 0.10

    cfg = Config()
    cfg.labels.horizon = 5
    cfg.strategy.exit_on_flip = False
    cfg.costs = CostConfig(taker_fee_bps=5.0, maker_fee_bps=5.0, slippage_bps=1.0)

    res = run_backtest(cfg, bars, sig)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t["reason"] == "timeout"
    # 12 bps de coste total sobre el nocional, con signo negativo.
    assert t["ret_bps"] == pytest.approx(-12.0, abs=0.3)


def test_backtest_executes_at_next_open():
    """La senal de la barra i debe llenarse a la apertura de i+1, no en i."""
    n = 20
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    price = np.full(n, 100.0)
    price[6:] = 110.0    # salto justo despues de la senal
    bars = pd.DataFrame({"open": price, "high": price, "low": price, "close": price},
                        index=idx)
    sig = pd.DataFrame({
        "side": 0, "size_frac": 0.0, "edge_bps": 0.0, "tp_bps": 5000.0,
        "sl_bps": 5000.0, "p_up": 0.5, "p_dn": 0.5}, index=idx)
    sig.iloc[5, sig.columns.get_loc("side")] = 1
    sig.iloc[5, sig.columns.get_loc("size_frac")] = 0.1

    cfg = Config()
    cfg.labels.horizon = 5
    cfg.costs = CostConfig(taker_fee_bps=0.0, maker_fee_bps=0.0, slippage_bps=0.0)
    res = run_backtest(cfg, bars, sig)
    # Se entra al open de la barra 6, que ya vale 110: no se captura el salto.
    assert res.trades.iloc[0]["entry_px"] == pytest.approx(110.0)
    assert res.trades.iloc[0]["pnl"] == pytest.approx(0.0, abs=1e-9)


def test_backtest_take_profit_and_stop():
    n = 30
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    bars = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
                        index=idx)
    bars.iloc[8, bars.columns.get_loc("high")] = 105.0  # dispara el TP

    sig = pd.DataFrame({"side": 0, "size_frac": 0.0, "edge_bps": 0.0,
                        "tp_bps": 100.0, "sl_bps": 100.0, "p_up": 0.6, "p_dn": 0.2},
                       index=idx)
    sig.iloc[5, sig.columns.get_loc("side")] = 1
    sig.iloc[5, sig.columns.get_loc("size_frac")] = 0.1

    cfg = Config()
    cfg.labels.horizon = 20
    cfg.costs = CostConfig(taker_fee_bps=0.0, maker_fee_bps=0.0, slippage_bps=0.0)
    res = run_backtest(cfg, bars, sig)
    assert res.trades.iloc[0]["reason"] == "take_profit"
    assert res.trades.iloc[0]["pnl"] > 0


def test_metrics_on_empty_trades():
    cfg = Config()
    eq = pd.Series([10_000.0] * 100,
                   index=pd.date_range("2025-01-01", periods=100, freq="1min", tz="UTC"))
    stats = compute_stats(cfg, eq, pd.DataFrame())
    assert stats["n_trades"] == 0
    assert stats["total_return_pct"] == pytest.approx(0.0)


# ------------------------------------------------------- control negativo ML

def test_no_signal_data_yields_no_skill():
    """Con datos de ruido puro el modelo no debe encontrar edge alguno.

    Si este test empieza a fallar, es que hay una fuga en el pipeline.
    """
    from scalpbot.model.train import train_walk_forward

    kl = generate_klines(n_bars=9000, seed=99, signal_strength=0.0)
    f = build_features(kl, warmup=300)
    cfg = Config()
    cfg.model.n_splits = 3
    cfg.model.min_train_bars = 2000
    cfg.model.n_estimators = 60
    cfg.model.max_features = 25
    lab = triple_barrier(f, cfg.labels)

    res = train_walk_forward(cfg, f, lab)
    acc = res.fold_report["dir_accuracy"].mean()
    assert 0.40 < acc < 0.60, (
        f"precision direccional {acc:.3f} sobre ruido puro: "
        "esto indica fuga de informacion en el pipeline")
