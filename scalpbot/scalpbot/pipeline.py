"""Pipeline de alto nivel: datos -> features -> etiquetas -> modelo -> backtest."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest.engine import BacktestResult, run_backtest
from .backtest.metrics import compute_stats, cost_sensitivity, format_stats
from .config import Config
from .data.loader import dataset_stem, load_raw
from .features.builder import build_features, feature_columns
from .labeling import Labels, triple_barrier
from .model.registry import ModelBundle
from .model.train import TrainResult, train_walk_forward
from .strategy.policy import decide_batch
from .utils import ensure_dir, get_logger, read_table, write_table

log = get_logger("pipeline")


def artifacts_dir(cfg: Config) -> Path:
    return ensure_dir(cfg.data_path / "artifacts")


def model_path(cfg: Config) -> Path:
    d = cfg.data
    return artifacts_dir(cfg) / f"model_{d.symbol}_{d.interval}_{d.market}.pkl"


def build_dataset(cfg: Config) -> tuple[pd.DataFrame, Labels]:
    """Carga el crudo, construye features y etiqueta con triple barrera."""
    raw = load_raw(cfg)
    feats = build_features(raw)
    labels = triple_barrier(feats, cfg.labels)
    stem = dataset_stem(cfg)
    write_table(feats, stem.with_name(stem.name + "_features"))
    write_table(labels.to_frame(), stem.with_name(stem.name + "_labels"))
    return feats, labels


def load_dataset(cfg: Config) -> tuple[pd.DataFrame, Labels]:
    """Recarga features y etiquetas ya materializadas, o las construye."""
    stem = dataset_stem(cfg)
    try:
        feats = read_table(stem.with_name(stem.name + "_features"))
        lab = read_table(stem.with_name(stem.name + "_labels"))
    except FileNotFoundError:
        return build_dataset(cfg)
    for df in (feats, lab):
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
    labels = Labels(y=lab["y"].astype(int), ret=lab["ret"], t1=lab["t1"].astype(int),
                    holding=lab["holding"], barrier_up=lab["barrier_up"],
                    barrier_dn=lab["barrier_dn"],
                    ambiguous=lab["ambiguous"].astype(bool), weights=lab["weight"])
    return feats, labels


def train(cfg: Config) -> TrainResult:
    """Entrena walk-forward y persiste modelo + predicciones OOS."""
    feats, labels = load_dataset(cfg)
    result = train_walk_forward(cfg, feats, labels)
    result.bundle.save(model_path(cfg))

    stem = dataset_stem(cfg)
    write_table(result.oos, stem.with_name(stem.name + "_oos"))
    (artifacts_dir(cfg) / "fold_report.csv").write_text(result.fold_report.to_csv(index=False))
    (artifacts_dir(cfg) / "feature_importance.csv").write_text(
        result.importances.to_csv(header=["importance"]))
    log.info("top-15 features: %s", list(result.importances.head(15).index))
    return result


@dataclass
class BacktestReport:
    result: BacktestResult
    stats: dict[str, Any]
    sensitivity: pd.DataFrame
    signals: pd.DataFrame


def backtest(cfg: Config, use_oos: bool = True) -> BacktestReport:
    """Backtestea sobre las predicciones OOS del walk-forward."""
    feats, labels = load_dataset(cfg)
    stem = dataset_stem(cfg)

    if use_oos:
        try:
            proba = read_table(stem.with_name(stem.name + "_oos"))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "no hay predicciones OOS; ejecuta 'train' antes de 'backtest'") from e
        if not isinstance(proba.index, pd.DatetimeIndex):
            proba.index = pd.to_datetime(proba.index, utc=True)
    else:
        # Modo diagnostico: predicciones IN-SAMPLE. Los resultados NO son reales.
        bundle = ModelBundle.load(model_path(cfg))
        X = feats.reindex(columns=bundle.features)
        p = bundle.model.predict_proba(X)
        proba = pd.DataFrame(p, index=feats.index, columns=["p_dn", "p_flat", "p_up"])
        log.warning("BACKTEST IN-SAMPLE: solo para depurar, no para decidir nada")

    tp_bps = (labels.barrier_up.reindex(proba.index) * 1e4).fillna(cfg.labels.min_barrier_bps)
    sl_bps = (labels.barrier_dn.reindex(proba.index) * 1e4).fillna(cfg.labels.min_barrier_bps)
    signals = decide_batch(proba[["p_dn", "p_flat", "p_up"]], tp_bps, sl_bps,
                           cfg.strategy, cfg.costs)

    bars = feats.loc[proba.index, ["open", "high", "low", "close"]]
    result = run_backtest(cfg, bars, signals)
    stats = compute_stats(cfg, result.equity, result.trades)
    sens = cost_sensitivity(cfg, result.trades)

    out = artifacts_dir(cfg)
    if not result.trades.empty:
        result.trades.to_csv(out / "trades.csv", index=False)
    result.equity.to_csv(out / "equity.csv")
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    if not sens.empty:
        sens.to_csv(out / "cost_sensitivity.csv", index=False)

    return BacktestReport(result=result, stats=stats, sensitivity=sens, signals=signals)


def sweep_min_edge(cfg: Config, values: tuple[float, ...]) -> pd.DataFrame:
    """Barre el umbral de EV minimo y reporta el resultado de cada nivel.

    Es el diagnostico mas informativo del paquete. Un edge real casi siempre
    aparece concentrado en las senales mas fuertes: si el rendimiento no mejora
    al ser mas selectivo, lo que tienes es ruido, no una estrategia.

    Ojo: este barrido evalua muchos umbrales sobre los MISMOS datos. El mejor
    valor esta sesgado al alza por seleccion. Usalo para entender la forma de
    la curva, y confirma el umbral elegido en datos que no hayas mirado.
    """
    rows = []
    original = cfg.strategy.min_edge_bps
    try:
        for v in values:
            cfg.strategy.min_edge_bps = v
            rep = backtest(cfg, use_oos=True)
            s = rep.stats
            rows.append({
                "min_edge_bps": v,
                "n_trades": s.get("n_trades", 0),
                "avg_trade_bps": s.get("avg_trade_bps", float("nan")),
                "win_rate_pct": s.get("win_rate_pct", float("nan")),
                "total_return_pct": s.get("total_return_pct", float("nan")),
                "sharpe": s.get("sharpe", float("nan")),
                "max_dd_pct": s.get("max_drawdown_pct", float("nan")),
                "t_stat": s.get("t_stat", float("nan")),
            })
    finally:
        cfg.strategy.min_edge_bps = original
    df = pd.DataFrame(rows)
    df.to_csv(artifacts_dir(cfg) / "edge_sweep.csv", index=False)
    return df


def print_report(cfg: Config, report: BacktestReport) -> None:
    print("\n" + "=" * 62)
    print(f"  BACKTEST  {cfg.data.symbol} {cfg.data.interval} ({cfg.data.market})")
    print("=" * 62)
    print(format_stats(report.stats))
    if not report.sensitivity.empty:
        print("\n  Sensibilidad a costes (bps extra por ida y vuelta):")
        for _, r in report.sensitivity.iterrows():
            flag = "OK " if r["profitable"] else "NEG"
            print(f"    +{r['extra_cost_bps']:>4.1f} bps -> "
                  f"{r['avg_trade_bps']:+7.2f} bps/trade  "
                  f"${r['total_pnl_usd']:>+12,.2f}  [{flag}]")
    n = report.signals["side"].ne(0).sum()
    print(f"\n  Senales generadas: {n:,} de {len(report.signals):,} barras "
          f"({100 * n / max(len(report.signals), 1):.1f}%)")
    print("=" * 62 + "\n")
