"""Interfaz de linea de comandos del bot.

Uso tipico:
    python -m scalpbot synth --bars 50000     # dataset de prueba offline
    python -m scalpbot download               # datos reales de Binance
    python -m scalpbot features
    python -m scalpbot train
    python -m scalpbot backtest
    python -m scalpbot paper
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import pipeline
from .config import Config, load_config
from .utils import get_logger

log = get_logger("cli")


def _cfg(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config)
    if args.symbol:
        cfg.data.symbol = args.symbol
    if args.interval:
        cfg.data.interval = args.interval
    if args.market:
        cfg.data.market = args.market
    if args.days:
        cfg.data.days = args.days
    if args.data_dir:
        cfg.data.data_dir = args.data_dir
    return cfg


def cmd_synth(args: argparse.Namespace) -> int:
    from .data.loader import make_synthetic
    cfg = _cfg(args)
    make_synthetic(cfg, n_bars=args.bars, seed=args.seed,
                   signal_strength=args.signal_strength)
    print(f"Dataset sintetico creado en {cfg.data_path}/")
    print("AVISO: datos simulados. Sirven para validar el codigo, no la rentabilidad.")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from .data.loader import download
    cfg = _cfg(args)
    try:
        download(cfg)
    except Exception as e:  # noqa: BLE001
        log.error("descarga fallida: %s", e)
        print("\nSi Binance esta bloqueado en tu red o region, usa 'synth' para "
              "probar el pipeline, o descarga los CSV publicos de "
              "https://data.binance.vision", file=sys.stderr)
        return 1
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    feats, labels = pipeline.build_dataset(cfg)
    from .features.builder import feature_columns
    cols = feature_columns(feats)
    print(f"Features: {len(feats):,} filas x {len(cols)} columnas")
    print(f"Rango: {feats.index[0]} -> {feats.index[-1]}")
    dist = labels.y.value_counts(normalize=True).sort_index()
    print("Distribucion de etiquetas: " +
          "  ".join(f"{int(k):+d}: {v:.1%}" for k, v in dist.items()))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    result = pipeline.train(cfg)
    print("\nInforme por fold:")
    print(result.fold_report.to_string(index=False))
    print(f"\nlogloss OOS medio: {result.bundle.meta['oos_logloss']:.4f}")
    print(f"Precision direccional OOS: {result.bundle.meta['oos_dir_accuracy']:.3f}")
    print(f"Modelo guardado en {pipeline.model_path(cfg)}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    if args.min_edge is not None:
        cfg.strategy.min_edge_bps = args.min_edge
    if args.taker_fee is not None:
        cfg.costs.taker_fee_bps = args.taker_fee
    if args.maker:
        # Simula ejecucion con ordenes limite en ambos lados. Optimista: asume
        # que TODAS se ejecutan. En la realidad parte no se llena y te pierdes
        # justo los movimientos que querias capturar.
        cfg.costs.entry_is_maker = cfg.costs.exit_is_maker = True
    report = pipeline.backtest(cfg, use_oos=not args.in_sample)
    pipeline.print_report(cfg, report)
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    values = tuple(float(v) for v in args.values.split(","))
    df = pipeline.sweep_min_edge(cfg, values)
    print("\n  Barrido del umbral de EV minimo")
    print("  " + "-" * 76)
    print("  " + df.to_string(index=False, float_format=lambda x: f"{x:8.2f}")
          .replace("\n", "\n  "))
    print("  " + "-" * 76)
    best = df.loc[df["avg_trade_bps"].idxmax()] if not df.empty else None
    if best is not None:
        print(f"\n  Mejor bps/trade: umbral {best['min_edge_bps']:.1f} bps "
              f"-> {best['avg_trade_bps']:+.2f} bps en {int(best['n_trades'])} operaciones")
        print("  Recuerda: este optimo esta sesgado por haber mirado los mismos datos.")
    print()
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    from .live.runner import LiveRunner
    from .model.registry import ModelBundle
    cfg = _cfg(args)
    bundle = ModelBundle.load(pipeline.model_path(cfg))
    runner = LiveRunner(cfg, bundle, paper=True, max_cycles=args.max_cycles)
    trades = runner.run()
    if not trades.empty:
        out = pipeline.artifacts_dir(cfg) / "paper_trades.csv"
        trades.to_csv(out, index=False)
        print(f"Operaciones registradas en {out}")
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from .live.runner import LiveRunner
    from .model.registry import ModelBundle
    cfg = _cfg(args)
    if args.real:
        cfg.live.dry_run = False
    if args.mainnet:
        cfg.live.testnet = False

    if not cfg.live.dry_run and not cfg.live.testnet:
        print("\n" + "!" * 62)
        print("  VAS A OPERAR CON DINERO REAL EN MAINNET.")
        print(f"  {cfg.data.symbol} {cfg.data.interval} | equity {cfg.risk.initial_equity}")
        print("!" * 62)
        if input("  Escribe 'ACEPTO EL RIESGO' para continuar: ") != "ACEPTO EL RIESGO":
            print("Cancelado.")
            return 1

    bundle = ModelBundle.load(pipeline.model_path(cfg))
    runner = LiveRunner(cfg, bundle, paper=False, max_cycles=args.max_cycles)
    trades = runner.run()
    if not trades.empty:
        trades.to_csv(pipeline.artifacts_dir(cfg) / "live_trades.csv", index=False)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    """Atajo: synth (si hace falta) -> features -> train -> backtest."""
    cfg = _cfg(args)
    from .data.loader import dataset_stem
    stem = dataset_stem(cfg)
    if not any(Path(f"{stem}_raw{ext}").exists() for ext in (".parquet", ".csv")):
        from .data.loader import make_synthetic
        log.info("no hay datos crudos; generando dataset sintetico")
        make_synthetic(cfg, n_bars=args.bars, seed=args.seed,
                       signal_strength=args.signal_strength)
    pipeline.build_dataset(cfg)
    pipeline.train(cfg)
    report = pipeline.backtest(cfg, use_oos=True)
    pipeline.print_report(cfg, report)
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    print(json.dumps(_cfg(args).to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scalpbot",
        description="Bot de micro-scalping para Binance con ML y walk-forward purgado.")
    p.add_argument("--config", default="config.yaml", help="ruta al YAML de configuracion")
    p.add_argument("--symbol", help="par, p.ej. BTCUSDT")
    p.add_argument("--interval", help="intervalo de vela, p.ej. 1m")
    p.add_argument("--market", choices=["spot", "futures"])
    p.add_argument("--days", type=int, help="dias de historico a descargar")
    p.add_argument("--data-dir", help="directorio de datos")

    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="genera un dataset sintetico para probar offline")
    s.add_argument("--bars", type=int, default=50_000)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--signal-strength", type=float, default=0.30,
                   help="0 = ruido puro (control negativo)")
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("download", help="descarga klines y contexto de Binance")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("features", help="construye features y etiquetas")
    s.set_defaults(func=cmd_features)

    s = sub.add_parser("train", help="entrena walk-forward y guarda el modelo")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("backtest", help="backtestea sobre predicciones OOS")
    s.add_argument("--in-sample", action="store_true",
                   help="usa predicciones in-sample (SOLO depuracion)")
    s.add_argument("--min-edge", type=float, help="sobrescribe strategy.min_edge_bps")
    s.add_argument("--taker-fee", type=float, help="sobrescribe costs.taker_fee_bps")
    s.add_argument("--maker", action="store_true",
                   help="simula ordenes limite en ambos lados (asume ejecucion total)")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("sweep", help="barre el umbral de EV y mide la selectividad")
    s.add_argument("--values", default="0,2,5,10,15,20,25,30",
                   help="lista de umbrales en bps separados por comas")
    s.set_defaults(func=cmd_sweep)

    s = sub.add_parser("paper", help="trading en papel con datos reales en vivo")
    s.add_argument("--max-cycles", type=int, help="para automaticamente tras N ciclos")
    s.set_defaults(func=cmd_paper)

    s = sub.add_parser("live", help="trading real (por defecto testnet + dry-run)")
    s.add_argument("--real", action="store_true", help="desactiva dry-run: envia ordenes")
    s.add_argument("--mainnet", action="store_true", help="usa mainnet en vez de testnet")
    s.add_argument("--max-cycles", type=int)
    s.set_defaults(func=cmd_live)

    s = sub.add_parser("all", help="pipeline completo de principio a fin")
    s.add_argument("--bars", type=int, default=50_000)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--signal-strength", type=float, default=0.30)
    s.set_defaults(func=cmd_all)

    s = sub.add_parser("config", help="muestra la configuracion efectiva")
    s.set_defaults(func=cmd_config)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrumpido.")
        return 130
    except Exception as e:  # noqa: BLE001
        log.error("%s: %s", type(e).__name__, e)
        if "--debug" in (argv or sys.argv):
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
