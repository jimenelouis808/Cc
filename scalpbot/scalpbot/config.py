"""Configuracion tipada del bot, cargada desde YAML con valores por defecto."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    market: str = "futures"          # "futures" | "spot"
    days: int = 60                   # historico a descargar
    data_dir: str = "data"
    use_context: bool = True         # funding / open interest / long-short / fear&greed


@dataclass
class CostConfig:
    """Costes de transaccion. Son EL factor decisivo en micro-scalping."""
    taker_fee_bps: float = 5.0       # 0.050% futures USDT-M taker (VIP0)
    maker_fee_bps: float = 2.0       # 0.020% futures USDT-M maker (VIP0)
    slippage_bps: float = 1.0        # deslizamiento medio por lado
    entry_is_maker: bool = False
    exit_is_maker: bool = False

    def round_trip_bps(self) -> float:
        entry = self.maker_fee_bps if self.entry_is_maker else self.taker_fee_bps
        exit_ = self.maker_fee_bps if self.exit_is_maker else self.taker_fee_bps
        return entry + exit_ + 2 * self.slippage_bps


@dataclass
class LabelConfig:
    """Triple barrier (Lopez de Prado)."""
    horizon: int = 20                # barras hasta la barrera vertical
    tp_sigma: float = 1.0            # take profit en multiplos de sigma escalada
    sl_sigma: float = 0.9            # stop loss en multiplos de sigma escalada
    vol_window: int = 60             # ventana EWMA para sigma
    scale_by_horizon: bool = True    # sigma_barra * sqrt(horizonte)
    # Suelo absoluto de la barrera. DEBE superar con holgura el coste de ida y
    # vuelta (~12 bps en futuros VIP0); si no, el edge no cabe en la barrera.
    min_barrier_bps: float = 20.0


@dataclass
class ModelConfig:
    n_splits: int = 6                # folds de walk-forward purgado
    embargo: int = 30                # barras de embargo entre train y test
    min_train_bars: int = 5000
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = 6
    n_estimators: int = 400
    min_child_samples: int = 80
    subsample: float = 0.8
    colsample_bytree: float = 0.7
    reg_lambda: float = 5.0
    seed: int = 42
    max_features: int = 60           # seleccion por importancia


@dataclass
class StrategyConfig:
    min_edge_bps: float = 2.0        # EV neto minimo para abrir posicion
    min_prob: float = 0.40           # probabilidad minima de la clase direccional
    kelly_fraction: float = 0.25     # fraccion de Kelly (nunca uses 1.0)
    max_position_pct: float = 0.20   # % del equity por trade
    allow_short: bool = True
    cooldown_bars: int = 2           # barras de espera tras cerrar
    exit_on_flip: bool = True        # cerrar si la senal se invierte


@dataclass
class RiskConfig:
    initial_equity: float = 10_000.0
    max_daily_loss_pct: float = 2.0
    max_trades_per_day: int = 120
    max_consecutive_losses: int = 6
    leverage: float = 1.0


@dataclass
class LiveConfig:
    testnet: bool = True
    poll_seconds: int = 5
    dry_run: bool = True             # True = no envia ordenes reales
    api_key_env: str = "BINANCE_API_KEY"
    api_secret_env: str = "BINANCE_API_SECRET"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    @property
    def data_path(self) -> Path:
        return Path(self.data.data_dir)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {
    "data": DataConfig,
    "costs": CostConfig,
    "labels": LabelConfig,
    "model": ModelConfig,
    "strategy": StrategyConfig,
    "risk": RiskConfig,
    "live": LiveConfig,
}


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Carga config.yaml. Claves ausentes usan el valor por defecto."""
    cfg = Config()
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        return cfg
    raw = yaml.safe_load(p.read_text()) or {}
    for name, klass in _SECTIONS.items():
        section = raw.get(name)
        if not isinstance(section, dict):
            continue
        valid = {f for f in klass.__dataclass_fields__}
        unknown = set(section) - valid
        if unknown:
            raise ValueError(f"config.yaml: claves desconocidas en '{name}': {sorted(unknown)}")
        setattr(cfg, name, klass(**{k: v for k, v in section.items() if k in valid}))
    return cfg
