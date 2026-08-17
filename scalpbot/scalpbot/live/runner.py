"""Bucle de trading en vivo (papel o Binance).

Diseno:
  1. Al arrancar se carga un buffer largo de velas cerradas para que los
     indicadores de ventana larga tengan historia suficiente.
  2. Cada ciclo se anaden las velas nuevas ya CERRADAS. Nunca se opera sobre la
     vela en curso: sus valores cambian hasta el ultimo segundo.
  3. Se recalculan features, se predice, se aplica la politica y el gestor de
     riesgo, y se ejecuta.
  4. TP/SL se vigilan localmente y, en modo Binance futuros, tambien se envian
     al exchange como ordenes reduce-only por si el proceso muere.
"""
from __future__ import annotations

import signal
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..data.binance import BinanceREST
from ..data.loader import _fetch_context
from ..features.builder import build_features
from ..labeling import barrier_fracs
from ..model.registry import ModelBundle
from ..strategy.policy import decide_one
from ..strategy.risk import RiskManager
from ..utils import get_logger

log = get_logger("live")


@dataclass
class LiveState:
    side: int = 0
    qty: float = 0.0
    entry_px: float = 0.0
    tp_px: float = 0.0
    sl_px: float = 0.0
    entry_time: pd.Timestamp | None = None
    bars_held: int = 0


class LiveRunner:
    def __init__(self, cfg: Config, bundle: ModelBundle, paper: bool = True,
                 buffer_bars: int = 6000, max_cycles: int | None = None):
        self.cfg = cfg
        self.bundle = bundle
        self.paper = paper
        self.buffer_bars = buffer_bars
        self.max_cycles = max_cycles
        self.client = BinanceREST(market=cfg.data.market)
        self.risk = RiskManager(cfg.risk)
        self.state = LiveState()
        self.buffer: pd.DataFrame = pd.DataFrame()
        self.context: pd.DataFrame = pd.DataFrame()
        self._last_ctx_refresh = 0.0
        self._stop = False
        self.trade_log: list[dict] = []

        self.broker = None
        if not paper:
            from .broker import BinanceBroker
            self.broker = BinanceBroker(
                symbol=cfg.data.symbol, market=cfg.data.market,
                testnet=cfg.live.testnet, dry_run=cfg.live.dry_run,
                key_env=cfg.live.api_key_env, secret_env=cfg.live.api_secret_env)
            self.filters = self.client.exchange_filters(cfg.data.symbol)
            log.info("filtros del simbolo: %s", self.filters)

        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

    def _handle_stop(self, *_args) -> None:
        log.warning("senal de parada recibida; cerrando de forma ordenada")
        self._stop = True

    # ------------------------------------------------------------------ datos

    def warmup(self) -> None:
        d = self.cfg.data
        end_ms = int(time.time() * 1000)
        from ..utils import INTERVAL_MS
        start_ms = end_ms - self.buffer_bars * INTERVAL_MS[d.interval]
        log.info("cargando buffer de %d velas...", self.buffer_bars)
        self.buffer = self.client.klines(d.symbol, d.interval, start_ms, end_ms)
        self.buffer = self.buffer[self.buffer["close_time"] <= end_ms]
        log.info("buffer listo: %d velas (%s -> %s)",
                 len(self.buffer), self.buffer.index[0], self.buffer.index[-1])
        self._refresh_context(force=True)

    def _refresh_context(self, force: bool = False) -> None:
        if not self.cfg.data.use_context:
            return
        if not force and time.time() - self._last_ctx_refresh < 300:
            return
        try:
            end_ms = int(time.time() * 1000)
            start_ms = int(self.buffer.index[0].timestamp() * 1000)
            self.context = _fetch_context(self.client, self.cfg.data.symbol,
                                          start_ms, end_ms, self.buffer.index,
                                          self.cfg.data.days)
            self._last_ctx_refresh = time.time()
        except Exception as e:  # noqa: BLE001 - el contexto es opcional en vivo
            log.warning("no se pudo refrescar el contexto (%s)", e)

    def _poll_new_bars(self) -> int:
        d = self.cfg.data
        fresh = self.client.recent_klines(d.symbol, d.interval, limit=200)
        if fresh.empty:
            return 0
        new = fresh[fresh.index > self.buffer.index[-1]]
        if new.empty:
            return 0
        self.buffer = pd.concat([self.buffer, new])
        self.buffer = self.buffer[~self.buffer.index.duplicated(keep="last")].sort_index()
        if len(self.buffer) > self.buffer_bars * 2:
            self.buffer = self.buffer.iloc[-self.buffer_bars:]
        return len(new)

    # ----------------------------------------------------------------- modelo

    def _predict_last(self) -> tuple[pd.Timestamp, np.ndarray, float, float, float]:
        """Devuelve (ts, [p_dn,p_flat,p_up], precio_cierre, tp_bps, sl_bps)."""
        raw = self.buffer
        if not self.context.empty:
            ctx = self.context.reindex(raw.index).ffill()
            raw = raw.join(ctx, how="left", rsuffix="_ctx")

        feats = build_features(raw, warmup=0)
        row = feats.iloc[[-1]]
        X = row.reindex(columns=self.bundle.features)

        proba = self.bundle.model.predict_proba(X)[0]
        full = np.zeros(3)
        for col, cls in enumerate(self.bundle.model.classes_):
            full[int(cls)] = proba[col]

        # Exactamente el mismo calculo de barreras que en el entrenamiento.
        up, dn = barrier_fracs(raw["close"], self.cfg.labels)
        return (row.index[-1], full, float(row["close"].iloc[-1]),
                float(up.iloc[-1]) * 1e4, float(dn.iloc[-1]) * 1e4)

    # --------------------------------------------------------------- ejecucion

    def _check_exit(self, price: float, high: float, low: float,
                    ts: pd.Timestamp) -> str | None:
        s = self.state
        if s.side == 0:
            return None
        s.bars_held += 1
        hit_tp = high >= s.tp_px if s.side > 0 else low <= s.tp_px
        hit_sl = low <= s.sl_px if s.side > 0 else high >= s.sl_px
        if hit_sl:
            return "stop_loss"
        if hit_tp:
            return "take_profit"
        if s.bars_held >= self.cfg.labels.horizon:
            return "timeout"
        return None

    def _close(self, price: float, reason: str, ts: pd.Timestamp) -> None:
        s = self.state
        if s.side == 0:
            return
        slip = self.cfg.costs.slippage_bps * 1e-4
        fill_px = price * (1 - s.side * slip)
        fee_rate = self.cfg.costs.taker_fee_bps * 1e-4
        notional_in = s.entry_px * s.qty
        notional_out = fill_px * s.qty
        pnl = s.side * (fill_px - s.entry_px) * s.qty - (notional_in + notional_out) * fee_rate

        if self.broker is not None:
            self.broker.cancel_all()
            self.broker.market_order(-s.side, s.qty, reduce_only=True)

        self.risk.register_close(pnl)
        self.trade_log.append({
            "entry_ts": s.entry_time, "exit_ts": ts, "side": s.side,
            "entry_px": s.entry_px, "exit_px": fill_px, "qty": s.qty,
            "pnl": pnl, "reason": reason, "equity": self.risk.state.equity,
        })
        log.info("CIERRE %s | %s | entrada %.2f salida %.2f | pnl %+.4f | equity %.2f",
                 "LONG" if s.side > 0 else "SHORT", reason, s.entry_px, fill_px,
                 pnl, self.risk.state.equity)
        self.state = LiveState()

    def _open(self, side: int, size_frac: float, price: float, tp_bps: float,
              sl_bps: float, ts: pd.Timestamp, edge: float) -> None:
        ok, reason = self.risk.can_open()
        if not ok:
            log.warning("apertura bloqueada: %s", reason)
            return
        notional = self.risk.notional(size_frac)
        if notional <= 0:
            return
        slip = self.cfg.costs.slippage_bps * 1e-4
        fill_px = price * (1 + side * slip)
        qty = notional / fill_px

        if self.broker is not None:
            from .broker import round_step
            qty = round_step(qty, self.filters.get("step_size", 0.001))
            if qty <= 0 or qty * fill_px < self.filters.get("min_notional", 0):
                log.warning("nocional %.2f por debajo del minimo del exchange", qty * fill_px)
                return
            self.broker.market_order(side, qty)

        self.state = LiveState(
            side=side, qty=qty, entry_px=fill_px,
            tp_px=fill_px * (1 + side * tp_bps * 1e-4),
            sl_px=fill_px * (1 - side * sl_bps * 1e-4),
            entry_time=ts, bars_held=0,
        )
        if self.broker is not None:
            try:
                self.broker.stop_orders(side, qty, self.state.tp_px, self.state.sl_px,
                                        self.filters.get("tick_size", 0.1))
            except Exception as e:  # noqa: BLE001
                log.error("no se pudieron colocar TP/SL en el exchange: %s", e)

        self.risk.register_open()
        log.info("ENTRADA %s | %.6f @ %.2f | tp %.2f sl %.2f | edge %+.2f bps",
                 "LONG" if side > 0 else "SHORT", qty, fill_px,
                 self.state.tp_px, self.state.sl_px, edge)

    # ------------------------------------------------------------------ bucle

    def run(self) -> pd.DataFrame:
        self.warmup()
        mode = "PAPEL" if self.paper else (
            "BINANCE DRY-RUN" if self.cfg.live.dry_run else
            f"BINANCE {'TESTNET' if self.cfg.live.testnet else 'REAL'}")
        log.info("=== modo %s | %s %s | equity inicial %.2f ===",
                 mode, self.cfg.data.symbol, self.cfg.data.interval,
                 self.risk.state.equity)

        cycles = 0
        last_bar = self.buffer.index[-1]
        while not self._stop:
            cycles += 1
            if self.max_cycles is not None and cycles > self.max_cycles:
                log.info("max_cycles alcanzado")
                break
            try:
                n_new = self._poll_new_bars()
                if n_new == 0:
                    time.sleep(self.cfg.live.poll_seconds)
                    continue
                self._refresh_context()

                bar = self.buffer.iloc[-1]
                ts = self.buffer.index[-1]
                if ts == last_bar:
                    continue
                last_bar = ts
                self.risk.roll_day(ts)

                reason = self._check_exit(float(bar["close"]), float(bar["high"]),
                                          float(bar["low"]), ts)
                if reason:
                    self._close(float(bar["close"]), reason, ts)

                ts_p, proba, price, tp_bps, sl_bps = self._predict_last()
                d = decide_one(proba[0], proba[1], proba[2], tp_bps, sl_bps,
                               self.cfg.strategy, self.cfg.costs)
                log.info("%s | close %.2f | p_dn %.3f p_flat %.3f p_up %.3f | "
                         "edge %+.2f bps | side %+d",
                         ts_p, price, proba[0], proba[1], proba[2], d.edge_bps, d.side)

                if self.state.side != 0 and d.side != 0 and d.side != self.state.side \
                        and self.cfg.strategy.exit_on_flip:
                    self._close(price, "flip", ts)
                if self.state.side == 0 and d.side != 0:
                    self._open(d.side, d.size_frac, price, tp_bps, sl_bps, ts, d.edge_bps)

            except KeyboardInterrupt:
                break
            except Exception as e:  # noqa: BLE001 - el bucle no debe morir por un fallo puntual
                log.exception("error en el ciclo: %s", e)
                time.sleep(min(60, self.cfg.live.poll_seconds * 4))

        if self.state.side != 0:
            self._close(float(self.buffer["close"].iloc[-1]), "shutdown",
                        self.buffer.index[-1])
        log.info("=== fin | equity %.2f | %d operaciones ===",
                 self.risk.state.equity, len(self.trade_log))
        return pd.DataFrame(self.trade_log)
