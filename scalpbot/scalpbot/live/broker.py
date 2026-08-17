"""Brokers: papel (simulado) y Binance (testnet o real).

El broker de Binance usa REST firmado con HMAC-SHA256. Empieza SIEMPRE con
testnet=True y dry_run=True.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import requests

from ..utils import get_logger

log = get_logger("live.broker")

TESTNET_FUTURES = "https://testnet.binancefuture.com"
TESTNET_SPOT = "https://testnet.binance.vision"
LIVE_FUTURES = "https://fapi.binance.com"
LIVE_SPOT = "https://api.binance.com"


@dataclass
class Position:
    side: int = 0
    qty: float = 0.0
    entry_px: float = 0.0
    tp_px: float = 0.0
    sl_px: float = 0.0
    opened_at: float = 0.0
    entry_bar: int = 0


@dataclass
class PaperBroker:
    """Simula ejecucion aplicando comision y deslizamiento. Sin riesgo real."""
    equity: float
    taker_fee_bps: float = 5.0
    slippage_bps: float = 1.0
    position: Position = field(default_factory=Position)
    fills: list[dict[str, Any]] = field(default_factory=list)

    def market_order(self, side: int, qty: float, ref_px: float, tag: str = "") -> dict:
        slip = self.slippage_bps * 1e-4
        px = ref_px * (1 + side * slip)
        notional = px * qty
        fee = notional * self.taker_fee_bps * 1e-4
        self.equity -= fee
        fill = {"ts": time.time(), "side": side, "qty": qty, "price": px,
                "fee": fee, "tag": tag, "mode": "paper"}
        self.fills.append(fill)
        log.info("[PAPEL] %s %.6f @ %.2f fee=%.4f (%s)",
                 "BUY" if side > 0 else "SELL", qty, px, fee, tag)
        return fill

    def close(self, ref_px: float, reason: str) -> float:
        p = self.position
        if p.side == 0:
            return 0.0
        fill = self.market_order(-p.side, p.qty, ref_px, tag=f"close:{reason}")
        pnl = p.side * (fill["price"] - p.entry_px) * p.qty
        self.equity += pnl
        log.info("[PAPEL] cierre %s pnl=%.4f equity=%.2f", reason, pnl, self.equity)
        self.position = Position()
        return pnl


class BinanceBroker:
    """Ordenes reales via REST firmado. Requiere API key/secret en el entorno."""

    def __init__(self, symbol: str, market: str = "futures", testnet: bool = True,
                 dry_run: bool = True, key_env: str = "BINANCE_API_KEY",
                 secret_env: str = "BINANCE_API_SECRET", recv_window: int = 5000):
        self.symbol = symbol
        self.market = market
        self.dry_run = dry_run
        self.recv_window = recv_window
        if market == "futures":
            self.base = TESTNET_FUTURES if testnet else LIVE_FUTURES
            self.prefix = "/fapi/v1"
        else:
            self.base = TESTNET_SPOT if testnet else LIVE_SPOT
            self.prefix = "/api/v3"

        self.key = os.environ.get(key_env, "")
        self.secret = os.environ.get(secret_env, "")
        if not dry_run and (not self.key or not self.secret):
            raise RuntimeError(
                f"faltan credenciales: exporta {key_env} y {secret_env}, "
                "o ejecuta con dry_run=True")
        self.session = requests.Session()
        if self.key:
            self.session.headers.update({"X-MBX-APIKEY": self.key})
        log.info("BinanceBroker %s testnet=%s dry_run=%s", market, testnet, dry_run)

    def _sign(self, params: dict[str, Any]) -> str:
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = self.recv_window
        query = urlencode(params)
        sig = hmac.new(self.secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        return f"{query}&signature={sig}"

    def _signed(self, method: str, path: str, params: dict[str, Any]) -> Any:
        if self.dry_run:
            log.info("[DRY-RUN] %s %s %s", method, path, params)
            return {"dry_run": True, **params}
        url = f"{self.base}{path}?{self._sign(dict(params))}"
        r = self.session.request(method, url, timeout=15)
        if r.status_code >= 400:
            raise RuntimeError(f"Binance {r.status_code}: {r.text}")
        return r.json()

    def account_balance(self) -> float:
        if self.dry_run:
            return float("nan")
        if self.market == "futures":
            data = self._signed("GET", "/fapi/v2/balance", {})
            for b in data:
                if b["asset"] == "USDT":
                    return float(b["balance"])
            return 0.0
        data = self._signed("GET", "/api/v3/account", {})
        for b in data["balances"]:
            if b["asset"] == "USDT":
                return float(b["free"])
        return 0.0

    def market_order(self, side: int, qty: float, reduce_only: bool = False) -> dict:
        params: dict[str, Any] = {
            "symbol": self.symbol, "side": "BUY" if side > 0 else "SELL",
            "type": "MARKET", "quantity": f"{qty:.8f}".rstrip("0").rstrip("."),
        }
        if reduce_only and self.market == "futures":
            params["reduceOnly"] = "true"
        return self._signed("POST", f"{self.prefix}/order", params)

    def stop_orders(self, side: int, qty: float, tp_px: float, sl_px: float,
                    tick: float = 0.1) -> list[dict]:
        """Coloca TP y SL como ordenes reduce-only en el exchange.

        Critico: si el bot muere, estas ordenes siguen protegiendo la posicion.
        """
        if self.market != "futures":
            log.warning("stop_orders solo implementado para futuros")
            return []
        close_side = "SELL" if side > 0 else "BUY"
        qty_s = f"{qty:.8f}".rstrip("0").rstrip(".")
        out = []
        for otype, price in (("TAKE_PROFIT_MARKET", tp_px), ("STOP_MARKET", sl_px)):
            out.append(self._signed("POST", f"{self.prefix}/order", {
                "symbol": self.symbol, "side": close_side, "type": otype,
                "stopPrice": _round_tick(price, tick), "quantity": qty_s,
                "reduceOnly": "true", "workingType": "MARK_PRICE",
            }))
        return out

    def cancel_all(self) -> Any:
        path = f"{self.prefix}/allOpenOrders" if self.market == "futures" \
            else f"{self.prefix}/openOrders"
        return self._signed("DELETE", path, {"symbol": self.symbol})


def _round_tick(price: float, tick: float) -> str:
    if tick <= 0:
        return f"{price:.2f}"
    rounded = round(price / tick) * tick
    decimals = max(0, len(f"{tick:.10f}".rstrip("0").split(".")[-1]))
    return f"{rounded:.{decimals}f}"


def round_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return float(int(qty / step) * step)
