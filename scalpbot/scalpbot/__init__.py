"""scalpbot: bot de micro-scalping para Binance con machine learning.

Modulos principales:
    data       adquisicion (Binance REST, contexto macro, mercado sintetico)
    features   analisis tecnico, microestructura y contexto
    labeling   triple barrera con barreras escaladas por volatilidad
    model      walk-forward purgado con embargo
    strategy   politica de EV neto de costes y gestion de riesgo
    backtest   motor event-driven con costes y metricas honestas
    live       ejecucion en papel / testnet / mainnet
"""
__version__ = "1.0.0"

__all__ = ["__version__"]
