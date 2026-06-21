# VP IFVG EMA Strategy V2 - Backtest Results

**Fecha:** 2025-11-15 11:00:02

**Datos:** 5000 barras BTC 15min

## Resultados Generales

- Capital inicial: $10,000.00
- Capital final: $14,723.70
- Retorno total: 47.24%
- Máx drawdown: -10.99%
- Trades totales: 64
- Win rate: 28.12%
- Profit factor: 1.13
- Expectancy: $0.4212
- Sharpe ratio: 0.86

## Mejoras Implementadas

- Gestión completa de posiciones con TradePosition class
- Stops dinámicos basados en ATR (2x SL, 4x TP)
- Risk management (2% capital por trade, 6% diario)
- Patrón mejorado: IFVG + Volumen alto (1.5x) + EMA cross (9/21) bullish
- Trailing stops activados en profit > 1.5x ATR
