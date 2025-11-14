# Pine Script Exporter

Este script exporta la estrategia BTC IFVG + Volume Profile + EMAs a Pine Script v5 para TradingView.

## Uso

### Desde parámetros optimizados

```bash
python scripts/pine_exporter.py --params=results/optimization/best_params.json
```

### Con parámetros personalizados

```bash
python scripts/pine_exporter.py --atr=1.5 --vol=1.2 --ema_fast=18 --ema_slow=48
```

### Parámetros disponibles

- `--atr`: Multiplicador ATR (default: 1.5)
- `--vol`: Umbral de volumen (default: 1.2)
- `--ema_fast`: EMA rápida 5m (default: 18)
- `--ema_slow`: EMA lenta 5m (default: 48)
- `--ema_fast_15m`: EMA rápida 15m (default: 18)
- `--ema_slow_15m`: EMA lenta 15m (default: 48)
- `--ema_fast_1h`: EMA rápida 1h (default: 95)
- `--ema_slow_1h`: EMA lenta 1h (default: 200)
- `--vp_rows`: Filas del perfil de volumen (default: 120)
- `--va_percent`: Porcentaje área de valor (default: 0.7)
- `--tp_rr`: Ratio riesgo/recompensa TP (default: 2.2)
- `--output`: Archivo de salida (opcional)

## Características del Pine Script generado

- **IFVG**: Detección de Implied Fair Value Gaps
- **Volume Profile**: Perfil de volumen simplificado con POC
- **EMAs**: Medias móviles exponenciales multi-timeframe
- **Confluence Scoring**: Sistema de puntuación de convergencia (≥4 para señales)
- **Risk Management**: Stop Loss y Take Profit basados en ATR
- **Alerts**: Alertas configurables para señales
- **Strategy Mode**: Modo estrategia para backtesting en Pine

## Archivos generados

Los scripts se guardan en `scripts_pine/` con nombres como:
- `btc_strategy_cli_YYYYMMDD_HHMMSS.pine`
- `btc_strategy_YYYYMMDD_HHMMSS.pine`

## Importar en TradingView

1. Copia el contenido del archivo `.pine` generado
2. Ve a TradingView → Pine Editor
3. Pega el código y guarda
4. Añade el indicador a tu gráfico BTC/USD

## Notas

- El Volume Profile está simplificado para compatibilidad con Pine Script
- Las EMAs multi-timeframe usan aproximaciones
- El sistema de alertas está configurado para señales de alta confianza
- Incluye modo estrategia para backtesting nativo en TradingView