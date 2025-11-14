# 🚀 Guía de Uso - Sistema IFVG Trading

## Inicio Rápido

### 1. Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt

# Configurar credenciales de Alpaca
cp .env.template .env
# Editar .env con tus credenciales
```

### 2. Backtesting

```bash
# Backtest simple
python main.py --mode backtest --start 2024-01-01 --end 2024-12-31

# Con capital custom
python main.py --mode backtest --start 2024-01-01 --end 2024-12-31 --capital 50000

# Resultados en:
# - results/backtest_trades_<timestamp>.csv
# - results/backtest_equity_<timestamp>.csv
# - results/backtest_metrics_<timestamp>.json
```

### 3. Dashboard Interactivo

```bash
# Lanzar dashboard
python main.py --mode dashboard

# O directamente
streamlit run src/dashboard.py

# Abrir en navegador: http://localhost:8501
```

**Funcionalidades del Dashboard:**

**📊 Modo: Backtest Results**
- Ver equity curve interactiva
- Analizar drawdown
- Distribución de P&L
- Tabla filtrable de trades
- Métricas de performance

**🤖 Modo: Paper Trading**
- Monitorear trades en vivo
- Ver P&L acumulado
- Tracking de win rate
- Historial de trades

**📈 Modo: Live Analysis**
- Gráfico de velas con señales IFVG
- EMAs superpuestas
- Señales de compra/venta
- Volume bars
- Lista de señales recientes

### 4. Paper Trading

```bash
# Iniciar paper trading
python main.py --mode paper --symbol BTC/USD --capital 10000

# El sistema:
# - Conecta con Alpaca Paper Trading
# - Genera señales cada X minutos
# - Ejecuta órdenes automáticamente
# - Guarda trades en logs/paper_trades.json
```

**Monitoreo:**
```bash
# Ver logs en tiempo real
tail -f logs/paper_trades.json

# O usar el dashboard:
python main.py --mode dashboard
# → Seleccionar "Paper Trading" mode
```

### 5. Optimización de Parámetros

```bash
# Grid Search
python main.py --mode optimize --start 2024-01-01 --end 2024-12-31

# Resultados en:
# - results/grid_search_results.csv
# - results/optimization_results.json
```

**Personalizar Grid Search:**

Edita `src/optimization.py` en la función `run_optimization_example()`:

```python
param_grid = {
    'risk_per_trade': [0.01, 0.015, 0.02],      # 1%, 1.5%, 2%
    'sl_atr_multiplier': [1.0, 1.5, 2.0],       # Stop loss
    'tp_risk_reward': [1.5, 2.0, 2.5],          # Take profit
    'commission': [0.0005, 0.001],               # 0.05%, 0.1%
    'slippage': [0.0001, 0.0005]                # 0.01%, 0.05%
}

optimizer = ParameterOptimizer()
results = optimizer.grid_search(
    param_grid=param_grid,
    start_date='2024-01-01',
    end_date='2024-12-31',
    optimize_metric='sharpe_ratio',  # o 'profit_factor', 'total_return'
    max_workers=4  # Paralelización
)
```

**Walk-Forward Analysis:**

```python
from src.optimization import ParameterOptimizer

optimizer = ParameterOptimizer()

wf_results = optimizer.walk_forward_analysis(
    param_grid=param_grid,
    start_date='2024-01-01',
    end_date='2024-12-31',
    train_period_days=90,   # Ventana de entrenamiento
    test_period_days=30,    # Ventana de validación
    optimize_metric='sharpe_ratio'
)

print(f"Avg Test Sharpe: {wf_results['avg_test_metric']:.3f}")
```

**Monte Carlo Simulation:**

```python
import pandas as pd
from src.optimization import ParameterOptimizer

# Cargar trades históricos
trades_df = pd.read_csv('results/backtest_trades_latest.csv')

optimizer = ParameterOptimizer()
mc_results = optimizer.monte_carlo_simulation(
    trades_df=trades_df,
    n_simulations=1000,
    initial_capital=10000
)

print(f"Probability of Profit: {mc_results['probability_profit']:.1f}%")
print(f"Risk of Ruin: {mc_results['risk_of_ruin']:.1f}%")
```

---

## 📊 Ejemplos de Análisis

### Ver Top 10 Parámetros

```python
import pandas as pd

# Cargar resultados de grid search
results = pd.read_csv('results/grid_search_results.csv')

# Top 10 por Sharpe
top_sharpe = results.nlargest(10, 'sharpe_ratio')
print(top_sharpe[['risk_per_trade', 'sl_atr_multiplier', 'tp_risk_reward', 'sharpe_ratio']])

# Top 10 por Profit Factor
top_pf = results.nlargest(10, 'profit_factor')
print(top_pf[['risk_per_trade', 'sl_atr_multiplier', 'tp_risk_reward', 'profit_factor']])
```

### Análisis de Sensibilidad

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Heatmap de Sharpe vs parámetros
pivot = results.pivot_table(
    values='sharpe_ratio',
    index='sl_atr_multiplier',
    columns='tp_risk_reward',
    aggfunc='mean'
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn')
plt.title('Sharpe Ratio: SL Multiplier vs TP Risk/Reward')
plt.show()
```

### Comparar Trades: Backtest vs Paper

```python
import pandas as pd

# Cargar datos
backtest = pd.read_csv('results/backtest_trades_latest.csv')
paper = pd.read_json('logs/paper_trades.json')

# Comparar win rate
bt_wr = (backtest['pnl'] > 0).mean() * 100
paper_wr = (paper['pnl'] > 0).mean() * 100

print(f"Backtest Win Rate: {bt_wr:.1f}%")
print(f"Paper Win Rate: {paper_wr:.1f}%")
print(f"Difference: {paper_wr - bt_wr:.1f}%")

# Comparar avg P&L
print(f"Backtest Avg P&L: ${backtest['pnl'].mean():.2f}")
print(f"Paper Avg P&L: ${paper['pnl'].mean():.2f}")
```

---

## 🔧 Configuración Avanzada

### Ajustar Parámetros de Trading

Edita `config/config.py`:

```python
BACKTEST_CONFIG = {
    'initial_capital': 10000,
    'risk_per_trade': 0.02,      # 2% riesgo por trade
    'commission': 0.001,          # 0.1% comisión
    'slippage': 0.0005,          # 0.05% slippage
}

IFVG_CONFIG = {
    'min_gap_size': 0.002,       # 0.2% mínimo
    'atr_multiplier': 1.5,       # Filtro ATR
    'lookback_periods': 20,      # Mitigación lookback
}
```

### Custom Indicators

```python
from src.indicators import calculate_all_indicators
import pandas as pd

# Cargar datos
df = pd.read_csv('data/BTCUSD_5Min.csv', index_col=0, parse_dates=True)

# Calcular indicadores
df = calculate_all_indicators(df)

# Acceder a señales
signals = df[df['signal'] != 0]
print(f"Total signals: {len(signals)}")
print(signals[['signal', 'confidence', 'Close']])
```

---

## 🐛 Troubleshooting

### Error: "No Alpaca credentials"
```bash
# Verificar .env
cat .env

# Debe contener:
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### Error: "No data available"
```bash
# Verificar fechas
python -c "from src.data_fetcher import DataFetcher; df = DataFetcher(); print(df.get_historical_data('BTCUSD', '5Min', '2024-01-01', '2024-01-31'))"
```

### Dashboard no muestra datos
```bash
# Verificar archivos
ls -la results/
ls -la logs/

# Ejecutar backtest primero
python main.py --mode backtest --start 2024-01-01 --end 2024-12-31
```

---

## 📚 Documentación Adicional

- **RESUMEN_ACTUALIZADO.md** - Estado completo del proyecto
- **CLEANUP_PLAN.md** - Plan de limpieza de archivos antiguos
- **config/config.py** - Todas las configuraciones disponibles
- **tests/** - Ejemplos de uso en tests

---

## 🎯 Mejores Prácticas

1. **Backtesting primero**: Siempre hacer backtest antes de paper trading
2. **Optimización prudente**: Usar walk-forward para evitar overfitting
3. **Monitoreo constante**: Revisar dashboard regularmente
4. **Logs detallados**: Revisar logs/ para debugging
5. **Tests frecuentes**: Correr `pytest tests/` antes de cambios

---

## 💡 Tips

- **Paralelización**: Aumentar `max_workers` en grid_search para optimización más rápida
- **Filtros de señales**: Ajustar `confidence_threshold` en config para calidad de señales
- **Risk management**: Mantener `risk_per_trade` bajo (1-2%) para preservar capital
- **Timeframes**: Probar múltiples timeframes (5Min, 15Min, 1H) para robustez

---

**¿Necesitas ayuda?** Revisa los tests en `tests/` para más ejemplos de uso.
