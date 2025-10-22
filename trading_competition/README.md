# Trading AI Competition: RL vs Genetic Algorithms

Sistema de competición entre agentes de Reinforcement Learning y Algoritmos Genéticos para trading algorítmico.

## 📊 Resultados de la Competición

| Agente | Valor Final | Retorno Total | Número de Trades |
|--------|-------------|---------------|------------------|
| **GA** | **$12,057.13** | **20.57%** | 24 |
| RL | $10,000.00 | 0.00% | 0 |

🏆 **Ganador: Agente GA** (margen: 20.57%)

## 🏗️ Arquitectura del Proyecto

```
trading_competition/
├── agents/
│   ├── rl_agent.py          # Agente RL con PPO
│   └── ga_agent.py          # Agente GA con DEAP
├── data/
│   ├── raw/                 # Datos crudos de SPY
│   └── processed/           # Datos con indicadores técnicos
├── environments/
│   └── trading_env.py       # Entorno Gymnasium personalizado
├── models/                  # Modelos entrenados guardados
├── results/                 # Resultados y visualizaciones
├── strategies/              # Estrategias adicionales
├── tests/                   # Tests del sistema
└── utils/                   # Utilidades
```

## 🚀 Componentes Implementados

### ✅ Completado
- **Entorno de Desarrollo**: Python 3.11.9 con venv y 15+ paquetes
- **Adquisición de Datos**: 1458 días de datos SPY (2020-2025)
- **Indicadores Técnicos**: 30+ indicadores (RSI, MACD, Bollinger Bands, etc.)
- **Agente RL**: PPO con Stable-Baselines3, entrenado 10k timesteps
- **Agente GA**: Evolución genética con DEAP, fitness 89.4%
- **Framework de Competición**: Comparación automática en datos de prueba
- **Visualizaciones**: Gráficos comparativos de rendimiento

### 🔄 Pendiente
- **Backtesting Avanzado**: Integración con backtesting.py
- **Optimización RL**: Mejorar función de recompensa y entrenamiento

## 🧠 Agentes Desarrollados

### Agente RL (PPO)
- **Framework**: Stable-Baselines3
- **Algoritmo**: Proximal Policy Optimization
- **Estado**: 11 indicadores técnicos normalizados
- **Acciones**: Hold/Buy/Sell
- **Recompensa**: Retorno del portfolio + penalizaciones
- **Resultado**: Política conservadora (0 trades en competición)

### Agente GA (DEAP)
- **Framework**: DEAP
- **Cromosoma**: [RSI_overbought, RSI_oversold, MACD_threshold, BB_width]
- **Fitness**: Retorno total del portfolio
- **Evolución**: 30 generaciones, población 50
- **Resultado**: 20.57% retorno, 24 trades optimizados

## 📈 Indicadores Técnicos

- **Momentum**: RSI, Stochastic, MACD, Williams %R
- **Tendencia**: SMA, EMA, ADX, DMP/DMN
- **Volatilidad**: ATR, Bollinger Bands, Volatility 20d
- **Volumen**: OBV, CMF, Volume
- **Retornos**: Returns 1d/5d/20d, Log returns

## 🛠️ Instalación y Uso

```bash
# Clonar y configurar entorno
git clone <repo>
cd trading_competition
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Ejecutar pipeline completo
python create_structure.py
python download_data.py
python indicators.py
python agents/rl_agent.py
python agents/ga_agent.py
python competition.py
```

## 🎯 Conclusiones

1. **GA Superó a RL**: El agente genético encontró una estrategia superior (20.57% vs 0%)
2. **Interpretabilidad**: GA proporciona parámetros claros vs "caja negra" del RL
3. **Eficiencia**: GA converge más rápido que RL en este dominio
4. **Limitaciones RL**: Función de recompensa necesita refinamiento

## 🔬 Próximos Pasos

- Mejorar reward function del RL (Sharpe ratio, drawdown)
- Implementar ensemble de agentes
- Agregar más indicadores y features
- Validación walk-forward y out-of-sample
- Integración con brokers reales (Paper trading)

## 📚 Tecnologías Utilizadas

- **Python 3.11.9**
- **Stable-Baselines3** (RL)
- **DEAP** (GA)
- **Gymnasium** (Entornos RL)
- **TA-Lib** (Indicadores técnicos)
- **Pandas/NumPy** (Procesamiento de datos)
- **Matplotlib/Seaborn** (Visualizaciones)
- **Rich** (CLI mejorada)