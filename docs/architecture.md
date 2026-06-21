# Arquitectura del Sistema - TradingIA

## 📋 Visión General

TradingIA es una plataforma de trading algorítmico para BTC/USD con soporte multi-timeframe (MTF), backtesting avanzado y ejecución en vivo a través de Alpaca API.

```
┌─────────────────────────────────────────────────────────────────────┐
│                           TRADING PLATFORM                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │   Dashboard  │  │     API      │  │      CLI / Scripts       │  │
│  │  (Streamlit) │  │  (FastAPI)   │  │   (Python/Terminal)      │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬──────────────┘  │
│         │                 │                      │                  │
│         └─────────────────┼──────────────────────┘                  │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      CORE ENGINE                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │
│  │  │  Strategy   │  │   Council   │  │   Risk Manager      │  │   │
│  │  │   Engine    │◄─┤  (Decision) │◄─┤  (Kill Switch)      │  │   │
│  │  └──────┬──────┘  └─────────────┘  └──────────┬──────────┘  │   │
│  │         │                                      │             │   │
│  │         ▼                                      ▼             │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │                   EXECUTION LAYER                    │    │   │
│  │  │  ┌──────────────┐        ┌──────────────────────┐   │    │   │
│  │  │  │ BacktesterCore│        │    LiveTrader        │   │    │   │
│  │  │  │ (Simulation) │        │ (Alpaca Integration) │   │    │   │
│  │  │  └──────────────┘        └──────────────────────┘   │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                           │                                         │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      DATA LAYER                              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │
│  │  │ DataManager │  │  Indicators │  │  DataValidator      │  │   │
│  │  │(Fetch/Cache)│  │  (IFVG, VP) │  │  (OHLC, NaN check)  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                           │                                         │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   EXTERNAL SERVICES                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │
│  │  │ Alpaca API  │  │   Alerts    │  │    Logging          │  │   │
│  │  │ (Broker)    │  │  (Webhook)  │  │  (File/Console)     │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Estructura de Directorios

```
tradingIA/
├── api/                    # Data fetching (Alpaca)
│   └── data_fetcher.py     # Historical data retrieval
│
├── backtesting/            # Backtesting utilities
│   └── advanced_backtest.py
│
├── config/                 # Configuration files
│   ├── config.py           # Base config (canonical)
│   ├── mtf_config.py       # MTF-specific config
│   ├── strategies_registry.json
│   └── presets/            # Strategy presets
│
├── core/                   # Core business logic
│   ├── backend_core.py     # DataManager, StrategyEngine
│   ├── council.py          # Decision making system
│   ├── constants.py        # Global constants
│   ├── rules_loader.py     # Trading rules
│   │
│   ├── alerts/             # Notification system
│   ├── api/                # FastAPI REST endpoints
│   ├── brokers/            # Broker integrations
│   │   └── alpaca_broker.py
│   ├── data/               # Data processing
│   │   ├── indicators.py   # Technical indicators
│   │   └── realtime_provider.py
│   ├── execution/          # Trade execution
│   │   ├── backtester_core.py  # Simulation engine
│   │   └── live_trader.py      # Live trading
│   ├── risk/               # Risk management
│   │   └── risk_manager.py # Kill switch, position sizing
│   └── strategies/         # Strategy implementations
│
├── dashboard/              # Streamlit dashboard
│   └── app.py
│
├── data/                   # Data storage
│   ├── raw/                # Raw market data
│   ├── processed/          # Processed data
│   ├── cache/              # API cache
│   └── logs/               # Application logs
│
├── docs/                   # Documentation
│   └── architecture.md     # This file
│
├── src/                    # Legacy/utility scripts
│   ├── main_platform.py    # GUI application
│   └── production_monitoring.py
│
├── tests/                  # Test suite
├── utils/                  # Utility modules
│   └── logging_config.py   # Centralized logging
│
├── .env.example            # Environment template
├── .pre-commit-config.yaml # Code quality hooks
├── Dockerfile              # Container build
├── docker-compose.yml      # Container orchestration
└── pyproject.toml          # Project configuration
```

---

## 🔄 Flujo de Datos

### 1. Data Pipeline

```
Alpaca API ──► DataFetcher ──► DataValidator ──► Cache (CSV/Parquet)
                                    │
                                    ▼
                              DataManager
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
                 5Min            15Min            1Hour
              (Entry TF)     (Momentum TF)     (Trend TF)
```

### 2. Signal Generation (MTF)

```
┌─────────────────────────────────────────────────────────────┐
│                    MULTI-TIMEFRAME ANALYSIS                  │
│                                                              │
│   1H (Trend)      ──► EMA200 Filter ──► MUST be aligned     │
│        │                                                     │
│        ▼                                                     │
│   15Min (Momentum) ──► EMA Cross ──► Confirmation           │
│        │                                                     │
│        ▼                                                     │
│   5Min (Entry)    ──► IFVG + VP ──► Entry Signal            │
│                                                              │
│   All aligned? ──► Council Decision ──► Execute/Reject      │
└─────────────────────────────────────────────────────────────┘
```

### 3. Execution Flow

```
Signal ──► Council ──► Risk Check ──► Position Size ──► Order
              │            │              │               │
              ▼            ▼              ▼               ▼
          Consensus    Kill Switch    Kelly Criterion   Broker API
          (Voting)     (Drawdown)    (Optimal f)       (Alpaca)
```

---

## 🧩 Componentes Principales

### BacktesterCore (`core/execution/backtester_core.py`)
- Simulación de trades con costos realistas
- Slippage, comisiones, market impact
- Walk-forward optimization
- Monte Carlo analysis

### LiveTrader (`core/execution/live_trader.py`)
- Conexión a Alpaca API (Paper/Live)
- Rate limiting (200 req/min)
- Retry con exponential backoff
- Kill switch integrado

### Council (`core/council.py`)
- Sistema de decisión multi-agente
- Votación ponderada por confianza
- Threshold configurable

### RiskManager (`core/risk/risk_manager.py`)
- Position sizing (Kelly Criterion)
- Max drawdown limits
- Kill switch automático

### DataValidator (`core/backend_core.py`)
- Validación OHLC
- Detección de gaps
- Filtrado de precios negativos/NaN

---

## 🔐 Seguridad

1. **API Keys**: Variables de entorno (`.env`)
2. **Logging**: Filtro de datos sensibles (`SensitiveDataFilter`)
3. **Rate Limiting**: 200 requests/min para Alpaca
4. **Kill Switch**: Desactivación automática en drawdown

---

## 📊 Configuración

### Prioridad de Configuración

```
1. Environment Variables (.env)        ─── Highest priority
2. User Preferences (user_preferences.json)
3. config/config.py                    ─── Canonical source
4. config/mtf_config.py                ─── MTF extensions
```

### Archivos de Configuración

| Archivo | Propósito |
|---------|-----------|
| `config.py` | Config base (API, trading, backtest) |
| `mtf_config.py` | Configuración multi-timeframe |
| `strategies_registry.json` | Definiciones de estrategias |
| `costs_params.json` | Parámetros de costos |
| `training_config.yaml` | Configuración ML |

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov-report=html

# Run specific test
pytest tests/test_corrupted_data.py -v
```

### Test Categories
- **Unit Tests**: Componentes individuales
- **Integration Tests**: Flujos completos
- **Data Tests**: Edge cases, datos corruptos

---

## 🚀 Deployment

### Development
```bash
python -m dashboard.app
```

### Docker
```bash
docker-compose up -d trading-app
```

### Production (Paper Trading)
```bash
docker-compose --profile production up -d
```

---

## 📈 Métricas Clave

| Métrica | Descripción | Threshold |
|---------|-------------|-----------|
| Sharpe Ratio | Risk-adjusted return | > 1.0 |
| Max Drawdown | Máxima pérdida | < 15% |
| Win Rate | Trades ganadores | > 50% |
| Profit Factor | Gross profit / Gross loss | > 1.5 |

---

## 🔗 Referencias

- [Alpaca API Docs](https://alpaca.markets/docs/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
