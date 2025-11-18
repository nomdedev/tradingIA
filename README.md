# TradingIA - Advanced Trading Platform

Sistema completo de trading algorítmico con backtesting robusto, paper trading en vivo y análisis avanzado de estrategias.

## 🚀 Inicio Rápido

### 1. Clona el repositorio
```bash
git clone https://github.com/nomdedev/tradingIA.git
cd tradingIA
```

### 2. Configura el entorno
```bash
# Crea entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Instala dependencias
pip install -r requirements_new.txt
```

### 3. Configura credenciales
```bash
cp .env.template .env
# Edita .env con tus credenciales de Alpaca
```

### 4. Ejecuta la aplicación
```bash
python main.py --mode gui
```

## 📊 Características Principales

### 🤖 Backtesting Avanzado
- **Walk-Forward Optimization**: Validación out-of-sample
- **Monte Carlo Simulation**: Análisis de distribución
- **Multi-Timeframe Analysis**: 5m, 15m, 1h, 1d
- **Métricas Completas**: Sharpe, Calmar, Sortino, Max DD
- **Kelly Position Sizing**: Dimensionamiento óptimo de posiciones
- **MAE/MFE Risk Tracking**: Maximum Adverse/Favorable Excursions

### 📈 Estrategias Incluidas (5+)
- **Momentum MACD + ADX**: Trading de momentum con filtro de tendencia
- **Pairs Trading Cointegration**: Arbitraje estadístico
- **HFT Momentum VMA**: High-frequency con volume moving average
- **LSTM ML Reversion**: Machine learning para reversión
- **Mean Reversion IBS + BB**: Reversión a la media

### 📊 Risk Metrics Dashboard ⭐ **NUEVO**
- **Métricas en Tiempo Real**: VaR, CVaR, Sharpe, Drawdown
- **MAE/MFE Analysis**: Análisis detallado de excursions por trade
- **6 Visualizaciones Interactivas**: Histogramas, scatter plots, time series
- **Stress Testing Automatizado**: 5 escenarios de crisis de mercado
- **Sistema de Alertas**: Notificaciones automáticas de riesgo
- **Reportes Automatizados**: Análisis completo con recomendaciones

### 🔔 Sistema de Alertas y Notificaciones
- **Alertas en Tiempo Real**: Monitoreo continuo de estrategias y sistema
- **Múltiples Canales**: GUI, sonido, email y logging
- **Reglas Configurables**: Umbrales personalizables por severidad
- **Historial Completo**: Registro y análisis de eventos
- **Auto-Acusado**: Gestión automática de notificaciones

### 🎯 Paper Trading en Vivo
- **Integración Alpaca**: Trading sin riesgo financiero
- **Monitoreo 24/7**: Dashboard en tiempo real
- **Risk Controls**: Stop-loss y límites automáticos

### 🔧 Arquitectura Modular
```
tradingIA/
├── src/                 # Código fuente
│   ├── data_fetcher.py # Obtención datos
│   ├── indicators.py   # Indicadores técnicos
│   ├── backtester.py   # Motor backtesting
│   └── main_platform.py # GUI principal
├── config/             # Configuración
├── tests/              # Suite de tests
└── results/            # Resultados
```

## 📊 Risk Metrics Dashboard

La **nueva pestaña Tab 11** proporciona análisis profesional de riesgo:

### Métricas en Tiempo Real
- **VaR/CVaR**: Value at Risk y Conditional VaR
- **Sharpe/Sortino/Calmar**: Ratios de riesgo ajustado
- **MAE/MFE Tracking**: Maximum Adverse/Favorable Excursions
- **Drawdown Analysis**: Análisis de caídas máximas

### Visualizaciones Interactivas
- 📈 **MAE/MFE Distribution**: Histogramas de riesgo por trade
- 📉 **Drawdown Analysis**: Evolución temporal de drawdowns
- 📊 **Volatility Clustering**: Análisis de agrupamiento de volatilidad
- ⚡ **Stress Test Scenarios**: Impacto de escenarios extremos
- 🎯 **Risk-Return Scatter**: Perfil riesgo vs retorno
- 🎲 **Tail Risk Analysis**: Análisis de riesgo de cola

### Stress Testing Automatizado
- **Market Crash (-20%)**: Escenarios de crisis severas
- **Flash Crash (-10%)**: Caídas rápidas e inesperadas
- **Volatility Spike (+50%)**: Incrementos extremos de volatilidad
- **Liquidity Crisis**: Escenarios de baja liquidez
- **Interest Rate Changes**: Cambios en tasas de interés

### Sistema de Alertas
- 🚨 **Drawdown Alerts**: Alertas por caídas excesivas
- 📊 **Volatility Alerts**: Notificaciones de volatilidad extrema
- 🎯 **MAE Alerts**: Alertas por riesgo excesivo en trades
- 📧 **Automated Reports**: Reportes diarios/semanales/mensuales

## 🛠️ Desarrollo

### Ejecutar Tests
```bash
pytest tests/ -v
```

### Generar Reporte de Cobertura
```bash
pytest --cov=src --cov-report=html
```

### Construir Ejecutable
```bash
python scripts/build_executable.py
```

## 📚 Documentación

- [Guía de Usuario](docs/GUIA_USUARIO_COMPLETA.md)
- [Sistema de Alertas](docs/ALERTS_SYSTEM_GUIDE.md)
- [Optimización Genética](docs/OPTIMIZATION_GUIDE.md)
- [Risk Metrics Dashboard](docs/TAB11_RISK_METRICS_DASHBOARD.md) ⭐ **NUEVO**
- [Sistema Completo - Guía Técnica](docs/SISTEMA_COMPLETO_GUIA_TECNICA.md) ⭐ **NUEVO**
- [Arquitectura](docs/full_project_docs.md)
- [API Reference](docs/README_PLATFORM.md)

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## ⚠️ Descargo de Responsabilidad

Este software es para fines educativos e investigación. No garantizamos rentabilidad. El trading conlleva riesgos financieros significativos. Usa bajo tu propio riesgo.

## 📞 Soporte

- [Issues](https://github.com/nomdedev/tradingIA/issues)
- [Discussions](https://github.com/nomdedev/tradingIA/discussions)

---

**Versión**: 1.0.0
**Última actualización**: 14 Noviembre 2025</content>
<parameter name="filePath">d:\martin\Proyectos\tradingIA\README_NEW.md