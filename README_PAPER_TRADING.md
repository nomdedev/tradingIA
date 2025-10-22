# Paper Trading con Alpaca

Sistema completo de paper trading en vivo con integración de múltiples agentes de IA.

## 🚀 Inicio Rápido

### 1. Instalar dependencias
```bash
pip install -r requirements_live.txt
```

### 2. Configurar API keys
```bash
cp .env.example .env
# Editar .env con tus claves reales de Alpaca
```

### 3. Probar conexión
```bash
python test_alpaca_connection.py
```

### 4. Ejecutar paper trading
```bash
python run_paper_trading.py
```

## 📁 Archivos Creados

- `trading_live/alpaca_client.py` - Cliente Alpaca completo
- `trading_live/live_engine.py` - Motor de trading en vivo
- `trading_live/__init__.py` - Módulo de inicialización
- `run_paper_trading.py` - Script principal
- `.env.example` - Template de configuración
- `requirements_live.txt` - Dependencias adicionales
- `test_alpaca_connection.py` - Script de prueba

## ⚙️ Configuración

### Variables de Entorno (.env)
```env
# Alpaca (requerido)
ALPACA_API_KEY=tu_api_key
ALPACA_SECRET_KEY=tu_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# LLMs (opcional)
GROQ_API_KEY=tu_groq_key
ANTHROPIC_API_KEY=tu_anthropic_key
```

### Opciones de Línea de Comando
```bash
python run_paper_trading.py --help
```

## 🤖 Agentes Soportados

- **RL Agent**: Reinforcement Learning (PPO)
- **GA Agent**: Genetic Algorithm
- **LLM Agent**: Multi-LLM (en desarrollo)

## 🛡️ Características de Seguridad

- Risk management integrado
- Position sizing automático
- Stop losses dinámicos
- Límite de drawdown máximo
- Validación de órdenes

## 📊 Monitoreo

- Logging detallado en `logs/paper_trading.log`
- Métricas en tiempo real
- Reportes de rendimiento
- Historial de operaciones

## ⚠️ Importante

- **PAPER TRADING ONLY**: Este sistema usa la API de paper trading de Alpaca
- **NO USAR DINERO REAL**: Las claves de paper trading no afectan tu cuenta real
- **TESTEAR PRIMERO**: Siempre ejecuta `test_alpaca_connection.py` antes de trading

## 🆘 Troubleshooting

1. **Error de conexión**: Verificar API keys en .env
2. **Mercado cerrado**: El sistema espera automáticamente la apertura
3. **Sin modelos**: Asegurarse de que los modelos RL/GA estén en `models/`
4. **LLM no funciona**: Las LLMs están deshabilitadas por defecto

¡Feliz trading! 📈