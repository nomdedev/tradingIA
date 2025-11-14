# 🚀 BTC Trading Platform - Sistema de Configuración Avanzada

## ✅ Estado del Proyecto

### Funcionalidades Implementadas

#### 1. **Sistema de Configuración Avanzada** (`src/advanced_config_manager.py`)
- ✅ **Gestión Multi-API**: Soporte para 4 proveedores (Alpaca, Binance, Coinbase, Polygon)
- ✅ **Almacenamiento Seguro**: Encriptación de credenciales con Fernet (AES)
- ✅ **Sistema de Presets**: Guardar/cargar/comparar configuraciones de estrategias
- ✅ **Gestión de APIs**: Activar/desactivar, configurar credenciales por API
- ✅ **Validación Completa**: Verificación de configuraciones antes de guardar

#### 2. **Integración con Agentes IA** (`src/ai_agent_integrator.py`)
- ✅ **4 Agentes Disponibles**: Copilot, Claude, ChatGPT, Custom
- ✅ **Análisis Automatizado**:
  - Resultados de backtest
  - Código de estrategias
  - Comparación entre estrategias
  - Sugerencias de optimización de parámetros
  - Validación de modelos matemáticos
- ✅ **Configuración Flexible**: Triggers automáticos y umbrales personalizables

#### 3. **Configuración del Sistema** (`config/app_config.json`)
```json
{
  "app_settings": {
    "name": "BTC Trading Platform",
    "version": "2.0.0",
    "default_api": "alpaca"
  },
  "api_connections": {
    "alpaca": { "active": true, "default": true },
    "binance": { "active": false },
    "coinbase": { "active": false },
    "polygon": { "active": false }
  },
  "ai_integration": {
    "copilot": { "enabled": false },
    "claude": { "enabled": false },
    "chatgpt": { "enabled": false },
    "custom": { "enabled": false }
  },
  "security_settings": {
    "encryption_enabled": true,
    "auto_logout_minutes": 60,
    "require_2fa": false
  },
  "testing_settings": {
    "run_unit_tests": true,
    "coverage_threshold": 80,
    "enable_integration_tests": true
  },
  "performance_settings": {
    "enable_multiprocessing": true,
    "max_workers": 4,
    "memory_limit_mb": 4096
  }
}
```

## 🎯 Características Principales

### Gestión de Estrategias
1. **Carga Dinámica**: Importar estrategias personalizadas desde cualquier módulo
2. **Presets**: Sistema completo de presets para guardar configuraciones favoritas
3. **Comparación**: Herramientas para comparar rendimiento entre estrategias
4. **Parámetros Flexibles**: Ajustar cualquier parámetro de estrategia en tiempo real

### APIs Soportadas
- **Alpaca**: Trading de acciones y criptomonedas (DEFAULT)
- **Binance**: Exchange de criptomonedas
- **Coinbase**: Exchange de criptomonedas
- **Polygon**: Datos de mercado en tiempo real

### Agentes IA
- **GitHub Copilot**: Análisis de código y sugerencias
- **Claude (Anthropic)**: Análisis profundo de estrategias
- **ChatGPT (OpenAI)**: Optimización y validación
- **Custom**: API personalizada para modelos propios

## 📋 Cómo Usar

### Opción 1: Lanzador Rápido (RECOMENDADO)
```powershell
# Desde el escritorio o cualquier ubicación:
.\LANZAR_PLATAFORMA.bat
```

Este archivo `.bat` puede copiarse al escritorio para acceso rápido.

### Opción 2: Ejecutar Directamente
```powershell
# Activar entorno Python (si usas virtualenv)
# Luego ejecutar:
python demo_advanced_config.py
```

### Opción 3: Desde IDE
Abrir `demo_advanced_config.py` en VS Code o tu IDE favorito y ejecutar.

## 🔧 Configuración Inicial

### 1. Configurar API de Trading (Ejemplo: Alpaca)
```python
from src.advanced_config_manager import AdvancedConfigManager

manager = AdvancedConfigManager()

# Establecer credenciales de Alpaca
manager.set_api_credentials(
    api_name='alpaca',
    api_key='TU_API_KEY',
    api_secret='TU_API_SECRET',
    base_url='https://paper-api.alpaca.markets'  # Para paper trading
)

# Activar API
manager.set_active_api('alpaca')
manager.save_config()
```

### 2. Configurar Agente IA (Opcional)
```python
# Configurar Claude
manager.configure_agent(
    agent_name='claude',
    api_key='TU_CLAUDE_API_KEY',
    model='claude-3-sonnet-20240229',
    enabled=True
)

# Activar agente
manager.set_active_agent('claude')
manager.save_config()
```

### 3. Guardar Preset de Estrategia
```python
preset_config = {
    'strategy_name': 'MACD_ADX',
    'parameters': {
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'adx_period': 14,
        'adx_threshold': 25
    },
    'risk_params': {
        'max_position_size': 0.1,
        'stop_loss': 0.02,
        'take_profit': 0.03
    }
}

manager.save_strategy_preset(
    preset_name='Aggressive_Momentum',
    config=preset_config,
    description='Configuración agresiva para tendencias fuertes'
)
```

### 4. Cargar y Usar Preset
```python
# Listar presets disponibles
presets = manager.list_strategy_presets()
print(f"Presets disponibles: {presets}")

# Cargar preset específico
config = manager.load_strategy_preset('Aggressive_Momentum')
print(f"Configuración cargada: {config}")
```

## 📊 Demo Completo

El archivo `demo_advanced_config.py` incluye 9 secciones de demostración:

1. **Gestión de APIs**: Listar y configurar APIs
2. **Presets de Estrategias**: Guardar/cargar configuraciones
3. **Integración IA**: Configurar agentes de análisis
4. **Configuración de Seguridad**: Encriptación y autenticación
5. **Settings de Testing**: Pruebas automáticas
6. **Configuración de Performance**: Optimización de recursos
7. **Workflow Completo**: Ejemplo end-to-end
8. **Validación**: Verificar configuración
9. **Resumen**: Estado actual del sistema

## 🔐 Seguridad

- **Encriptación**: Credenciales encriptadas con Fernet (AES-256)
- **Variables de Entorno**: Soporte para `.env` files
- **Auto-logout**: Sesiones expiran automáticamente
- **2FA**: Soporte para autenticación de dos factores (configurable)

## 📦 Dependencias

```txt
cryptography>=41.0.0   # Encriptación de credenciales
requests>=2.31.0        # Llamadas a APIs externas
python-dotenv>=1.0.0    # Variables de entorno
pyyaml>=6.0.1           # Configuración YAML
```

Instalar con:
```powershell
pip install -r requirements_platform.txt
```

## ⚠️ Notas Importantes

### Ejecutable (.exe)
El intento de crear un ejecutable con PyInstaller encontró conflictos con:
- **PyTorch**: Problemas de DLL loading (access violations)
- **PyQt6**: Issues con binarios de Qt
- **xml/plistlib**: Módulos de la biblioteca estándar no incluidos correctamente

**Recomendación**: Usar el lanzador `.bat` en lugar del ejecutable compilado. Funciona perfectamente y es más flexible.

### Archivos de Configuración
- `config/app_config.json`: Configuración principal del sistema
- `config/strategies_registry.json`: Registro de presets de estrategias
- `.env`: Variables de entorno sensibles (NO compartir)

### Variables de Entorno (.env)
Crear un archivo `.env` en la raíz del proyecto:
```ini
ALPACA_API_KEY=tu_api_key
ALPACA_API_SECRET=tu_api_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

CLAUDE_API_KEY=tu_claude_key
OPENAI_API_KEY=tu_openai_key
```

## 🎓 Ejemplos de Uso

### Ejemplo 1: Backtesting con Preset
```python
# 1. Cargar preset
config = manager.load_strategy_preset('Aggressive_Momentum')

# 2. Ejecutar backtest
# (código de backtesting existente)

# 3. Analizar con IA
if manager.get_active_agent():
    analysis = ai_integrator.analyze_backtest_results(results, strategy_name='MACD_ADX')
    print(f"Análisis IA: {analysis}")
```

### Ejemplo 2: Comparar Estrategias
```python
strategy_codes = {
    'MACD': open('strategies/macd_strategy.py').read(),
    'RSI': open('strategies/rsi_strategy.py').read()
}

comparison = ai_integrator.compare_strategies(
    strategy_codes=strategy_codes,
    comparison_criteria=['performance', 'risk', 'complexity']
)
```

## 📈 Próximos Pasos

1. ✅ **Sistema de configuración completo** - IMPLEMENTADO
2. ✅ **Integración multi-API** - IMPLEMENTADO
3. ✅ **Agentes IA para análisis** - IMPLEMENTADO
4. ⏳ **Dashboard visual** - PENDIENTE
5. ⏳ **Live trading con monitoring** - PENDIENTE
6. ⏳ **Sistema de alertas** - PENDIENTE

## 🐛 Troubleshooting

### Error: "No module named 'cryptography'"
```powershell
pip install cryptography
```

### Error: "Failed to decrypt credentials"
Las credenciales en `.env` no están encriptadas por defecto. La encriptación solo se aplica a las guardadas en `app_config.json`.

### Error: "API credentials not found"
Configurar credenciales usando `set_api_credentials()` antes de usar la API.

## 📞 Soporte

Para problemas o dudas:
1. Revisar este README
2. Consultar `demo_advanced_config.py` para ejemplos
3. Verificar logs en `logs/`

## 📄 Licencia

Este proyecto es propietario. Todos los derechos reservados.

---

**Versión**: 2.0.0  
**Última actualización**: 2024  
**Autor**: Sistema de Trading Automatizado
