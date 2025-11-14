# Plan de Limpieza y Restructuración del Proyecto

## Resumen Ejecutivo

Se ha creado una estructura limpia y modular para el proyecto de backtesting BTC con estrategia IFVG + Volume Profile + EMAs. Los nuevos archivos están listos para usarse.

## ✅ Archivos Nuevos Creados

### Estructura Principal
```
tradingIA/
├── main.py                       ✅ Entry point con CLI
├── requirements_new.txt          ✅ Dependencias consolidadas
├── .env.template                 ✅ Template de credenciales
├── .gitignore_new                ✅ Gitignore completo
├── README_NEW.md                 ✅ README detallado
│
├── src/                          📁 Código principal
│   ├── __init__.py               ✅
│   ├── data_fetcher.py           ✅ Obtención datos Alpaca
│   ├── indicators.py             ✅ IFVG + VP + EMAs
│   ├── backtester.py             ⏳ Pendiente
│   ├── paper_trader.py           ⏳ Pendiente
│   ├── dashboard.py              ⏳ Pendiente
│   └── optimization.py           ⏳ Pendiente
│
├── config/                       📁 Configuración
│   ├── __init__.py               ✅
│   ├── config.py                 ✅ Configuración centralizada
│   └── best_params.json          ⏳ Se genera después
│
├── data/                         📁 Datos cacheados
│   └── .gitkeep                  ✅
│
├── results/                      📁 Resultados
│   └── .gitkeep                  ✅
│
└── tests/                        📁 Tests unitarios
    └── __init__.py               (ya existe)
```

## 📋 Siguiente Paso: Limpieza

### Archivos a ELIMINAR (comandos PowerShell)

```powershell
# 1. Archivos antiguos redundantes
Remove-Item -Recurse -Force agents/
Remove-Item -Recurse -Force backtesting/
Remove-Item -Recurse -Force build/
Remove-Item -Recurse -Force dist/
Remove-Item -Recurse -Force models/checkpoints/
Remove-Item -Recurse -Force scripts/

# 2. GUIs y demos antiguos
Remove-Item ifvg_demo_trading.py
Remove-Item ifvg_live_trading.py
Remove-Item trading_gui.py
Remove-Item trading_gui_advanced.py
Remove-Item trading_gui.spec

# 3. Tests antiguos (migrar lógica útil primero)
Remove-Item test_ifvg_strategy.py
Remove-Item test_ifvg_live_integration.py
Remove-Item test_historical_data.py

# 4. Dashboard antiguo (reemplazar con nuevo)
Remove-Item -Recurse -Force dashboard/clean*

# 5. __pycache__ (todos)
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force

# 6. Archivos temporales
Remove-Item IFVG_README.md  # Ya está en docs/
Remove-Item PROJECT_RESTRUCTURE.md  # Este documento
```

### Archivos a RENOMBRAR

```powershell
# Activar nuevos archivos
Move-Item .gitignore_new .gitignore -Force
Move-Item requirements_new.txt requirements.txt -Force
Move-Item README_NEW.md README.md -Force
```

### Archivos a MANTENER (adaptados)

```powershell
# Mantener pero revisar/adaptar:
# - .env (revisar credenciales)
# - strategies/ifvg_strategy.py (referencia, ya migrado a src/indicators.py)
# - trading_live/alpaca_client.py (referencia, migrado a src/data_fetcher.py)
# - logs/ (mantener directorio pero limpiar logs antiguos)
```

## 🔧 Pasos de Migración

### 1. Verificar Nuevos Archivos
```bash
python main.py --mode backtest --help
python config/config.py  # Test configuration
python src/indicators.py  # Test indicators
```

### 2. Migrar Código Útil

**De `strategies/ifvg_strategy.py`**:
- ✅ Ya migrado a `src/indicators.py`
- Lógica IFVG mejorada
- Volume Profile integrado
- EMAs multi-timeframe

**De `trading_live/alpaca_client.py`**:
- ✅ Ya migrado a `src/data_fetcher.py`
- Métodos de obtención de datos
- Manejo de rate limits
- Cache de datos

**Tests útiles**:
- Migrar asserts a `tests/test_indicators.py`
- Adaptar tests a nueva estructura

### 3. Crear Módulos Faltantes

**Prioridad Alta**:
1. `src/backtester.py` - Motor de backtesting
2. `src/paper_trader.py` - Paper trading en vivo
3. `tests/test_*.py` - Tests unitarios completos

**Prioridad Media**:
4. `src/dashboard.py` - Dashboard Streamlit
5. `src/optimization.py` - Optimización de parámetros

**Prioridad Baja**:
6. `docs/strategy_analysis.md` - Análisis detallado
7. `docs/api_guide.md` - Guía API

## 📦 Testing Post-Limpieza

```bash
# 1. Instalar dependencias limpias
python -m venv .venv_clean
.venv_clean\Scripts\activate
pip install -r requirements.txt

# 2. Validar configuración
python config/config.py

# 3. Test data fetching
python src/data_fetcher.py

# 4. Test indicators
python src/indicators.py

# 5. Test main
python main.py --mode backtest

# 6. Run tests
pytest tests/ -v
```

## 🎯 Beneficios de la Nueva Estructura

### Modularidad
- ✅ Separación clara de responsabilidades
- ✅ Imports limpios sin dependencias circulares
- ✅ Fácil testing y debugging

### Escalabilidad
- ✅ Agregar nuevas estrategias sin modificar core
- ✅ Multi-símbolo y multi-timeframe desde diseño
- ✅ Configuración centralizada y flexible

### Mantenibilidad
- ✅ Código documentado y tipado
- ✅ Logging comprehensivo
- ✅ Tests unitarios >80% coverage (objetivo)

### Profesionalismo
- ✅ Estructura estándar de proyecto Python
- ✅ CLI completo con argparse
- ✅ Configuración via .env
- ✅ README detallado

## ⚠️ Advertencias

1. **Backup antes de eliminar**: 
   ```bash
   # Crear backup del proyecto actual
   Copy-Item -Recurse tradingIA tradingIA_backup_$(Get-Date -Format 'yyyyMMdd')
   ```

2. **Verificar .env**:
   - Asegurar credenciales Alpaca están en `.env`
   - No commitear `.env` a git

3. **Tests graduales**:
   - No eliminar todo de golpe
   - Verificar cada módulo funciona antes de borrar código antiguo

4. **Git commits incrementales**:
   ```bash
   git add src/ config/ main.py requirements.txt
   git commit -m "Add: Clean project structure with IFVG strategy"
   
   git rm -r agents/ backtesting/ build/
   git commit -m "Remove: Old redundant modules"
   ```

## 📊 Estado Actual

| Módulo | Estado | Notas |
|--------|--------|-------|
| config.py | ✅ Completo | Configuración centralizada |
| data_fetcher.py | ✅ Completo | Integración Alpaca OK |
| indicators.py | ✅ Completo | IFVG + VP + EMAs |
| backtester.py | ⏳ Pendiente | Crear siguiente |
| paper_trader.py | ⏳ Pendiente | Después de backtester |
| dashboard.py | ⏳ Pendiente | Streamlit UI |
| optimization.py | ⏳ Pendiente | Grid search |
| Tests | ⏳ Parcial | Migrar y expandir |

## 🚀 Próximos Pasos Inmediatos

1. **Ahora**: Revisar archivos nuevos creados
2. **Luego**: Crear `src/backtester.py`
3. **Después**: Tests completos
4. **Finalmente**: Limpieza de archivos antiguos

---

**Fecha**: 12 de noviembre de 2025  
**Estado**: ✅ Estructura base creada, lista para limpieza
