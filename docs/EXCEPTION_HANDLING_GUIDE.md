# Guía de Exception Handling - Auditoría Round 12

## Problema Identificado

El proyecto tiene ~70+ instancias de `except Exception` genérico, lo cual:
- Oculta bugs reales
- Dificulta el debugging
- Captura excepciones que no deberían ser capturadas (KeyboardInterrupt, SystemExit)

## Patrón Correcto

### ❌ Incorrecto
```python
try:
    result = do_something()
except Exception as e:
    logger.error(f"Error: {e}")
    return None
```

### ✅ Correcto
```python
try:
    result = do_something()
except (ValueError, KeyError, TypeError) as e:
    logger.error(f"Error de validación: {e}")
    return None
except FileNotFoundError as e:
    logger.error(f"Archivo no encontrado: {e}")
    raise
```

## Excepciones Comunes por Contexto

### Operaciones de I/O
```python
except (FileNotFoundError, IOError, PermissionError) as e:
```

### Parsing de JSON/YAML
```python
except (json.JSONDecodeError, yaml.YAMLError) as e:
```

### Operaciones numéricas
```python
except (ValueError, ZeroDivisionError, OverflowError) as e:
```

### Acceso a datos (pandas/numpy)
```python
except (KeyError, IndexError, TypeError) as e:
```

### Llamadas a API
```python
except (requests.RequestException, urllib.error.URLError) as e:
```

### Trading específico
```python
except (alpaca_trade_api.rest.APIError, ConnectionError) as e:
```

## Archivos Prioritarios (Core)

| Archivo | Instancias | Prioridad |
|---------|------------|-----------|
| `core/ui/dashboard_controller.py` | 12 | Alta |
| `core/ui/main_window.py` | 6 | Alta |
| `core/tracking/mlflow_tracker.py` | 3 | Media |
| `core/alerts/alert_manager.py` | 1 | Media |
| `core/strategies/strategy_registry.py` | 1 | Baja |

## Archivos Secundarios (src/)

| Archivo | Instancias |
|---------|------------|
| `src/analysis_engines.py` | 1 |
| `src/causality_stress_tests.py` | 3 |
| `src/metrics_validation.py` | 2 |

## Archivos de Tests (Menos Críticos)

Los tests pueden mantener `except Exception` en casos específicos donde:
- Se está probando que algo NO lanza excepción
- Es un test de smoke/integración

## Regla General

1. **Nunca** usar `except:` sin tipo
2. **Evitar** `except Exception` - ser específico
3. **Siempre** loguear la excepción con contexto
4. **Re-lanzar** si no se puede manejar apropiadamente
5. **Documentar** por qué se captura una excepción

## Ejemplo de Refactorización

### Antes (dashboard_controller.py):
```python
try:
    self.run_backtest(params)
except Exception as e:
    self.logger.error(f"Backtest error: {e}")
```

### Después:
```python
try:
    self.run_backtest(params)
except (ValueError, KeyError) as e:
    self.logger.error(f"Parámetros inválidos: {e}")
except FileNotFoundError as e:
    self.logger.error(f"Datos no encontrados: {e}")
except Exception as e:
    # Último recurso - loguear traceback completo
    self.logger.exception(f"Error inesperado en backtest")
    raise RuntimeError("Backtest falló") from e
```

## Progreso

- [x] Documentación creada
- [ ] `core/ui/dashboard_controller.py` - 0/12 corregidos
- [ ] `core/ui/main_window.py` - 0/6 corregidos
- [ ] `core/tracking/mlflow_tracker.py` - 0/3 corregidos

---
*Creado: Auditoría Round 12*
