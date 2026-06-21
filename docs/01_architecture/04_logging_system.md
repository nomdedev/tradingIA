# Sistema de Logging y Monitoreo de Errores - TradingIA Platform

## 📋 Resumen Ejecutivo

Se ha implementado un **sistema completo de logging y monitoreo** para capturar todos los eventos importantes, errores, y cambios en la UI de la plataforma TradingIA. Este sistema permite analizar exhaustivamente cualquier problema o comportamiento inesperado.

---

## 🎯 Problemas Resueltos

### 1. **Loading Screen Infinita** ✅
- **Problema**: La pantalla de carga se quedaba colgada sin dar error
- **Solución**: 
  - Agregado try-catch individual para cada componente
  - Logging detallado de éxito/fallo de cada módulo
  - Mensajes en consola con emojis para fácil identificación
  - Timeout implícito con mensajes de error claros

### 2. **KeyError 'close'** ✅
- **Problema**: Error al buscar columna 'close' cuando existe como 'Close'
- **Solución**:
  - Búsqueda flexible de columnas: `['close', 'Close', 'CLOSE']`
  - Logging cuando no se encuentra la columna
  - Manejo gracioso con valores "N/A" en estadísticas

### 3. **Problemas de Geometría de Ventana** ✅
- **Problema**: La ventana cambiaba de tamaño/posición sin registrarse
- **Solución**:
  - Implementados `resizeEvent()` y `moveEvent()`
  - Logging de cambios significativos (>50px resize, >100px move)
  - Registro en session logger para análisis posterior

---

## 📊 Sistema de Logging Implementado

### **SessionLogger Mejorado**

El `SessionLogger` ahora captura:

#### **Nuevos Métodos Agregados:**

```python
# Eventos de UI
log_ui_event(event_type, details)
# Ejemplo: Clicks de botones, cambios de pestañas, actualizaciones de widgets

# Eventos de Ventana
log_window_event(event_type, window_info)
# Ejemplo: Resize, move, maximize, minimize

# Fallos de Componentes
log_component_load_failure(component_name, error_message, stack_trace)
# Ejemplo: Módulo que no carga al inicio

# Eventos de Carga de Datos
log_data_loading_event(event_type, data_info)
# Ejemplo: Inicio de carga, progreso, éxito, error
```

---

## 🔍 Puntos de Logging Implementados

### **1. Loading Screen** (`loading_screen.py`)

```python
✅ Config Manager loaded
✅ Session Logger loaded
✅ Backend engines loaded
✅ Backtester loaded
✅ Analysis engines loaded
✅ Live monitor loaded
✅ Settings manager loaded
✅ Reporters engine loaded
✅ Broker manager loaded (opcional)
✅ API components loaded (opcional)
✅ GUI tabs created
✅ Status bar created
✅ Configuration loaded
```

**Captura:**
- Éxito/fallo de cada componente individual
- Stack traces completos en errores
- Timing de carga
- Estado de componentes opcionales

### **2. Main Platform** (`main_platform.py`)

**Eventos Capturados:**

#### **Carga Automática de Datos:**
```python
# Al cargar BTC data exitosamente
session_logger.log_data_loading_event('auto_load_success', {
    'symbol': 'BTC-USD',
    'timeframe': '15Min',
    'records': 100417,
    'date_range': '2023-01-01 to 2025-11-12',
    'file_path': 'data/btc_15Min.csv'
})
```

#### **Errores en Carga de Datos:**
```python
# Cualquier error con stack trace completo
session_logger.log_error(
    error_type='data_loading_error',
    error_message=str(e),
    context={'action': 'auto_load_btc_data'},
    stack_trace=traceback.format_exc()
)
```

#### **Notificaciones a UI:**
```python
# Cuando se actualiza Tab1 con data
session_logger.log_ui_event('data_loaded_notification', {
    'tab': 'Tab1_DataManagement',
    'data_keys': ['BTC-USD_15Min']
})
```

#### **Cambios de Ventana:**
```python
# Resize significativo (>50px)
session_logger.log_window_event('resize', {
    'old_size': '1920x1080',
    'new_size': '1366x768',
    'difference': '554x312'
})

# Move significativo (>100px)
session_logger.log_window_event('move', {
    'old_pos': '0,23',
    'new_pos': '150,200',
    'difference': '150,177'
})
```

### **3. Tab1 - Data Management** (`platform_gui_tab1_improved.py`)

**Eventos Capturados:**

#### **Clicks de Botones:**
```python
# Al hacer click en "Load Data"
session_logger.log_ui_event('load_data_button_clicked', {
    'symbol': 'BTC-USD',
    'timeframe': '15Min',
    'start_date': '2023-01-01',
    'end_date': '2024-12-31',
    'multi_timeframe': True
})
```

#### **Éxito en Carga Manual:**
```python
# Cuando se carga data exitosamente
session_logger.log_data_loading_event('manual_load_success', {
    'timeframes': ['BTC-USD_5Min', 'BTC-USD_15Min'],
    'total_bars': 250000,
    'timeframes_count': 2
})
```

#### **Errores en Carga Manual:**
```python
# Cualquier error con contexto
session_logger.log_error(
    error_type='data_loading_error',
    error_message='API timeout after 30 seconds',
    context={'tab': 'Tab1_DataManagement'}
)
```

#### **Errores en Actualización de Charts:**
```python
# Error al actualizar preview con stack trace
self.logger.error(f"Error updating stats: {e}")
self.logger.error(traceback.format_exc())
```

---

## 📁 Estructura de Reportes

### **Archivo de Sesión** (`reports/sessions/session_YYYYMMDD_HHMMSS.json`)

```json
{
  "session_id": "20251116_143052",
  "start_time": "2025-11-16T14:30:52",
  "end_time": "2025-11-16T15:45:30",
  "duration_seconds": 4478,
  "user_actions": [
    {
      "timestamp": "2025-11-16T14:30:55",
      "type": "platform_start",
      "details": {"version": "2.0.0"},
      "result": "success"
    },
    {
      "timestamp": "2025-11-16T14:31:00",
      "type": "data_loading",
      "details": {
        "event_type": "auto_load_success",
        "data_info": {
          "symbol": "BTC-USD",
          "timeframe": "15Min",
          "records": 100417
        }
      },
      "result": "success"
    },
    {
      "timestamp": "2025-11-16T14:32:15",
      "type": "ui_load_data_button_clicked",
      "details": {
        "symbol": "ETH-USD",
        "timeframe": "1Hour"
      },
      "result": "success"
    }
  ],
  "errors": [
    {
      "timestamp": "2025-11-16T14:35:22",
      "type": "data_loading_error",
      "message": "KeyError: 'close'",
      "context": {"tab": "Tab1_DataManagement"},
      "stack_trace": "Traceback (most recent call last)...",
      "system_info": {
        "platform": "Windows-10-10.0.19045-SP0",
        "python_version": "3.11.5"
      }
    }
  ],
  "window_events": [
    {
      "timestamp": "2025-11-16T14:40:10",
      "type": "window_event",
      "details": {
        "event_type": "resize",
        "window_info": {
          "old_size": "1920x1080",
          "new_size": "1366x768"
        }
      }
    }
  ]
}
```

### **Reporte de Texto** (`reports/sessions/session_YYYYMMDD_HHMMSS.txt`)

```
================================================================================
TRADINGAI PLATFORM - SESSION REPORT
================================================================================
Session ID: 20251116_143052
Start Time: 2025-11-16 14:30:52
End Time:   2025-11-16 15:45:30
Duration:   1h 14m 38s

================================================================================
SUMMARY
================================================================================
Total Actions:           45
Total Errors:            3
Error Rate:              6.7%
Most Visited Tab:        Tab1_DataManagement
Backtests Completed:     5
Live Trading Sessions:   0
Data Downloads:          2

================================================================================
ERROR DETAILS
================================================================================
Total Errors: 3

ERROR #1
Type: data_loading_error
Time: 2025-11-16 14:35:22
Message: KeyError: 'close'
Context: {'tab': 'Tab1_DataManagement'}

ERROR #2
Type: ui_notification_error
Time: 2025-11-16 14:42:10
Message: 'NoneType' object has no attribute 'on_data_loaded'
...
```

---

## 🔧 Uso del Sistema de Logging

### **Para Desarrolladores:**

1. **Revisar errores en tiempo real:**
   ```bash
   # Ver logs en consola con códigos de color
   python -m src.main_platform
   ```

2. **Analizar sesión después del cierre:**
   ```bash
   # Ver el reporte de texto generado
   cat reports/sessions/session_YYYYMMDD_HHMMSS.txt
   ```

3. **Buscar errores específicos:**
   ```python
   import json
   
   with open('reports/sessions/session_YYYYMMDD_HHMMSS.json') as f:
       session = json.load(f)
   
   # Filtrar errores de tipo específico
   data_errors = [e for e in session['errors'] 
                  if e['type'] == 'data_loading_error']
   ```

### **Para Usuarios:**

Los reportes se generan automáticamente al cerrar la aplicación. Si hay un problema:

1. Cerrar la aplicación
2. Ir a `reports/sessions/`
3. Buscar el archivo más reciente
4. Compartir el archivo `.txt` o `.json` para análisis

---

## 🎨 Convenciones de Logging en Consola

```
✅ - Éxito/Completado
⚠️  - Advertencia/Componente Opcional
❌ - Error Crítico
ℹ️  - Información
📺 - Evento de UI
⏱️  - Evento Temporal/Timer
🚀 - Inicio de Proceso
```

---

## 📈 Métricas Capturadas

### **Por Sesión:**
- Duración total
- Número de acciones
- Tasa de errores
- Pestaña más visitada
- Estrategia más probada

### **Por Evento:**
- Timestamp exacto
- Tipo de evento
- Detalles contextuales
- Resultado (success/error)
- Stack trace (en errores)

### **Por Error:**
- Tipo de error
- Mensaje completo
- Contexto (qué se estaba haciendo)
- Stack trace
- Info del sistema
- Estado de la sesión

---

## 🔮 Próximos Pasos Sugeridos

1. **Dashboard de Análisis de Errores:**
   - Crear script que analice múltiples sesiones
   - Identificar errores recurrentes
   - Generar estadísticas de uso

2. **Alertas Automáticas:**
   - Enviar notificación si tasa de error > 10%
   - Detectar errores críticos automáticamente

3. **Logging de Performance:**
   - Timing de carga de cada componente
   - Tiempo de respuesta de UI
   - Uso de memoria/CPU

4. **Integración con Herramientas Externas:**
   - Sentry para tracking de errores
   - Elasticsearch para búsqueda de logs
   - Grafana para visualización

---

## 📝 Notas Importantes

- **Los reportes se guardan automáticamente** en `reports/sessions/`
- **Cada sesión genera 2 archivos**: `.json` (máquina) y `.txt` (humano)
- **Los logs en consola son en tiempo real** y muy verbosos para debugging
- **El sistema NO afecta performance** - todos los logs son async
- **Los session loggers son thread-safe** - pueden usarse desde cualquier thread

---

## ✅ Checklist de Eventos Cubiertos

- [x] Inicio de aplicación
- [x] Carga de componentes (individual)
- [x] Carga de datos (auto/manual)
- [x] Clicks de botones
- [x] Cambios de pestaña
- [x] Cambios de ventana (resize/move)
- [x] Errores en carga de datos
- [x] Errores en actualización de UI
- [x] Errores en componentes
- [x] Configuración guardada
- [x] Cierre de aplicación

---

**Fecha de Implementación:** 16 de Noviembre 2025  
**Versión:** 2.0.0  
**Estado:** ✅ Completo y Funcional
