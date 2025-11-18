# Sistema de Alertas y Notificaciones

## Descripción General

El sistema de alertas y notificaciones de Trading IA proporciona un mecanismo completo para monitorear eventos importantes del sistema, resultados de estrategias y condiciones de mercado. Permite a los usuarios recibir notificaciones en tiempo real sobre eventos críticos y mantener un registro histórico de todas las actividades.

## Características Principales

### Tipos de Alertas
- **Trading Signal**: Señales de trading generadas por estrategias
- **Performance Threshold**: Umbrales de rendimiento alcanzados
- **System Status**: Estado del sistema y componentes
- **Strategy Error**: Errores en la ejecución de estrategias
- **Data Issue**: Problemas con datos de mercado
- **Connection Lost/Restored**: Pérdida/restauración de conexiones
- **Custom**: Alertas personalizadas definidas por el usuario

### Niveles de Severidad
- **Low**: Información general, no requiere acción inmediata
- **Medium**: Requiere atención pero no es crítico
- **High**: Requiere acción inmediata
- **Critical**: Situación crítica que puede afectar operaciones

### Métodos de Notificación
- **GUI**: Notificaciones visuales en la interfaz
- **Sound**: Alertas sonoras
- **Email**: Notificaciones por correo electrónico
- **Log**: Registro en archivos de log

## Arquitectura del Sistema

### Componentes Principales

#### AlertManager
- Gestiona todas las alertas activas e históricas
- Maneja reglas de alerta y condiciones de activación
- Coordina la entrega de notificaciones
- Proporciona auto-acuse de recibo de alertas

#### NotificationHandler
- Gestiona la entrega de notificaciones a través de diferentes canales
- Soporta notificaciones GUI, sonido, email y logging
- Maneja configuración específica de cada método

#### Alert Types
- Define estructuras de datos para alertas y reglas
- Proporciona enumeraciones para tipos y severidades
- Incluye configuraciones para métodos de notificación

#### AlertsPanel (UI)
- Interfaz gráfica para gestión de alertas
- Visualización de alertas activas e históricas
- Configuración de reglas y preferencias
- Diálogos de configuración avanzada

### Reglas de Alerta

Las reglas de alerta permiten automatizar la generación de notificaciones basadas en condiciones específicas:

```python
from core.alerts import AlertRule, AlertType, NotificationMethod

# Ejemplo: Regla para alto rendimiento
high_perf_rule = AlertRule(
    id="high_performance",
    name="Alto Rendimiento",
    type=AlertType.PERFORMANCE_THRESHOLD,
    conditions={"min_severity": "medium"},
    notification_methods=[NotificationMethod.GUI, NotificationMethod.LOG],
    cooldown_minutes=30  # Evitar spam de notificaciones
)
```

## Configuración

### Configuración Básica

```python
from core.alerts import AlertConfig

config = AlertConfig(
    enabled=True,
    sound_enabled=True,
    gui_notifications_enabled=True,
    log_alerts=True,
    auto_acknowledge_after_minutes=60,  # Auto-acuse después de 1 hora
    max_alerts_history=1000
)
```

### Configuración de Email

```python
email_config = {
    'server': 'smtp.gmail.com',
    'port': 587,
    'username': 'your-email@gmail.com',
    'password': 'your-app-password',
    'to': ['alerts@yourcompany.com']
}
```

## Uso Programático

### Activación de Alertas

```python
from core.ui.dashboard_controller import DashboardController

controller = DashboardController()

# Alerta de error crítico
controller.trigger_alert(
    AlertType.STRATEGY_ERROR,
    AlertSeverity.CRITICAL,
    "Error Crítico en Estrategia",
    "La estrategia Momentum ha fallado con error de división por cero",
    "momentum_strategy",
    {"error_details": "ZeroDivisionError", "strategy_params": {...}}
)

# Alerta de buen rendimiento
controller.trigger_alert(
    AlertType.PERFORMANCE_THRESHOLD,
    AlertSeverity.MEDIUM,
    "Buen Rendimiento Detectado",
    f"Sharpe ratio de {sharpe:.2f} supera el umbral de 1.5",
    "backtester",
    {"sharpe_ratio": sharpe, "total_return": total_return}
)
```

### Gestión de Alertas

```python
# Obtener alertas activas
active_alerts = controller.get_active_alerts()

# Marcar alerta como leída
controller.acknowledge_alert(alert_id)

# Obtener historial
history = controller.get_alert_history(limit=50)
```

## Integración con la UI

### Panel de Alertas

El panel de alertas se integra como una pestaña en la interfaz principal:

1. **Alertas Activas**: Muestra alertas no reconocidas con colores por severidad
2. **Historial**: Registro completo de todas las alertas
3. **Configuración**: Personalización de reglas y métodos de notificación

### Notificaciones en Tiempo Real

- Alertas críticas muestran diálogos modales inmediatamente
- Notificaciones visuales aparecen en la barra de estado
- Alertas sonoras para eventos importantes
- Integración con el sistema de logging

## Alertas Automáticas

El sistema incluye reglas preconfiguradas para eventos comunes:

- **Errores de Backtest**: Notificación inmediata para errores en ejecución
- **Alto Rendimiento**: Alerta cuando se alcanzan umbrales de Sharpe ratio
- **Problemas de Conexión**: Monitoreo de conectividad de datos
- **Errores de Estrategia**: Detección de excepciones en estrategias

## Monitoreo y Mantenimiento

### Limpieza Automática
- Auto-acuse de recibo después de tiempo configurable
- Límite máximo de historial para evitar crecimiento excesivo
- Limpieza periódica de alertas antiguas

### Logging Integrado
- Todas las alertas se registran con timestamp y contexto
- Niveles de log apropiados por severidad
- Archivos de log rotativos para mantenimiento

## Extensibilidad

### Agregar Nuevos Tipos de Alerta

```python
# En alert_types.py
class AlertType(Enum):
    # ... tipos existentes ...
    MARKET_VOLATILITY = "market_volatility"
    PORTFOLIO_RISK = "portfolio_risk"
```

### Métodos de Notificación Personalizados

```python
# En notification_handler.py
def _notify_custom(self, alert: Alert):
    # Implementar notificación personalizada
    # Ej: integración con Slack, Telegram, etc.
    pass
```

## Solución de Problemas

### Alertas No Aparecen
1. Verificar que el sistema de alertas esté habilitado
2. Comprobar configuración de métodos de notificación
3. Revisar reglas de alerta y condiciones

### Notificaciones Duplicadas
1. Verificar configuración de cooldown en reglas
2. Comprobar lógica de auto-acuse de recibo
3. Revisar múltiples reglas activando la misma alerta

### Problemas de Email
1. Verificar configuración SMTP
2. Comprobar credenciales y permisos
3. Revisar lista de destinatarios

## Rendimiento

- Procesamiento asíncrono de notificaciones
- Almacenamiento eficiente de historial
- Monitoreo en background sin afectar operaciones principales
- Optimización de consultas de base de datos (futuro)

## Futuras Mejoras

- Integración con bases de datos para persistencia
- API REST para acceso remoto a alertas
- Notificaciones push móviles
- Análisis de patrones en historial de alertas
- Reglas de alerta basadas en machine learning