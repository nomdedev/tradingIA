# 📋 PLAN DE CONSOLIDACIÓN Y LIMPIEZA (FASE 2.5)

**Fecha:** 16 de Diciembre, 2025
**Estado:** DRAFT
**Objetivo:** Unificar arquitectura, eliminar deuda técnica y preparar para Fase 3.

---

## 🔍 DIAGNÓSTICO ACTUAL

### 1. Duplicidad Estructural (`src` vs `core`)
Existe una división confusa entre `src/` (código original/legacy y nuevas implementaciones) y `core/` (arquitectura refactorizada).

*   **Riesgo**: `src/risk/` (Kelly, Metrics) vs `core/risk/` (RiskManager/KillSwitch).
*   **Ejecución**: `src/execution/` (Realistic components) vs `core/execution/` (BacktesterCore).
*   **Estrategias**: Dispersas en `src/`, `strategies/`, `core/strategies/`.

### 2. Componentes de Riesgo Desconectados
Hemos implementado herramientas poderosas de riesgo, pero viven en silos:
*   `KellyPositionSizer` (src/risk) -> Usado por `BacktesterCore`.
*   `RiskMetricsCalculator` (src/risk) -> Usado por `Tab11` (UI).
*   `RiskManager` (core/risk) -> Kill Switch y Drawdown check (Runtime). **NO integrado** con el BacktesterCore actual.

### 3. Documentación Desactualizada
Los documentos en `docs/` reflejan planes anteriores. Necesitamos actualizar:
*   `full_project_docs.md`: Incluir nuevos módulos de riesgo.
*   `GUIA_USUARIO_COMPLETA.md`: Explicar cómo usar Kelly y Risk Dashboard.

---

## 🚀 PLAN DE ACCIÓN INMEDIATO

### PASO 1: Integración de Risk Manager (Runtime)
El `RiskManager` (Kill Switch) existe en `core/risk` pero `BacktesterCore` no lo está usando activamente para detener el trading si se viola el Max Drawdown diario.
*   **Tarea**: Inyectar `RiskManager` en `BacktesterCore`.
*   **Beneficio**: Protección real contra pérdidas catastróficas durante el backtest/live simulation.

### PASO 2: Consolidación de Módulos de Riesgo
Mover todo lo relacionado con riesgo a una estructura unificada en `core/risk/`.
*   Mover `src/risk/kelly_sizer.py` -> `core/risk/kelly_sizer.py`.
*   Mover `src/risk/risk_metrics.py` -> `core/risk/risk_metrics.py`.
*   Actualizar imports en `BacktesterCore` y `Tab11`.

### PASO 3: Actualización de Documentación
Generar una guía actualizada de las capacidades de riesgo.
*   Crear `docs/RISK_MANAGEMENT_GUIDE.md`.

### PASO 4: Limpieza de Código Muerto
Identificar scripts en `src/` que ya no se usan o han sido reemplazados por `core/`.

---

## 📝 TAREAS PRIORITARIAS (Siguiente Sprint)

1.  [x] **Refactor**: Mover `kelly_sizer.py` y `risk_metrics.py` a `core/risk/`.
2.  [x] **Integration**: Conectar `RiskManager` (Kill Switch) dentro del bucle de `BacktesterCore`.
3.  [x] **Docs**: Crear documentación unificada de riesgo.
4.  [x] **Test**: Verificar que el Kill Switch detiene el backtest cuando se excede el DD.
5.  [x] **Cleanup**: Mover scripts legacy de `src/` a `archive/legacy_src/`.
