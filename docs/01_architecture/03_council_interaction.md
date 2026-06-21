# Protocolo de Interacción del Consejo (Council Interaction Protocol)

## 1. Filosofía del Consejo
El Consejo no es una simple lista de reglas; es un organismo colegiado compuesto por **Agentes Expertos** (Personas) que evalúan el mercado y el sistema desde diferentes perspectivas (Dominios). La decisión final de trading es el resultado de un consenso ponderado, sujeto a vetos jerárquicos.

## 2. Estructura de Expertos y Dominios

| Experto | Rol | Dominio | Peso | Responsabilidad | Poder Especial |
|---------|-----|---------|------|-----------------|----------------|
| **Risk Warden** | Security Officer | `RISK` | 2.5 | Seguridad de capital, Drawdown, Apalancamiento. | **VETO ABSOLUTO**: Puede detener cualquier operación si viola límites críticos. |
| **Trend Master** | Strategy Lead | `TRADING` | 2.0 | Dirección del mercado, Estructura, Momentum. | **Voto Calificado**: Su opinión vale doble en decisiones de entrada. |
| **Data Oracle** | Data Engineer | `DATA` | 1.0 | Integridad de datos, Latencia, Anomalías. | **Veto Técnico**: Bloquea si los datos no son fiables. |
| **Architect Prime** | System Architect | `SYSTEM` | 1.0 | Salud del sistema, Recursos, Conectividad. | **Veto Técnico**: Bloquea si el sistema está inestable. |
| **Sentiment Seer** | Analyst | `SENTIMENT` | 1.0 | Noticias, Redes Sociales (Futuro). | Voto consultivo. |

## 3. Flujo de Toma de Decisiones

### Fase 1: Recolección de Evidencia (Rules Execution)
Cada experto tiene asignado un conjunto de **Reglas**.
- El sistema ejecuta todas las reglas disponibles para el contexto actual.
- Cada regla genera un resultado: `PASS`, `FAIL`, o `NEUTRAL`.

### Fase 2: Formación de Opinión del Experto
Cada experto agrega los resultados de *sus* reglas para formar su **Voto Individual**:
- Si **cualquier** regla crítica del experto falla -> El Experto vota **REJECT (-1)**.
- Si todas las reglas pasan -> El Experto vota **APPROVE (1)**.
- Si no tiene reglas aplicables o datos -> El Experto vota **ABSTAIN (0)**.

### Fase 3: Ronda de Vetos (The Gatekeepers)
Antes de contar votos, se verifican los vetos:
1.  **Risk Warden**: Si su voto es REJECT, la decisión final es **REJECT** (Razón: "Risk Veto").
2.  **Data/System**: Si detectan fallos críticos, la decisión es **REJECT** (Razón: "System Integrity Veto").

### Fase 4: Consenso Ponderado (The Vote)
Si no hay vetos, se calcula el **Score del Consejo**:

$$ Score = \frac{\sum (Voto_{experto} \times Peso_{experto})}{\sum Peso_{activos}} $$

- **Score > 0.5**: **STRONG BUY/SELL** (Alta convicción).
- **Score > 0.0**: **WEAK BUY/SELL** (Requiere confirmación adicional o tamaño reducido).
- **Score <= 0.0**: **NO TRADE**.

## 4. Resolución de Conflictos
- En caso de empate técnico (Score = 0), el voto de **Risk Warden** rompe el empate (conservador).
- Si **Trend Master** y **Risk Warden** están en desacuerdo (uno quiere operar, el otro ve riesgo moderado), prevalece la prudencia (Risk Warden reduce el tamaño de la posición o cancela).

## 5. Cosas a Considerar (Mejoras Futuras)
- **Adaptabilidad**: Los pesos deberían ajustarse dinámicamente según la volatilidad del mercado (ej. en alta volatilidad, Risk Warden sube a 3.0).
- **Memoria**: El Consejo debería recordar decisiones pasadas ("Ayer rechazamos esto, ¿qué cambió?").
- **Disidencia**: Registrar qué experto votó en contra de una operación ganadora (para recalibrar su peso).
