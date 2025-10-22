"""
Safe Trading Wrapper - Moon Dev AI Agents Style
==============================================

Este wrapper integra agentes de trading (RL/GA) con gestión de riesgo profesional,
siguiendo el patrón de Moon Dev AI Agents para trading de acciones (SPY).

Características:
- Envuelve cualquier agente de trading con validación de riesgo
- Aplica MoonDevRiskAgent para control de posiciones
- Mantiene compatibilidad con el sistema existente de competencia
- Logging detallado de decisiones de riesgo
- Fallback seguro en caso de fallos

Autor: Moon Dev AI Agents (adaptado para trading de acciones)
"""

import logging
from typing import Dict, Any
from datetime import datetime
import traceback

from base_agent import BaseAgent
from moondev_risk_agent import MoonDevRiskAgent


class SafeTradingWrapper(BaseAgent):
    """
    Wrapper que combina un agente de trading con gestión de riesgo Moon Dev.

    Este wrapper actúa como un proxy entre el sistema de competencia y los agentes
    de trading, aplicando validación de riesgo antes de ejecutar cualquier trade.

    Attributes:
        trading_agent: El agente de trading subyacente (RL, GA, etc.)
        risk_agent: Instancia de MoonDevRiskAgent para validación de riesgo
        fallback_mode: Si está activado, permite trades básicos en caso de error
        risk_override: Permite bypass de riesgo en situaciones especiales
    """

    def __init__(self,
                 trading_agent: Any,
                 risk_config: Dict[str, Any] = None,
                 fallback_mode: bool = True,
                 risk_override: bool = False,
                 name: str = "SafeTradingWrapper"):
        """
        Inicializar Safe Trading Wrapper.

        Args:
            trading_agent: Agente de trading a envolver (debe tener método get_action)
            risk_config: Configuración para MoonDevRiskAgent
            fallback_mode: Si True, permite HOLD en caso de error del risk agent
            risk_override: Si True, permite bypass de validaciones de riesgo
            name: Nombre del wrapper para logging
        """

        # Configuración por defecto para risk agent
        default_risk_config = {
            'max_drawdown': 0.15,  # 15% max drawdown
            'max_portfolio_heat': 0.30,  # 30% portfolio heat
            'max_risk_per_trade': 0.05,  # 5% risk per trade
            'volatility_lookback': 20,
            'initial_balance': 10000.0
        }

        if risk_config:
            default_risk_config.update(risk_config)

        # Inicializar BaseAgent
        super().__init__(name=name, initial_balance=default_risk_config['initial_balance'])

        # Almacenar agente de trading
        self.trading_agent = trading_agent

        # Inicializar risk agent
        self.risk_agent = MoonDevRiskAgent(**default_risk_config)

        # Configuración del wrapper
        self.fallback_mode = fallback_mode
        self.risk_override = risk_override

        # Estadísticas del wrapper
        self.total_trades_processed = 0
        self.trades_blocked_by_risk = 0
        self.trades_adjusted_by_risk = 0
        self.risk_evaluation_errors = 0
        self.fallback_trades = 0

        # Logging específico del wrapper
        self.wrapper_logger = logging.getLogger(f"{name}.wrapper")
        self.wrapper_logger.setLevel(logging.INFO)

        # Crear handler si no existe
        if not self.wrapper_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.wrapper_logger.addHandler(handler)

        self.wrapper_logger.info(f"🛡️ SafeTradingWrapper inicializado con agente: {trading_agent.__class__.__name__}")
        self.wrapper_logger.info(f"⚙️ Configuración de riesgo: DD={default_risk_config['max_drawdown']:.1%}, "
                                f"Heat={default_risk_config['max_portfolio_heat']:.1%}, "
                                f"Risk/Trade={default_risk_config['max_risk_per_trade']:.1%}")

    def get_action(self,
                   state: Dict[str, Any],
                   market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Obtener acción del agente de trading con validación de riesgo.

        Args:
            state: Estado actual del portfolio y mercado
            market_data: Datos adicionales del mercado

        Returns:
            Dict con acción validada por riesgo
        """

        self.total_trades_processed += 1
        timestamp = datetime.now().isoformat()

        try:
            # 1. Obtener acción del agente de trading subyacente
            raw_action = self._get_trading_agent_action(state, market_data)

            # 2. Preparar trade proposal para risk evaluation
            trade_proposal = self._prepare_trade_proposal(raw_action, state, market_data)

            # 3. Evaluar riesgo con MoonDevRiskAgent
            risk_evaluation = self._evaluate_risk(trade_proposal, state, market_data)

            # 4. Decidir acción final basada en evaluación de riesgo
            final_action = self._decide_final_action(raw_action, risk_evaluation, state)

            # 5. Logging detallado
            self._log_decision(raw_action, risk_evaluation, final_action, timestamp)

            # 6. Actualizar estadísticas
            self._update_statistics(risk_evaluation, final_action)

            return final_action

        except Exception as e:
            # Fallback en caso de error
            return self._handle_error(e, state, timestamp)

    def _get_trading_agent_action(self,
                                state: Dict[str, Any],
                                market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Obtener acción del agente de trading subyacente.

        Maneja diferentes formatos de agentes (RL con dict de acción, GA con tupla, etc.)
        """

        try:
            # Intentar método get_action (estándar)
            if hasattr(self.trading_agent, 'get_action'):
                action = self.trading_agent.get_action(state, market_data)

            # Intentar método predict (estilo RL)
            elif hasattr(self.trading_agent, 'predict'):
                action = self.trading_agent.predict(state)

            # Intentar llamada directa (función)
            elif callable(self.trading_agent):
                action = self.trading_agent(state)

            else:
                raise AttributeError(f"Trading agent {self.trading_agent.__class__.__name__} "
                                   "no tiene método get_action, predict o no es callable")

            # Normalizar formato de acción
            return self._normalize_action_format(action)

        except Exception as e:
            self.wrapper_logger.error(f"❌ Error obteniendo acción del trading agent: {str(e)}")
            raise

    def _normalize_action_format(self, action: Any) -> Dict[str, Any]:
        """
        Normalizar diferentes formatos de acción a formato estándar.

        Formats soportados:
        - Dict: {'action': int, 'size': float, 'confidence': float}
        - Tuple: (action, size) o (action, size, confidence)
        - Int: solo action (0=hold, 1=buy, 2=sell)
        """

        if isinstance(action, dict):
            # Ya está en formato correcto
            return action

        elif isinstance(action, (tuple, list)):
            if len(action) == 2:
                return {
                    'action': int(action[0]),
                    'size': float(action[1]),
                    'confidence': 0.5  # default
                }
            elif len(action) == 3:
                return {
                    'action': int(action[0]),
                    'size': float(action[1]),
                    'confidence': float(action[2])
                }

        elif isinstance(action, (int, float)):
            # Solo action, asumir size basado en estado
            return {
                'action': int(action),
                'size': 0,  # Se calculará después
                'confidence': 0.5
            }

        else:
            raise ValueError(f"Formato de acción no soportado: {type(action)} - {action}")

    def _prepare_trade_proposal(self,
                               action: Dict[str, Any],
                               state: Dict[str, Any],
                               market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Preparar proposal de trade para evaluación de riesgo.
        """

        # Obtener precio actual
        current_price = market_data.get('Close', state.get('price', 0)) if market_data else state.get('price', 0)

        # Para BUY, si size=0, calcular basado en balance disponible
        if action['action'] == 1 and action.get('size', 0) == 0:  # BUY
            balance = state.get('balance', 0)
            # Calcular size máximo posible (dejando margen)
            max_size = (balance * 0.95) / current_price if current_price > 0 else 0
            action['size'] = max_size

        return {
            'action': action['action'],
            'price': current_price,
            'size': action.get('size', 0),
            'confidence': action.get('confidence', 0.5)
        }

    def _evaluate_risk(self,
                      trade_proposal: Dict[str, Any],
                      state: Dict[str, Any],
                      market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluar riesgo del trade usando MoonDevRiskAgent.
        """

        try:
            # Preparar portfolio state para risk agent
            portfolio_state = self._prepare_portfolio_state(state)

            # Evaluar con risk agent
            risk_result = self.risk_agent.evaluate_trade_risk(
                trade_proposal, portfolio_state, market_data
            )

            return risk_result

        except Exception as e:
            self.risk_evaluation_errors += 1
            self.wrapper_logger.error(f"❌ Error en evaluación de riesgo: {str(e)}")
            self.wrapper_logger.error(f"Traceback: {traceback.format_exc()}")

            # En caso de error, rechazar trade si no hay fallback
            if not self.fallback_mode:
                return {
                    'approved': False,
                    'adjusted_size': 0,
                    'stop_loss': None,
                    'take_profit': None,
                    'risk_score': 1.0,
                    'reason': f'Error en risk evaluation: {str(e)}'
                }

            # Fallback: aprobar con parámetros conservadores
            return {
                'approved': True,
                'adjusted_size': trade_proposal['size'] * 0.1,  # Solo 10%
                'stop_loss': trade_proposal['price'] * 0.97,  # 3% stop loss
                'take_profit': None,
                'risk_score': 0.5,
                'reason': 'FALLBACK: Risk evaluation failed, using conservative parameters'
            }

    def _prepare_portfolio_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preparar estado del portfolio en formato esperado por risk agent.
        """

        return {
            'balance': state.get('balance', 0),
            'shares': state.get('shares', 0),
            'net_worth': state.get('net_worth', state.get('balance', 0)),
            'current_drawdown': state.get('current_drawdown', 0),
            'peak_value': state.get('peak_value', state.get('balance', 0))
        }

    def _decide_final_action(self,
                           raw_action: Dict[str, Any],
                           risk_evaluation: Dict[str, Any],
                           state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decidir acción final basada en evaluación de riesgo.
        """

        if not risk_evaluation['approved']:
            # Risk agent rechazó el trade
            return {
                'action': 0,  # HOLD
                'size': 0,
                'confidence': 0,
                'risk_approved': False,
                'risk_reason': risk_evaluation['reason'],
                'stop_loss': None,
                'take_profit': None
            }

        # Risk agent aprobó, usar parámetros ajustados
        final_action = raw_action.copy()
        final_action['size'] = risk_evaluation['adjusted_size']
        final_action['risk_approved'] = True
        final_action['risk_reason'] = risk_evaluation['reason']
        final_action['stop_loss'] = risk_evaluation.get('stop_loss')
        final_action['take_profit'] = risk_evaluation.get('take_profit')
        final_action['risk_score'] = risk_evaluation['risk_score']

        return final_action

    def _log_decision(self,
                     raw_action: Dict[str, Any],
                     risk_evaluation: Dict[str, Any],
                     final_action: Dict[str, Any],
                     timestamp: str):
        """
        Logging detallado de la decisión de trading.
        """

        action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}

        self.wrapper_logger.info(
            f"🤖 [{timestamp}] Trading Decision: "
            f"Agent={action_names.get(raw_action['action'], 'UNKNOWN')} "
            f"Size={raw_action.get('size', 0):.2f} → "
            f"Risk={risk_evaluation['approved']} "
            f"Final={action_names.get(final_action['action'], 'UNKNOWN')} "
            f"Size={final_action.get('size', 0):.2f}"
        )

        if not risk_evaluation['approved']:
            self.wrapper_logger.warning(f"🚫 Trade BLOCKED: {risk_evaluation['reason']}")
        elif risk_evaluation['adjusted_size'] != raw_action.get('size', 0):
            self.wrapper_logger.info(f"⚖️ Size ADJUSTED: {risk_evaluation['reason']}")
        else:
            self.wrapper_logger.info(f"✅ Trade APPROVED: {risk_evaluation['reason']}")

    def _update_statistics(self, risk_evaluation: Dict[str, Any], final_action: Dict[str, Any]):
        """
        Actualizar estadísticas del wrapper.
        """

        if not risk_evaluation['approved']:
            self.trades_blocked_by_risk += 1
        elif risk_evaluation['adjusted_size'] != final_action.get('original_size', risk_evaluation['adjusted_size']):
            self.trades_adjusted_by_risk += 1

    def _handle_error(self, error: Exception, state: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """
        Manejar errores en el proceso de decisión.
        """

        self.wrapper_logger.error(f"💥 CRITICAL ERROR in SafeTradingWrapper: {str(error)}")
        self.wrapper_logger.error(f"Traceback: {traceback.format_exc()}")

        if self.fallback_mode:
            self.fallback_trades += 1
            self.wrapper_logger.warning("🛟 FALLBACK MODE: Executing HOLD due to error")

            return {
                'action': 0,  # HOLD
                'size': 0,
                'confidence': 0,
                'error': str(error),
                'fallback': True
            }
        else:
            # Sin fallback, relanzar error
            raise error

    def get_wrapper_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del wrapper.
        """

        risk_stats = self.risk_agent.get_risk_stats()

        return {
            'total_trades_processed': self.total_trades_processed,
            'trades_blocked_by_risk': self.trades_blocked_by_risk,
            'trades_adjusted_by_risk': self.trades_adjusted_by_risk,
            'risk_evaluation_errors': self.risk_evaluation_errors,
            'fallback_trades': self.fallback_trades,
            'block_rate': self.trades_blocked_by_risk / self.total_trades_processed if self.total_trades_processed > 0 else 0,
            'adjustment_rate': self.trades_adjusted_by_risk / self.total_trades_processed if self.total_trades_processed > 0 else 0,
            'error_rate': self.risk_evaluation_errors / self.total_trades_processed if self.total_trades_processed > 0 else 0,
            'risk_agent_stats': risk_stats
        }

    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        SafeTradingWrapper delega análisis al trading agent subyacente,
        pero aplica validación de riesgo.

        Este método mantiene compatibilidad con la interfaz BaseAgent.
        """
        # Obtener análisis del trading agent
        if hasattr(self.trading_agent, 'analyze'):
            raw_analysis = self.trading_agent.analyze(market_data)
        else:
            # Fallback si no tiene analyze
            raw_analysis = {
                'action': 0,
                'confidence': 0.5,
                'reasoning': 'Trading agent analysis not available'
            }

        # Preparar portfolio state (usando estado actual del wrapper)
        portfolio_state = {
            'balance': self.balance,
            'shares': self.shares,
            'net_worth': self.portfolio_value,
            'current_drawdown': self.current_drawdown
        }

        # Preparar trade proposal
        trade_proposal = {
            'action': raw_analysis['action'],
            'price': market_data.get('Close', 0),
            'size': 0,  # Se calculará después
            'confidence': raw_analysis.get('confidence', 0.5)
        }

        # Evaluar riesgo
        risk_evaluation = self._evaluate_risk(trade_proposal, portfolio_state, market_data)

        # Decidir acción final
        if risk_evaluation['approved']:
            final_action = raw_analysis['action']
            confidence = raw_analysis.get('confidence', 0.5)
            reasoning = f"Approved by risk management: {risk_evaluation['reason']}"
        else:
            final_action = 0  # HOLD
            confidence = 0.0
            reasoning = f"Blocked by risk management: {risk_evaluation['reason']}"

        return {
            'action': final_action,
            'confidence': confidence,
            'reasoning': reasoning,
            'risk_score': risk_evaluation.get('risk_score', 0),
            'stop_loss': risk_evaluation.get('stop_loss'),
            'take_profit': risk_evaluation.get('take_profit')
        }

    def reset(self):
        """
        Resetear estado del wrapper y risk agent.
        """

        super().reset()
        self.risk_agent.reset_risk_state()

        self.total_trades_processed = 0
        self.trades_blocked_by_risk = 0
        self.trades_adjusted_by_risk = 0
        self.risk_evaluation_errors = 0
        self.fallback_trades = 0

        self.wrapper_logger.info("🔄 SafeTradingWrapper reseteado")

    def __str__(self) -> str:
        """
        Representación string del wrapper.
        """

        stats = self.get_wrapper_stats()
        return (f"SafeTradingWrapper(trading_agent={self.trading_agent.__class__.__name__}, "
                f"trades_processed={stats['total_trades_processed']}, "
                f"blocked={stats['trades_blocked_by_risk']}, "
                f"errors={stats['risk_evaluation_errors']})")