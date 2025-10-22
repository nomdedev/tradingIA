"""
Live Trading Engine
Motor de trading en vivo que integra todos los agentes con Alpaca paper trading
"""

import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import pytz

from .alpaca_client import AlpacaClient
from ..agents.agent_adapters import RLAgentAdapter, GAAgentAdapter
from ..agents.safe_trading_wrapper import SafeTradingWrapper

logger = logging.getLogger(__name__)

class LiveTradingEngine:
    """
    Motor completo de trading en vivo con integración de múltiples agentes
    """

    def __init__(self, alpaca_client: AlpacaClient, config: Dict[str, Any] = None):
        """
        Inicializar el motor de trading

        Args:
            alpaca_client: Cliente Alpaca inicializado
            config: Configuración del sistema
        """
        self.alpaca = alpaca_client
        self.config = config or self._get_default_config()

        # Inicializar agentes
        self.agents = {}
        self._initialize_agents()

        # Estado del sistema
        self.is_running = False
        self.last_update = None
        self.portfolio_history = []
        self.trading_log = []

        logger.info("🚀 Live Trading Engine inicializado")

    def _get_default_config(self) -> Dict[str, Any]:
        """Configuración por defecto"""
        return {
            'symbols': ['SPY'],  # Símbolos a tradear
            'check_interval': 60,  # Segundos entre checks
            'max_positions': 5,    # Máximo número de posiciones
            'risk_per_trade': 0.02,  # 2% riesgo por trade
            'max_drawdown': 0.05,   # 5% max drawdown
            'min_trade_amount': 100, # Monto mínimo por trade
            'agents': {
                'rl': {'enabled': True, 'weight': 0.4},
                'ga': {'enabled': True, 'weight': 0.3},
                'llm': {'enabled': True, 'weight': 0.3}
            }
        }

    def _initialize_agents(self):
        """Inicializar todos los agentes disponibles"""
        try:
            # RL Agent
            if self.config['agents']['rl']['enabled']:
                rl_model_path = os.path.join('models', 'ppo_spy_model.zip')
                if os.path.exists(rl_model_path):
                    self.agents['rl'] = RLAgentAdapter(rl_model_path)
                    logger.info("✅ RL Agent cargado")
                else:
                    logger.warning("⚠️ RL model no encontrado, RL Agent deshabilitado")

            # GA Agent
            if self.config['agents']['ga']['enabled']:
                ga_model_path = os.path.join('models', 'ga_strategy.pkl')
                if os.path.exists(ga_model_path):
                    self.agents['ga'] = GAAgentAdapter(ga_model_path)
                    logger.info("✅ GA Agent cargado")
                else:
                    logger.warning("⚠️ GA model no encontrado, GA Agent deshabilitado")

            # Multi-LLM Agent (TODO: Implementar)
            if self.config['agents']['llm']['enabled']:
                logger.warning("⚠️ Multi-LLM Agent no implementado aún, deshabilitado")
                # TODO: Implementar MultiLLMAgent cuando esté disponible
                # llm_config = {
                #     'groq_api_key': os.getenv('GROQ_API_KEY'),
                #     'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
                #     'xai_api_key': os.getenv('XAI_API_KEY'),
                #     'deepseek_api_key': os.getenv('DEEPSEEK_API_KEY')
                # }
                # if any(llm_config.values()):
                #     self.agents['llm'] = MultiLLMAgent(llm_config)
                #     logger.info("✅ Multi-LLM Agent cargado")
                # else:
                #     logger.warning("⚠️ No LLM API keys encontradas, LLM Agent deshabilitado")

            # Envolver agentes con risk management
            for agent_name, agent in self.agents.items():
                risk_config = {
                    'max_drawdown': self.config['max_drawdown'],
                    'risk_per_trade': self.config['risk_per_trade'],
                    'portfolio_heat_limit': 0.25
                }
                wrapped_agent = SafeTradingWrapper(agent, risk_config)
                self.agents[agent_name] = wrapped_agent
                logger.info(f"🛡️ {agent_name.upper()} Agent envuelto con risk management")

        except Exception as e:
            logger.error(f"❌ Error inicializando agentes: {e}")
            raise

    def start_trading(self):
        """Iniciar el trading en vivo"""
        if self.is_running:
            logger.warning("⚠️ Trading ya está ejecutándose")
            return

        logger.info("🎯 Iniciando trading en vivo...")
        self.is_running = True

        try:
            # Verificar que el mercado esté abierto
            if not self.alpaca.is_market_open():
                logger.info("⏰ Mercado cerrado, esperando apertura...")
                self._wait_for_market_open()

            # Ciclo principal de trading
            while self.is_running:
                try:
                    self._trading_cycle()
                    time.sleep(self.config['check_interval'])

                except KeyboardInterrupt:
                    logger.info("🛑 Interrupción detectada, deteniendo trading...")
                    break
                except Exception as e:
                    logger.error(f"❌ Error en ciclo de trading: {e}")
                    time.sleep(30)  # Esperar antes de reintentar

        except Exception as e:
            logger.error(f"❌ Error fatal en trading loop: {e}")
        finally:
            self.stop_trading()

    def stop_trading(self):
        """Detener el trading"""
        logger.info("🛑 Deteniendo trading en vivo...")
        self.is_running = False

        # Cancelar todas las órdenes abiertas
        try:
            open_orders = self.alpaca.get_open_orders()
            for order in open_orders:
                self.alpaca.cancel_order(order.id)
                logger.info(f"❌ Orden cancelada: {order.id}")
        except Exception as e:
            logger.error(f"❌ Error cancelando órdenes: {e}")

        # Generar reporte final
        self._generate_final_report()

    def _wait_for_market_open(self):
        """Esperar hasta que el mercado abra"""
        while not self.alpaca.is_market_open():
            market_info = self.alpaca.get_market_hours()
            next_open = market_info['next_open']

            if next_open:
                wait_seconds = (next_open - datetime.now(pytz.UTC)).total_seconds()
                if wait_seconds > 0:
                    logger.info(f"⏰ Esperando {wait_seconds/3600:.1f} horas hasta apertura del mercado")
                    time.sleep(min(wait_seconds, 3600))  # Esperar máximo 1 hora
                else:
                    break
            else:
                logger.warning("⚠️ No se pudo obtener próxima apertura, esperando 5 minutos...")
                time.sleep(300)

    def _trading_cycle(self):
        """Un ciclo completo de trading"""
        current_time = datetime.now()

        # Verificar estado del mercado
        if not self.alpaca.is_market_open():
            logger.debug("⏰ Mercado cerrado, saltando ciclo")
            return

        logger.info(f"🔄 Iniciando ciclo de trading - {current_time.strftime('%H:%M:%S')}")

        try:
            # Obtener datos actuales
            portfolio_value = self.alpaca.get_portfolio_value()
            positions = self.alpaca.get_positions()

            # Registrar en historial
            self.portfolio_history.append({
                'timestamp': current_time,
                'portfolio_value': portfolio_value,
                'positions_count': len(positions)
            })

            # Procesar cada símbolo
            for symbol in self.config['symbols']:
                self._process_symbol(symbol)

            self.last_update = current_time

        except Exception as e:
            logger.error(f"❌ Error en ciclo de trading: {e}")

    def _process_symbol(self, symbol: str):
        """Procesar decisiones de trading para un símbolo"""
        try:
            # Obtener precio actual
            current_price = self.alpaca.get_current_price(symbol)

            # Obtener datos históricos recientes (últimos 30 días)
            historical_data = self.alpaca.get_historical_data(symbol, days=30)

            if historical_data.empty:
                logger.warning(f"⚠️ No hay datos históricos para {symbol}")
                return

            # Preparar observación para agentes
            observation = self._prepare_observation(symbol, current_price, historical_data)

            # Obtener decisiones de todos los agentes
            decisions = {}
            for agent_name, agent in self.agents.items():
                try:
                    decision = agent.decide(observation)
                    decisions[agent_name] = decision
                    logger.debug(f"🤖 {agent_name.upper()}: {decision}")
                except Exception as e:
                    logger.error(f"❌ Error obteniendo decisión de {agent_name}: {e}")
                    decisions[agent_name] = {'action': 'HOLD', 'size': 0.0}

            # Combinar decisiones usando voting ensemble
            final_decision = self._combine_decisions(decisions)

            # Ejecutar trade si es necesario
            if final_decision['action'] != 'HOLD':
                self._execute_trade(symbol, final_decision, current_price)

            # Registrar decisión
            self.trading_log.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'price': current_price,
                'decisions': decisions,
                'final_decision': final_decision
            })

        except Exception as e:
            logger.error(f"❌ Error procesando {symbol}: {e}")

    def _prepare_observation(self, symbol: str, current_price: float, historical_data: pd.DataFrame) -> np.ndarray:
        """Preparar observación para los agentes"""
        try:
            # Calcular indicadores técnicos básicos
            close_prices = historical_data['close'].tail(20)  # Últimos 20 días

            # SMA 5 y 10
            sma5 = close_prices.rolling(5).mean().iloc[-1] if len(close_prices) >= 5 else current_price
            sma10 = close_prices.rolling(10).mean().iloc[-1] if len(close_prices) >= 10 else current_price

            # RSI (simplificado)
            gains = close_prices.diff().clip(lower=0).rolling(14).mean().iloc[-1]
            losses = -close_prices.diff().clip(upper=0).rolling(14).mean().iloc[-1]
            rsi = 100 - (100 / (1 + gains/losses)) if losses != 0 else 50

            # Volatilidad (std dev de retornos)
            returns = close_prices.pct_change()
            volatility = returns.std() * np.sqrt(252)  # Annualized

            # Crear observación de 15 elementos (compatible con TradingEnv)
            observation = np.array([
                current_price,           # 0: Precio actual
                sma5,                    # 1: SMA 5
                sma10,                   # 2: SMA 10
                rsi,                     # 3: RSI
                volatility,              # 4: Volatilidad
                close_prices.iloc[-1],   # 5: Precio cierre anterior
                close_prices.iloc[-2] if len(close_prices) > 1 else current_price,  # 6: Precio cierre -2
                historical_data['volume'].tail(5).mean(),  # 7: Volumen promedio 5d
                (current_price - sma5) / sma5,  # 8: Distancia a SMA5
                (current_price - sma10) / sma10,  # 9: Distancia a SMA10
                returns.iloc[-1],        # 10: Retorno diario
                returns.tail(5).std(),   # 11: Volatilidad 5d
                len(self.alpaca.get_positions()),  # 12: Número de posiciones
                self.alpaca.get_portfolio_value(),  # 13: Valor portfolio
                self.alpaca.get_buying_power()      # 14: Buying power
            ], dtype=np.float32)

            return observation

        except Exception as e:
            logger.error(f"❌ Error preparando observación para {symbol}: {e}")
            # Retornar observación básica en caso de error
            return np.array([current_price] * 15, dtype=np.float32)

    def _combine_decisions(self, decisions: Dict[str, Dict]) -> Dict[str, Any]:
        """Combinar decisiones de múltiples agentes usando weighted voting"""
        actions = []
        sizes = []

        for agent_name, decision in decisions.items():
            weight = self.config['agents'][agent_name]['weight']

            action = decision.get('action', 'HOLD')
            size = decision.get('size', 0.0)

            # Convertir acción a numérico para voting
            action_value = {'BUY': 1, 'SELL': -1, 'HOLD': 0}.get(action, 0)

            actions.append(action_value * weight)
            sizes.append(size * weight)

        # Decisión final por mayoría ponderada
        final_action_value = sum(actions)
        final_size = sum(sizes)

        if final_action_value > 0.1:  # Threshold para BUY
            final_action = 'BUY'
        elif final_action_value < -0.1:  # Threshold para SELL
            final_action = 'SELL'
        else:
            final_action = 'HOLD'
            final_size = 0.0

        final_decision = {
            'action': final_action,
            'size': abs(final_size),
            'confidence': abs(final_action_value)
        }

        logger.info(f"🎯 Decisión final: {final_action} (confianza: {final_decision['confidence']:.2f})")
        return final_decision

    def _execute_trade(self, symbol: str, decision: Dict[str, Any], current_price: float):
        """Ejecutar trade basado en decisión"""
        try:
            action = decision['action']
            size_percentage = decision['size']

            # Calcular cantidad basada en position sizing
            position_size = self.alpaca.calculate_position_size(
                symbol=symbol,
                risk_percentage=self.config['risk_per_trade'],
                stop_loss_percentage=0.02
            )

            # Ajustar por porcentaje de decisión
            qty = position_size * size_percentage

            # Verificar límites
            if qty < 1:
                logger.info(f"📊 Trade muy pequeño ({qty:.2f}), cancelado")
                return

            # Verificar posición existente para SELL
            if action == 'SELL':
                current_position = self.alpaca.get_position(symbol)
                if not current_position or float(current_position.qty) <= 0:
                    logger.info(f"📊 No hay posición para vender {symbol}")
                    return
                qty = min(qty, float(current_position.qty))

            # Verificar buying power para BUY
            if action == 'BUY':
                required_cash = qty * current_price
                available_cash = self.alpaca.get_buying_power()
                if required_cash > available_cash:
                    qty = available_cash / current_price
                    if qty < 1:
                        logger.info(f"📊 Insuficiente buying power para {symbol}")
                        return

            # Ejecutar orden
            order = self.alpaca.submit_order(
                symbol=symbol,
                qty=qty,
                side=action.lower(),
                order_type='market',
                time_in_force='day'
            )

            logger.info(f"✅ Trade ejecutado: {action} {qty:.0f} {symbol} @ ${current_price:.2f} (Order ID: {order.id})")

        except Exception as e:
            logger.error(f"❌ Error ejecutando trade para {symbol}: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Obtener estado actual del sistema"""
        return {
            'is_running': self.is_running,
            'last_update': self.last_update,
            'portfolio_value': self.alpaca.get_portfolio_value(),
            'positions_count': len(self.alpaca.get_positions()),
            'open_orders_count': len(self.alpaca.get_open_orders()),
            'agents_active': list(self.agents.keys()),
            'trading_log_entries': len(self.trading_log),
            'portfolio_history_points': len(self.portfolio_history)
        }

    def get_trading_history(self) -> List[Dict[str, Any]]:
        """Obtener historial de trading"""
        return self.trading_log.copy()

    def get_portfolio_history(self) -> List[Dict[str, Any]]:
        """Obtener historial del portfolio"""
        return self.portfolio_history.copy()

    def _generate_final_report(self):
        """Generar reporte final de la sesión"""
        try:
            logger.info("📊 Generando reporte final...")

            if not self.portfolio_history:
                logger.info("⚠️ No hay datos para generar reporte")
                return

            # Calcular métricas
            initial_value = self.portfolio_history[0]['portfolio_value']
            final_value = self.portfolio_history[-1]['portfolio_value']
            total_return = (final_value - initial_value) / initial_value * 100

            # Max drawdown
            values = [p['portfolio_value'] for p in self.portfolio_history]
            peak = values[0]
            max_drawdown = 0
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)

            # Trades ejecutados
            trades_executed = len([log for log in self.trading_log
                                 if log['final_decision']['action'] != 'HOLD'])

            logger.info("📈 REPORTE FINAL DE TRADING")
            logger.info("=" * 50)
            logger.info(f"Valor inicial: ${initial_value:.2f}")
            logger.info(f"Valor final: ${final_value:.2f}")
            logger.info(f"Retorno total: {total_return:.2f}%")
            logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
            logger.info(f"Trades ejecutados: {trades_executed}")
            logger.info(f"Duración: {(self.portfolio_history[-1]['timestamp'] - self.portfolio_history[0]['timestamp']).total_seconds() / 3600:.1f} horas")
            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"❌ Error generando reporte final: {e}")