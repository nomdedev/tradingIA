#!/usr/bin/env python3
"""
ADAPTADORES: Integración de agentes RL/GA existentes con SafeTradingWrapper

Adaptadores que convierten las estrategias de backtesting existentes
en agentes compatibles con SafeTradingWrapper para gestión de riesgo.

Autor: Moon Dev AI Agents Integration
Fecha: 2025
"""

import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import pickle
from typing import Dict, Any

# Importar agentes existentes
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'trading_competition'))
from environments.trading_env import TradingEnv

# Importar Moon Dev AI Agents
from agents.base_agent import BaseAgent


class RLAgentAdapter(BaseAgent):
    """
    Adaptador para agente RL (PPO) existente.
    Convierte el modelo PPO en un agente compatible con SafeTradingWrapper.
    """

    def __init__(self, model_path: str = "models/ppo_trading_agent.zip", **kwargs):
        """
        Inicializar adaptador RL.

        Args:
            model_path: Ruta al modelo PPO guardado
            **kwargs: Parámetros adicionales para BaseAgent
        """
        super().__init__(**kwargs)
        self.model_path = model_path
        self.model = None
        self.env = None
        self._load_model()

    def _load_model(self):
        """Cargar modelo PPO entrenado."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo RL no encontrado: {self.model_path}")

        try:
            self.model = PPO.load(self.model_path)
            self.logger.info(f"✓ Modelo RL cargado desde {self.model_path}")

            # Crear entorno para predicciones (necesario para normalización)
            data_path = "data/processed/SPY_with_indicators.csv"
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, index_col=0, parse_dates=True)
                self.env = TradingEnv(df)
                self.logger.info("✓ Entorno de trading creado para predicciones")
            else:
                self.logger.warning("⚠️ Datos procesados no encontrados, algunas funcionalidades limitadas")

        except Exception as e:
            self.logger.error(f"❌ Error cargando modelo RL: {e}")
            raise

    def get_action(self, state: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Obtener acción del agente RL.

        Args:
            state: Estado del portfolio {'balance', 'shares', 'net_worth', etc.}
            market_data: Datos de mercado con indicadores técnicos

        Returns:
            Dict con 'action' (0=hold, 1=buy, 2=sell), 'size', 'confidence'
        """
        if self.model is None:
            self.logger.warning("Modelo RL no cargado, retornando HOLD")
            return {
                'action': 0,  # HOLD
                'size': 0,
                'confidence': 0.0
            }

        try:
            # Crear observación usando el mismo método que el TradingEnv
            obs = self._create_observation_from_market_data(market_data, state)

            # Predecir acción
            action, _states = self.model.predict(obs, deterministic=True)

            # Convertir acción numérica a formato estándar
            # Asumiendo que el modelo retorna: 0=hold, 1=buy, 2=sell
            action_int = int(action)

            # Calcular confianza basada en el valor Q (si disponible)
            confidence = 0.7  # Valor por defecto, podríamos mejorar esto

            # Calcular tamaño sugerido basado en estado del portfolio
            size = self._calculate_position_size(state, market_data)

            result = {
                'action': action_int,
                'size': size,
                'confidence': confidence
            }

            self.logger.debug(f"RL Agent action: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error en predicción RL: {e}")
            return {
                'action': 0,  # HOLD por seguridad
                'size': 0,
                'confidence': 0.0
            }

    def _create_observation_from_market_data(self, market_data: Dict[str, Any], state: Dict[str, Any]) -> np.ndarray:
        """
        Crear observación de 15 elementos compatible con TradingEnv.
        Usa la misma lógica que TradingEnv._get_observation()
        """
        try:
            # Extraer valores de market_data (asumiendo que contiene todos los indicadores)
            row = market_data

            # Calcular precio normalizado (simplificado - en producción usar ventana histórica)
            close = row.get('Close', 0)
            close_normalized = close / 1000.0  # Normalización simple

            # Returns
            returns_1d = row.get('returns_1d', 0.0)
            returns_5d = row.get('returns_5d', 0.0)

            # Momentum indicators (normalizados)
            rsi_14 = row.get('RSI_14', 50) / 100.0
            rsi_21 = row.get('RSI_21', 50) / 100.0
            macd = row.get('MACD_12_26_9', 0) / 10.0  # Escalar MACD
            macd_signal = row.get('MACDs_12_26_9', 0) / 10.0

            # Volatilidad
            atr_relative = row.get('ATR_14', 0) / (close + 1e-8)
            volatility_20d = row.get('volatility_20d', 0) * 100

            # Tendencia
            sma_ratio = row.get('SMA_20_50_ratio', 1.0) - 1.0  # Centrar en 0
            adx = row.get('ADX_14', 25) / 100.0

            # Volumen (simplificado)
            obv_normalized = row.get('OBV', 0) / 1000000.0  # Normalización simple
            cmf = row.get('CMF', 0.0)

            # Bollinger position
            bb_position = row.get('bb_position', 0.5)

            # Estado de posición
            position_state = 1.0 if state.get('shares', 0) > 0 else 0.0

            # Crear array de observación (15 elementos)
            obs = [
                close_normalized,
                returns_1d,
                returns_5d,
                rsi_14,
                rsi_21,
                macd,
                macd_signal,
                atr_relative,
                volatility_20d,
                sma_ratio,
                adx,
                obv_normalized,
                cmf,
                bb_position,
                position_state
            ]

            # Convertir a numpy array, manejar NaN, clipear
            obs_array = np.array(obs, dtype=np.float32)
            obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=10.0, neginf=-10.0)
            obs_array = np.clip(obs_array, -10.0, 10.0)

            return obs_array

        except Exception as e:
            self.logger.error(f"Error creando observación RL: {e}")
            return np.zeros(15, dtype=np.float32)

    def _create_observation(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        Método legacy - mantener compatibilidad.
        """
        return self._create_observation_from_market_data(market_data, {})

    def _calculate_position_size(self, state: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """
        Calcular tamaño de posición sugerido.
        Simplificación - en producción usar lógica más sofisticada.
        """
        balance = state.get('balance', 0)
        price = market_data.get('Close', 0)

        if balance <= 0 or price <= 0:
            return 0

        # Usar 10% del balance por defecto (conservador)
        max_position_value = balance * 0.1
        size = max_position_value / price

        return max(0, size)

    def analyze(self):
        """Implementar método abstracto de BaseAgent."""
        return {
            'agent_type': 'RL_PPO',
            'model_loaded': self.model is not None,
            'status': 'active'
        }


class GAAgentAdapter(BaseAgent):
    """
    Adaptador para agente GA (Genetic Algorithm) existente.
    Convierte la estrategia GA en un agente compatible con SafeTradingWrapper.
    """

    def __init__(self, model_path: str = "models/ga_best_individual.pkl", **kwargs):
        """
        Inicializar adaptador GA.

        Args:
            model_path: Ruta al individuo GA guardado
            **kwargs: Parámetros adicionales para BaseAgent
        """
        super().__init__(**kwargs)
        self.model_path = model_path
        self.ga_strategy = None
        self._load_model()

    def _load_model(self):
        """Cargar estrategia GA entrenada."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo GA no encontrado: {self.model_path}")

        try:
            with open(self.model_path, 'rb') as f:
                # El archivo contiene un diccionario con información del entrenamiento
                data = pickle.load(f)
                # Extraer la estrategia GA del diccionario
                self.ga_strategy = data['strategy']
            self.logger.info(f"✓ Modelo GA cargado desde {self.model_path}")

        except Exception as e:
            self.logger.error(f"❌ Error cargando modelo GA: {e}")
            raise

    def get_action(self, state: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Obtener acción del agente GA.

        Args:
            state: Estado del portfolio {'balance', 'shares', 'net_worth', etc.}
            market_data: Datos de mercado con indicadores técnicos

        Returns:
            Dict con 'action' (0=hold, 1=buy, 2=sell), 'size', 'confidence'
        """
        if self.ga_strategy is None:
            self.logger.warning("Estrategia GA no cargada, retornando HOLD")
            return {
                'action': 0,  # HOLD
                'size': 0,
                'confidence': 0.0
            }

        try:
            # Obtener decisión de la estrategia GA
            # El método decide retorna (acción, razón)
            result = self.ga_strategy.decide(pd.Series(market_data), state.get('shares', 0))

            # Verificar que sea una tupla
            if not isinstance(result, tuple) or len(result) != 2:
                self.logger.warning(f"GA decide retornó {type(result)}: {result}, usando HOLD")
                decision, reason = "HOLD", "Error en decisión GA"
            else:
                decision, reason = result

            # Convertir decisión a acción estándar
            if decision == "BUY":
                action = 1  # BUY
                confidence = 0.8
            elif decision == "SELL":
                action = 2  # SELL
                confidence = 0.8
            else:  # HOLD
                action = 0  # HOLD
                confidence = 0.5

            # Calcular tamaño sugerido basado en estado del portfolio
            size = self._calculate_position_size(state, market_data)

            result = {
                'action': action,
                'size': size,
                'confidence': confidence
            }

            self.logger.debug(f"GA Agent decision: {decision}, action: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error en predicción GA: {e}")
            return {
                'action': 0,  # HOLD por seguridad
                'size': 0,
                'confidence': 0.0
            }

            # Calcular tamaño sugerido basado en estado del portfolio
            size = self._calculate_position_size(state, market_data)

            result = {
                'action': action,
                'size': size,
                'confidence': confidence
            }

            self.logger.debug(f"GA Agent decision: {decision}, action: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error en predicción GA: {e}")
            return {
                'action': 0,  # HOLD por seguridad
                'size': 0,
                'confidence': 0.0
            }

    def _calculate_position_size(self, state: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """
        Calcular tamaño de posición sugerido.
        """
        balance = state.get('balance', 0)
        price = market_data.get('Close', 0)

        if balance <= 0 or price <= 0:
            return 0

        # Para GA, usar un porcentaje fijo conservador del balance
        max_position_value = balance * 0.05  # Máximo 5% del balance por trade
        size = max_position_value / price

        return max(size, 0)

    def analyze(self):
        """Implementar método abstracto de BaseAgent."""
        return {
            'agent_type': 'GA_Evolutionary',
            'strategy_loaded': self.ga_strategy is not None,
            'status': 'active'
        }


class SimpleTradingAgent(BaseAgent):
    """
    Agente simple de ejemplo para testing.
    Implementa una estrategia básica buy & hold.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hold_period = 20  # Días para mantener posición
        self.current_hold_days = 0

    def get_action(self, state: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estrategia simple: comprar y mantener por hold_period días.
        """
        shares = state.get('shares', 0)
        balance = state.get('balance', 0)
        price = market_data.get('Close', 0)

        if shares > 0:
            # Ya tenemos posición, contar días
            self.current_hold_days += 1
            if self.current_hold_days >= self.hold_period:
                # Vender después del período de hold
                return {
                    'action': 2,  # SELL
                    'size': shares,
                    'confidence': 0.8
                }
            else:
                # Mantener
                return {
                    'action': 0,  # HOLD
                    'size': 0,
                    'confidence': 0.5
                }
        else:
            # No tenemos posición, comprar si hay balance
            if balance > 0 and price > 0:
                size = (balance * 0.5) / price  # Usar 50% del balance
                return {
                    'action': 1,  # BUY
                    'size': size,
                    'confidence': 0.7
                }
            else:
                return {
                    'action': 0,  # HOLD
                    'size': 0,
                    'confidence': 0.5
                }

    def analyze(self):
        """Implementar método abstracto de BaseAgent."""
        return {
            'agent_type': 'Simple_BuyHold',
            'hold_period': self.hold_period,
            'current_hold_days': self.current_hold_days,
            'status': 'active'
        }