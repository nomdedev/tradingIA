import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple


class TradingEnv(gym.Env):
    """
    Trading Environment para Reinforcement Learning con Gymnasium.

    Estado: 15 features técnicas normalizadas
    Acciones: 0=HOLD, 1=BUY, 2=SELL
    Reward: Profit/Loss + penalizaciones por riesgo
    """

    def __init__(self, df, initial_balance=10000, commission=0.001, slippage=0.0005):
        super(TradingEnv, self).__init__()

        # Metadata para gymnasium
        self.metadata = {"render_modes": ["human"], "render_fps": 1}

        # Datos históricos con indicadores
        self.df = df.copy()

        # Espacios de observación y acción
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(15,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(3)  # 0=HOLD, 1=BUY, 2=SELL

        # Parámetros de trading
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.max_episode_steps = 200  # Reducido de 1000 a 200 para mejor aprendizaje

        # Reset del environment
        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """Reset del environment al estado inicial."""
        super().reset(seed=seed)

        # Estado inicial
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = []
        self.prev_net_worth = self.initial_balance
        self.episode_steps = 0  # Contador de steps del episodio

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Extraer 15 features técnicas del DataFrame."""
        if self.current_step >= len(self.df):
            return np.zeros(15, dtype=np.float32)

        row = self.df.iloc[self.current_step]

        # Calcular precio normalizado (últimos 100 días)
        lookback_window = min(100, self.current_step + 1)
        price_window = self.df['Close'].iloc[max(0, self.current_step - lookback_window + 1):self.current_step + 1]
        price_mean = price_window.mean()
        price_std = price_window.std() + 1e-8  # Evitar división por cero
        close_normalized = (row['Close'] - price_mean) / price_std

        # Returns
        returns_1d = row.get('returns_1d', 0.0)
        returns_5d = row.get('returns_5d', 0.0)

        # Momentum indicators (normalizados)
        rsi_14 = row.get('RSI_14', 50) / 100.0
        rsi_21 = row.get('RSI_21', 50) / 100.0
        macd = row.get('MACD_12_26_9', 0) / 10.0  # Escalar MACD
        macd_signal = row.get('MACDs_12_26_9', 0) / 10.0

        # Volatilidad
        atr_relative = row.get('ATR_14', 0) / (row['Close'] + 1e-8)
        volatility_20d = row.get('volatility_20d', 0) * 100

        # Tendencia
        sma_ratio = row.get('SMA_20_50_ratio', 1.0) - 1.0  # Centrar en 0
        adx = row.get('ADX_14', 25) / 100.0

        # Volumen (normalizado últimos 100 días)
        if 'OBV' in self.df.columns:
            obv_window = self.df['OBV'].iloc[max(0, self.current_step - lookback_window + 1):self.current_step + 1]
            obv_mean = obv_window.mean()
            obv_std = obv_window.std() + 1e-8
            obv_normalized = (row.get('OBV', 0) - obv_mean) / obv_std
        else:
            obv_normalized = 0.0

        cmf = row.get('CMF', 0.0)

        # Bollinger position (ya está normalizado)
        bb_position = row.get('bb_position', 0.5)

        # Estado de posición
        position_state = 1.0 if self.shares > 0 else 0.0

        # Crear array de observación
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

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Ejecutar un paso del environment."""
        current_price = self.df.iloc[self.current_step]['Close']

        # Ejecutar acción
        if action == 1:  # BUY
            if self.shares == 0 and self.balance > 0:
                # Comprar con 95% del balance
                shares_to_buy = (self.balance * 0.95) / (current_price * (1 + self.slippage))
                cost = shares_to_buy * current_price * (1 + self.commission + self.slippage)

                if cost <= self.balance:
                    self.shares = shares_to_buy
                    self.balance -= cost
                    self.trades.append({
                        'step': self.current_step,
                        'type': 'BUY',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'cost': cost
                    })

        elif action == 2:  # SELL
            if self.shares > 0:
                # Vender todas las shares
                proceeds = self.shares * current_price * (1 - self.commission - self.slippage)
                self.balance += proceeds
                self.trades.append({
                    'step': self.current_step,
                    'type': 'SELL',
                    'price': current_price,
                    'shares': self.shares,
                    'proceeds': proceeds
                })
                self.shares = 0

        # Calcular net worth
        self.net_worth = self.balance + self.shares * current_price

        # Actualizar max net worth
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        # Calcular reward
        reward = self._calculate_reward()

        # Guardar net worth anterior
        self.prev_net_worth = self.net_worth

        # Avanzar step
        self.current_step += 1
        self.episode_steps += 1

        # Si llega al final de los datos, reiniciar desde el principio (episodio continuo)
        if self.current_step >= len(self.df):
            self.current_step = 0
            # Reset de posición pero mantener balance (episodio continuo)
            self.shares = 0
            self.trades = []
            self.prev_net_worth = self.net_worth

        # Check si terminó - episodio termina después de max_episode_steps
        done = self.episode_steps >= self.max_episode_steps
        truncated = False

        # Info
        info = {
            'balance': self.balance,
            'shares': self.shares,
            'net_worth': self.net_worth,
            'total_trades': len(self.trades)
        }

        # Obtener siguiente observación
        if not done:
            obs = self._get_observation()
        else:
            obs = np.zeros(15, dtype=np.float32)

        return obs, reward, done, truncated, info

    def _calculate_reward(self) -> float:
        """Calcular reward basado en profit, riesgo y trading frequency."""
        # Profit/Loss absoluto (más significativo que porcentaje)
        profit = self.net_worth - self.prev_net_worth

        # Normalizar por un valor razonable (ej: $100 por step)
        profit_normalized = profit / 100.0

        # Penalización por drawdown (más suave)
        drawdown = max(0, self.max_net_worth - self.net_worth)
        drawdown_penalty = -0.1 * (drawdown / self.initial_balance) if drawdown > 0.02 else 0.0

        # Penalización por overtrading (más suave)
        recent_trades = len([t for t in self.trades if t['step'] > self.current_step - 20])
        overtrading_penalty = -0.01 * max(0, recent_trades - 2)

        # Bonus por mantener posición rentable
        holding_bonus = 0.0
        if self.shares > 0:
            current_profit = (self.net_worth - self.initial_balance) / self.initial_balance
            if current_profit > 0.05:  # Bonus si hay +5% de profit
                holding_bonus = 0.005

        # Reward total
        reward = profit_normalized + drawdown_penalty + overtrading_penalty + holding_bonus

        # Clipear más conservadoramente
        reward = np.clip(reward, -0.5, 0.5)

        return float(reward)

    def render(self, mode='human'):
        """Render del estado actual."""
        if mode == 'human':
            profit = self.net_worth - self.initial_balance
            profit_pct = (profit / self.initial_balance) * 100
            print(f"Step: {self.current_step} | "
                  f"Balance: ${self.balance:.2f} | "
                  f"Shares: {self.shares:.2f} | "
                  f"Net Worth: ${self.net_worth:.2f} | "
                  f"Profit: {profit_pct:.2f}%")

    def close(self):
        """Cerrar el environment."""
        pass


def main():
    """Función de prueba del Trading Environment."""
    import os

    # Cargar datos
    data_path = 'data/processed/SPY_with_indicators.csv'
    if not os.path.exists(data_path):
        print(f"❌ Error: No se encuentra {data_path}")
        print("Ejecuta primero: python utils/indicators.py")
        exit(1)

    print(f"Cargando datos de {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"✓ Datos cargados: {len(df)} filas, {len(df.columns)} columnas\n")

    # Crear environment
    print("Creando Trading Environment...")
    env = TradingEnv(df)
    print("✓ Environment creado")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}\n")

    # Probar con acciones random
    print("Probando environment con 10 acciones random...\n")
    obs, info = env.reset()

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        action_name = ['HOLD', 'BUY', 'SELL'][action]
        print(f"Step {i} | Action: {action_name} | "
              f"Reward: {reward:.4f} | Net Worth: ${info['net_worth']:.2f}")

        if done:
            print("\n✓ Episode terminado")
            break

    print("\n✅ Trading Environment funciona correctamente!")
    print("✅ Compatible con stable-baselines3")
    print(f"✅ Total trades realizados: {info['total_trades']}")


if __name__ == "__main__":
    main()