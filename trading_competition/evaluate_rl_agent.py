#!/usr/bin/env python3
"""
Script para evaluar el agente RL entrenado.
"""

import os
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

from environments.trading_env import TradingEnv


def load_data(filepath: str) -> pd.DataFrame:
    """Carga los datos procesados con indicadores."""
    print(f"📥 Cargando datos desde: {filepath}")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"✅ Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
    return df


def evaluate_trained_agent(model_path: str, df: pd.DataFrame, num_episodes: int = 10):
    """Evalúa el rendimiento del agente entrenado."""

    print(f"🤖 Cargando modelo desde: {model_path}")
    model = PPO.load(model_path)

    print(f"📊 Evaluando agente en {num_episodes} episodios...")

    episode_rewards = []
    portfolio_values = []
    trades_executed = []
    returns = []

    for episode in range(num_episodes):
        env = TradingEnv(df, initial_balance=10000.0)
        obs, info = env.reset()
        episode_reward = 0
        done = False
        initial_value = env.net_worth  # Acceder directamente al net_worth del entorno

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            if done or truncated:
                break

        final_value = info['net_worth']
        total_return = (final_value - initial_value) / initial_value * 100

        episode_rewards.append(episode_reward)
        portfolio_values.append(final_value)
        trades_executed.append(info['total_trades'])
        returns.append(total_return)

        print(f"Episode {episode + 1}: Reward: {episode_reward:.2f}, "
              f"Final Value: ${final_value:.2f}, Return: {total_return:.2f}%, "
              f"Trades: {info['total_trades']}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_final_value = np.mean(portfolio_values)
    mean_return = np.mean(returns)
    mean_trades = np.mean(trades_executed)

    print("\n📈 Resultados de evaluación:")
    print(f"   Recompensa media: {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"   Valor final medio: ${mean_final_value:.2f}")
    print(f"   Retorno medio: {mean_return:.2f}%")
    print(f"   Trades promedio: {mean_trades:.1f}")

    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_final_value': mean_final_value,
        'mean_return': mean_return,
        'mean_trades': mean_trades
    }


def main():
    """Función principal."""
    print("🚀 Evaluando agente RL entrenado\n")

    # Cargar datos procesados
    data_file = "data/processed/SPY_with_indicators.csv"

    if not os.path.exists(data_file):
        print(f"❌ Archivo de datos no encontrado: {data_file}")
        return

    df = load_data(data_file)

    # Evaluar agente
    model_path = "models/trading_agent.zip"

    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        return

    results = evaluate_trained_agent(model_path, df, num_episodes=20)

    print("\n✅ Evaluación completada!")
    print(f"📈 Retorno medio en evaluación: {results['mean_return']:.2f}%")


if __name__ == "__main__":
    main()