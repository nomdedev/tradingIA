#!/usr/bin/env python3
"""
Script de diagnóstico para entender el problema con los rewards del agente RL.
"""

import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from environments.trading_env import TradingEnv

def diagnose_environment():
    """Diagnosticar el comportamiento del entorno con acciones específicas."""
    print("Diagnóstico del Trading Environment")
    print("="*50)

    # Cargar datos
    data_path = 'data/processed/SPY_with_indicators.csv'
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df_val = df.iloc[int(len(df)*0.6):int(len(df)*0.8)].copy()

    # Crear entorno
    env = TradingEnv(df_val)

    print(f"Datos de validación: {len(df_val)} filas")
    print(f"Balance inicial: ${env.initial_balance}")
    print(f"Comisión: {env.commission}, Slippage: {env.slippage}")
    print()

    # Probar diferentes estrategias
    strategies = {
        'HOLD': [0] * 100,  # Solo HOLD
        'BUY_HOLD': [1] + [0] * 99,  # Comprar al inicio y mantener
        'RANDOM': np.random.randint(0, 3, 100),  # Acciones random
    }

    for strategy_name, actions in strategies.items():
        print(f"Estrategia: {strategy_name}")
        obs, info = env.reset()
        total_reward = 0
        rewards = []

        for action in actions:
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            rewards.append(reward)

            if done:
                break

        final_net_worth = info['net_worth']
        profit_pct = ((final_net_worth - env.initial_balance) / env.initial_balance) * 100

        print(f"  Reward total: {total_reward:.4f}")
        print(f"  Net worth final: ${final_net_worth:.2f}")
        print(f"  Profit: {profit_pct:.2f}%")
        print(f"  Trades realizados: {info['total_trades']}")
        print()

def diagnose_trained_agent():
    """Diagnosticar el comportamiento del agente entrenado."""
    print("Diagnóstico del Agente Entrenado")
    print("="*50)

    # Cargar modelo
    model_path = 'models/ppo_trading_agent.zip'
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        return

    model = PPO.load(model_path)
    print("✓ Modelo cargado")

    # Cargar datos de validación
    data_path = 'data/processed/SPY_with_indicators.csv'
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df_val = df.iloc[int(len(df)*0.6):int(len(df)*0.8)].copy()

    # Crear entorno
    env = TradingEnv(df_val)

    print(f"Datos de validación: {len(df_val)} filas")
    print()

    # Evaluar agente
    episodes = 5
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        print(f"Episodio {ep+1}:")

        while not done and step_count < 100:  # Limitar a 100 steps para debug
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            total_reward += reward
            step_count += 1

            # Log cada 10 steps
            if step_count % 10 == 0:
                print(f"  Step {step_count}: Action={action}, Reward={reward:.4f}, Net Worth=${info['net_worth']:.2f}")

        profit_pct = ((info['net_worth'] - env.initial_balance) / env.initial_balance) * 100
        print(f"  Total Reward: {total_reward:.4f}")
        print(f"  Net Worth Final: ${info['net_worth']:.2f}")
        print(f"  Profit: {profit_pct:.2f}%")
        print(f"  Trades: {info['total_trades']}")
        print()

def main():
    print("Diagnóstico del Sistema de Trading RL")
    print("="*60)

    diagnose_environment()
    print()
    diagnose_trained_agent()

if __name__ == "__main__":
    main()