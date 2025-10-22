#!/usr/bin/env python3
"""
Agente de Reinforcement Learning para trading usando Stable-Baselines3.
Entrena un agente PPO en el entorno de trading personalizado.
"""

import os
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

from environments.trading_env import TradingEnv


class TradingCallback(BaseCallback):
    """Callback personalizado para monitorear el entrenamiento."""

    def __init__(self, eval_freq=5000, verbose=1):
        super(TradingCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluar rendimiento cada eval_freq steps
            # Por simplicidad, solo registramos las métricas del entrenamiento
            # La evaluación completa se hará al final
            print(f"Step {self.n_calls}: Training in progress...")

        return True


def load_data(filepath: str) -> pd.DataFrame:
    """Carga los datos procesados con indicadores."""
    print(f"📥 Cargando datos desde: {filepath}")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"✅ Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
    return df


def create_and_train_agent(df: pd.DataFrame, total_timesteps: int = 50000, model_path: str = "models/trading_agent"):
    """Crea y entrena el agente RL."""

    print("🏗️ Creando entorno de trading...")

    # Crear directorios necesarios
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Crear entorno
    env = TradingEnv(df, initial_balance=10000.0)

    # Wrap con Monitor para logging
    monitor_env = Monitor(env, filename="results/logs/rl_training.log", allow_early_resets=True)

    # Wrap con VecEnv para SB3
    vec_env = DummyVecEnv([lambda: monitor_env])

    print("🤖 Creando agente PPO...")

    # Crear modelo PPO con hiperparámetros optimizados para trading
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=512,  # Reducido para episodios más cortos
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,  # Pequeña entropía para exploración
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="results/logs/tensorboard",
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64])  # Red neuronal más pequeña
        ),
        verbose=1,
        seed=42,
        device="auto",
        _init_setup_model=True,
    )

    print("🚀 Iniciando entrenamiento...")

    # Callback para monitoreo
    callback = TradingCallback(eval_freq=5000)

    # Entrenar modelo
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    print("💾 Guardando modelo entrenado...")

    # Guardar modelo
    model.save(model_path)
    print(f"✅ Modelo guardado en: {model_path}.zip")

    # Graficar progreso de entrenamiento
    plot_training_progress(callback)

    return model, vec_env


def plot_training_progress(callback):
    """Grafica el progreso del entrenamiento."""
    if len(callback.episode_rewards) > 0:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(callback.episode_rewards)
        plt.title('Recompensa Media por Evaluación')
        plt.xlabel('Evaluación')
        plt.ylabel('Recompensa')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(callback.episode_lengths)
        plt.title('Longitud Media de Episodio')
        plt.xlabel('Evaluación')
        plt.ylabel('Pasos')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('results/logs/training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("📊 Gráfico de progreso guardado en: results/logs/training_progress.png")


def evaluate_agent(model, df: pd.DataFrame, num_episodes: int = 10):
    """Evalúa el rendimiento del agente entrenado."""

    print(f"📊 Evaluando agente en {num_episodes} episodios...")

    episode_rewards = []
    portfolio_values = []
    trades_executed = []

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

        print(f"Episode {episode + 1}: Reward: {episode_reward:.2f}, "
              f"Final Value: ${final_value:.2f}, Return: {total_return:.2f}%, "
              f"Trades: {info['total_trades']}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_final_value = np.mean(portfolio_values)
    mean_return = np.mean([(v - 10000) / 10000 * 100 for v in portfolio_values])
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
    print("🚀 Iniciando entrenamiento del agente RL para trading\n")

    # Cargar datos procesados
    data_file = "data/processed/SPY_with_indicators.csv"

    if not os.path.exists(data_file):
        print(f"❌ Archivo de datos no encontrado: {data_file}")
        print("💡 Ejecuta primero utils/indicators.py para procesar los datos")
        return

    df = load_data(data_file)

    # Entrenar agente
    print("🎯 Configuración de entrenamiento:")
    print("   Timesteps totales: 50,000")
    print(f"   Datos de entrenamiento: {len(df)} filas")
    print()

    model, env = create_and_train_agent(
        df,
        total_timesteps=50000,
        model_path="models/trading_agent"
    )

    # Evaluar agente
    results = evaluate_agent(model, df, num_episodes=10)

    print("\n✅ Entrenamiento completado!")
    print("📁 Modelo guardado en: models/trading_agent.zip")
    print("📊 Logs guardados en: results/logs/")
    print(f"📈 Retorno medio en evaluación: {results['mean_return']:.2f}%")


if __name__ == "__main__":
    main()