#!/usr/bin/env python3
"""
Script completo para entrenar agente PPO de Reinforcement Learning para trading.

Este script entrena un agente PPO usando Stable-Baselines3 en el entorno de trading
personalizado, con callbacks avanzados, checkpoints y evaluación automática.
"""

import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
import matplotlib.pyplot as plt

# Import del environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.trading_env import TradingEnv


class TradingCallback(BaseCallback):
    """Callback personalizado para logging durante entrenamiento."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_net_worths = []

    def _on_step(self) -> bool:
        # Capturar info del environment
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]

            # Si el episodio terminó
            if self.locals.get('dones', [False])[0]:
                episode_reward = self.locals.get('rewards', [0])[0]
                net_worth = info.get('net_worth', 10000)

                self.episode_rewards.append(episode_reward)
                self.episode_net_worths.append(net_worth)

                # Log cada 10 episodios
                if len(self.episode_rewards) % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_net_worth = np.mean(self.episode_net_worths[-10:])
                    profit = ((avg_net_worth - 10000) / 10000) * 100

                    print(f"Episode {len(self.episode_rewards)} | "
                          f"Avg Reward: {avg_reward:.4f} | "
                          f"Avg Net Worth: ${avg_net_worth:.2f} | "
                          f"Avg Profit: {profit:.2f}%")

        return True


def prepare_data(data_path='data/processed/SPY_with_indicators.csv'):
    """Cargar y dividir datos en train/val/test"""
    print(f"Cargando datos de {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"[SUCCESS] Datos cargados: {len(df)} filas\n")

    # Split: 60% train, 20% val, 20% test
    train_size = int(len(df) * 0.6)
    val_size = int(len(df) * 0.2)

    df_train = df.iloc[:train_size].copy()
    df_val = df.iloc[train_size:train_size+val_size].copy()
    df_test = df.iloc[train_size+val_size:].copy()

    print("Split de datos:")
    print(f"  Train: {len(df_train)} días ({df_train.index[0]} a {df_train.index[-1]})")
    print(f"  Val:   {len(df_val)} días ({df_val.index[0]} a {df_val.index[-1]})")
    print(f"  Test:  {len(df_test)} días ({df_test.index[0]} a {df_test.index[-1]})\n")

    return df_train, df_val, df_test


def create_env(df):
    """Crear environment wrapped para stable-baselines3"""
    def _init():
        return TradingEnv(df)
    return DummyVecEnv([_init])


def train_ppo_agent(df_train, df_val,
                    total_timesteps=100000,
                    save_path='models/ppo_trading_agent'):
    """Entrenar agente PPO"""

    print("="*60)
    print("ENTRENAMIENTO AGENTE PPO (REINFORCEMENT LEARNING)")
    print("="*60)

    # Crear environments
    print("\n1. Creando environments...")
    train_env = create_env(df_train)
    val_env = create_env(df_val)
    print("[SUCCESS] Environments creados\n")

    # Crear directorios
    os.makedirs('models/checkpoints', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)

    # Configurar callbacks
    print("2. Configurando callbacks...")

    # Callback personalizado
    trading_callback = TradingCallback()

    # Checkpoint cada 10k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='models/checkpoints',
        name_prefix='ppo_checkpoint'
    )

    # Evaluation callback - menos frecuente con episodios más cortos
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path='models/',
        log_path='results/logs/',
        eval_freq=2000,  # Reducido para más evaluaciones
        deterministic=True,
        render=False,
        verbose=1
    )

    print("[SUCCESS] Callbacks configurados\n")

    # Crear modelo PPO
    print("3. Creando modelo PPO...")
    print("Hiperparámetros:")
    print("  - learning_rate: 0.0003")
    print("  - n_steps: 1024 (reducido)")
    print("  - batch_size: 32 (reducido)")
    print("  - n_epochs: 20 (aumentado)")
    print("  - gamma: 0.95 (reducido)")
    print("  - gae_lambda: 0.9 (ajustado)")
    print("  - clip_range: 0.3 (aumentado)")
    print("  - ent_coef: 0.05 (aumentado)")
    print("  - max_episode_steps: 200 (reducido)\n")

    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=0.0003,
        n_steps=1024,  # Reducido para más updates
        batch_size=32,  # Reducido para más updates
        n_epochs=20,    # Aumentado para mejor aprendizaje
        gamma=0.95,     # Reducido para foco a corto plazo
        gae_lambda=0.9, # Ajustado
        clip_range=0.3, # Aumentado para más exploración
        ent_coef=0.05,  # Aumentado para más exploración
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log='results/logs/tensorboard/'
    )

    print("[SUCCESS] Modelo PPO creado\n")

    # Entrenar
    print(f"4. Entrenando agente por {total_timesteps} timesteps...")
    print("   (Esto puede tomar 30-60 minutos)\n")
    print("="*60)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[trading_callback, checkpoint_callback, eval_callback],
        progress_bar=True
    )

    print("\n" + "="*60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*60)

    # Guardar modelo final
    print(f"\n5. Guardando modelo en {save_path}...")
    model.save(save_path)
    print("[SUCCESS] Modelo guardado\n")

    # Guardar estadísticas
    stats = {
        'episode_rewards': trading_callback.episode_rewards,
        'episode_net_worths': trading_callback.episode_net_worths
    }

    return model, stats


def plot_training_results(stats, save_path='results/figures/training_progress.png'):
    """Graficar progreso del entrenamiento"""

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Episode Rewards
    ax1 = axes[0]
    ax1.plot(stats['episode_rewards'], alpha=0.3, color='blue', label='Episode Reward')

    # Moving average
    if len(stats['episode_rewards']) > 10:
        ma = np.convolve(stats['episode_rewards'], np.ones(10)/10, mode='valid')
        ax1.plot(range(9, len(stats['episode_rewards'])), ma,
                color='darkblue', linewidth=2, label='Moving Avg (10 eps)')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Progress - Episode Rewards')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Net Worth
    ax2 = axes[1]
    ax2.plot(stats['episode_net_worths'], alpha=0.3, color='green', label='Episode Net Worth')
    ax2.axhline(y=10000, color='red', linestyle='--', label='Initial Balance')

    # Moving average
    if len(stats['episode_net_worths']) > 10:
        ma = np.convolve(stats['episode_net_worths'], np.ones(10)/10, mode='valid')
        ax2.plot(range(9, len(stats['episode_net_worths'])), ma,
                color='darkgreen', linewidth=2, label='Moving Avg (10 eps)')

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Net Worth ($)')
    ax2.set_title('Training Progress - Portfolio Value')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    # Guardar
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Grafico guardado en {save_path}\n")

    plt.close()


def main():
    print("\n" + "="*60)
    print("ENTRENAMIENTO DE AGENTE RL (PPO) PARA TRADING")
    print("="*60 + "\n")

    # Preparar datos
    df_train, df_val, df_test = prepare_data()

    # Entrenar
    model, stats = train_ppo_agent(
        df_train,
        df_val,
        total_timesteps=100000,  # Puedes reducir a 50000 para pruebas más rápidas
        save_path='models/ppo_trading_agent'
    )

    # Graficar resultados
    print("6. Generando gráficos de entrenamiento...")
    plot_training_results(stats)

    # Resumen final
    print("="*60)
    print("RESUMEN DEL ENTRENAMIENTO")
    print("="*60)
    print(f"Total episodios: {len(stats['episode_rewards'])}")
    print(f"Reward promedio final (últimos 10): {np.mean(stats['episode_rewards'][-10:]):.4f}")
    print(f"Net worth promedio final: ${np.mean(stats['episode_net_worths'][-10:]):.2f}")
    final_profit = ((np.mean(stats['episode_net_worths'][-10:]) - 10000) / 10000) * 100
    print(f"Profit promedio: {final_profit:.2f}%")
    print("="*60)

    print("\n✅ Proceso completado exitosamente!")
    print("\nArchivos generados:")
    print("  - models/ppo_trading_agent.zip (modelo entrenado)")
    print("  - models/checkpoints/ (checkpoints intermedios)")
    print("  - results/figures/training_progress.png (gráficos)")
    print("  - results/logs/ (logs de tensorboard)")

    print("\nPróximos pasos:")
    print("  1. Revisar gráficos de entrenamiento")
    print("  2. Entrenar agente GA (Genetic Algorithm)")
    print("  3. Competir ambos agentes en datos de test")


if __name__ == "__main__":
    main()