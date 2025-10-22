#!/usr/bin/env python3
"""
Sistema de competición entre agentes RL y GA para trading.

Este script compara el rendimiento de ambos agentes en datos de test,
calcula métricas detalladas y declara un ganador.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from environments.trading_env import TradingEnv

# Import del GA
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))
from train_ga_agent import GAStrategy


class TradingCompetition:
    """Sistema de competición entre agentes de trading"""

    def __init__(self):
        self.results = {}
        self.metrics = {}

    def load_agents(self):
        """Cargar ambos agentes entrenados"""
        print("Cargando agentes...")

        # Cargar agente RL
        rl_path = 'models/ppo_trading_agent.zip'
        if os.path.exists(rl_path):
            self.rl_agent = PPO.load(rl_path)
            print("✓ Agente RL cargado")
        else:
            print("❌ Agente RL no encontrado")
            self.rl_agent = None

        # Cargar agente GA
        ga_path = 'models/ga_best_individual.pkl'
        if os.path.exists(ga_path):
            with open(ga_path, 'rb') as f:
                self.ga_agent = pickle.load(f)
            print("✓ Agente GA cargado")
        else:
            print("❌ Agente GA no encontrado")
            self.ga_agent = None

    def load_test_data(self):
        """Cargar datos de test (últimos 20%)"""
        data_path = 'data/processed/SPY_with_indicators.csv'
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)

        # Test: últimos 20%
        test_size = int(len(df) * 0.2)
        self.df_test = df.iloc[-test_size:].copy()

        print(f"✓ Datos de test cargados: {len(self.df_test)} días")
        print(f"   Periodo: {self.df_test.index[0]} a {self.df_test.index[-1]}")

    def evaluate_agent(self, agent_type, agent, df):
        """
        Evaluar un agente en datos dados

        Args:
            agent_type: 'RL' o 'GA'
            agent: modelo del agente
            df: DataFrame con datos

        Returns:
            dict: resultados detallados
        """
        print(f"\nEvaluando agente {agent_type}...")

        # Crear environment
        env = TradingEnv(df)

        balance = 10000
        shares = 0
        trades = []
        portfolio_values = [10000]
        entry_price = 0
        entry_step = 0

        # Para GA: crear estrategia
        if agent_type == 'GA':
            strategy = GAStrategy(agent)

        # Ejecutar trades
        for i in range(len(df)):
            current_price = df.iloc[i]['Close']

            if agent_type == 'RL':
                # Agente RL
                obs, _ = env.reset()
                env.current_step = i  # Forzar step específico
                env._get_observation()  # Actualizar observación

                action, _ = agent.predict(obs, deterministic=True)

            else:
                # Agente GA
                position_info = {
                    'has_position': shares > 0,
                    'entry_price': entry_price,
                    'entry_step': entry_step,
                    'current_step': i
                }

                action = strategy.decide(df.iloc[i], position_info)

            # Ejecutar acción
            if action == 1 and shares == 0 and balance > 0:  # BUY
                # Usar 95% del balance
                cost_per_share = current_price * 1.001
                shares_to_buy = (balance * 0.95) / cost_per_share
                cost = shares_to_buy * cost_per_share

                if cost <= balance:
                    shares = shares_to_buy
                    balance -= cost
                    entry_price = current_price
                    entry_step = i
                    trades.append({
                        'type': 'BUY',
                        'price': current_price,
                        'step': i,
                        'date': df.index[i]
                    })

            elif action == 2 and shares > 0:  # SELL
                proceeds = shares * current_price * 0.999
                balance += proceeds

                profit_pct = (current_price - entry_price) / entry_price
                trades.append({
                    'type': 'SELL',
                    'price': current_price,
                    'profit_pct': profit_pct,
                    'step': i,
                    'date': df.index[i]
                })
                shares = 0

            # Calcular valor de portfolio
            portfolio_value = balance + shares * current_price
            portfolio_values.append(portfolio_value)

        # Cerrar posición final
        if shares > 0:
            final_price = df.iloc[-1]['Close']
            balance += shares * final_price * 0.999
            shares = 0

        # Calcular métricas
        final_value = balance
        total_return = (final_value - 10000) / 10000 * 100

        # Retornos diarios
        daily_returns = []
        for i in range(1, len(portfolio_values)):
            ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            daily_returns.append(ret)

        # Sharpe Ratio (anualizado)
        if len(daily_returns) > 1:
            avg_daily_return = np.mean(daily_returns)
            std_daily_return = np.std(daily_returns)
            sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0
        else:
            sharpe_ratio = 0

        # Max Drawdown
        peak = portfolio_values[0]
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        # Win Rate
        winning_trades = [t for t in trades if t.get('profit_pct', 0) > 0]
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

        # Profit Factor
        gross_profit = sum(t.get('profit_pct', 0) for t in trades if t.get('profit_pct', 0) > 0)
        gross_loss = abs(sum(t.get('profit_pct', 0) for t in trades if t.get('profit_pct', 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        results = {
            'agent_type': agent_type,
            'final_value': final_value,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'portfolio_values': portfolio_values,
            'trades': trades,
            'daily_returns': daily_returns
        }

        print(f"  Retorno total: {total_return:.2f}%")
        print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {max_drawdown*100:.2f}%")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Profit Factor: {profit_factor:.2f}")
        print(f"  Total trades: {len(trades)}")

        return results

    def run_competition(self):
        """Ejecutar competición completa"""
        print("="*70)
        print("🏆 COMPETICIÓN DE AGENTES DE TRADING")
        print("="*70)

        # Cargar componentes
        self.load_agents()
        self.load_test_data()

        # Evaluar agentes
        if self.rl_agent:
            self.results['RL'] = self.evaluate_agent('RL', self.rl_agent, self.df_test)

        if self.ga_agent:
            self.results['GA'] = self.evaluate_agent('GA', self.ga_agent, self.df_test)

        # Determinar ganador
        self.determine_winner()

        # Generar reportes
        self.generate_report()
        self.plot_comparison()

        print("\n" + "="*70)
        print("✅ COMPETICIÓN COMPLETADA")
        print("="*70)

    def determine_winner(self):
        """Determinar el agente ganador basado en múltiples métricas"""
        if len(self.results) < 2:
            print("❌ Se necesitan ambos agentes para competir")
            return

        rl_metrics = self.results['RL']
        ga_metrics = self.results['GA']

        # Sistema de puntuación (0-5 puntos por métrica)
        rl_score = 0
        ga_score = 0

        metrics_weights = {
            'sharpe_ratio': 2,      # Más importante
            'total_return_pct': 2,  # Más importante
            'max_drawdown_pct': 1,  # Menos importante (inverso)
        }

        print("\n📊 ANALISIS DE METRICAS:")
        print("-" * 50)

        for metric, weight in metrics_weights.items():
            rl_val = rl_metrics[metric]
            ga_val = ga_metrics[metric]

            if metric == 'max_drawdown_pct':
                # Menor drawdown es mejor
                rl_better = rl_val < ga_val
            else:
                # Mayor valor es mejor
                rl_better = rl_val > ga_val

            if rl_better:
                rl_score += weight
                winner = 'RL'
            else:
                ga_score += weight
                winner = 'GA'

            print(f"{metric}: RL={rl_val:.2f}, GA={ga_val:.2f} → {winner} gana")

        # Ganador final
        if rl_score > ga_score:
            self.winner = 'RL'
            margin = rl_score - ga_score
        elif ga_score > rl_score:
            self.winner = 'GA'
            margin = ga_score - rl_score
        else:
            self.winner = 'EMPATE'
            margin = 0

        print(f"\n🏆 GANADOR: {self.winner}")
        if self.winner != 'EMPATE':
            print(f"   Margen de victoria: {margin} puntos")

    def generate_report(self):
        """Generar reporte detallado"""
        report_path = 'results/competition_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("🏆 REPORTE DE COMPETICIÓN DE AGENTES DE TRADING\n")
            f.write("="*70 + "\n\n")

            f.write(f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Periodo de test: {self.df_test.index[0]} a {self.df_test.index[-1]}\n")
            f.write(f"Días de test: {len(self.df_test)}\n\n")

            for agent_type, result in self.results.items():
                f.write(f"🤖 AGENTE {agent_type}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Retorno Total: {result['total_return_pct']:.2f}%\n")
                f.write(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}\n")
                f.write(f"Max Drawdown: {result['max_drawdown_pct']:.2f}%\n")
                f.write(f"Win Rate: {result['win_rate_pct']:.1f}%\n")
                f.write(f"Profit Factor: {result['profit_factor']:.2f}\n")
                f.write(f"Total Trades: {result['total_trades']}\n\n")

            f.write(f"🏆 GANADOR: {self.winner}\n")
            f.write("="*70 + "\n")

        print(f"✓ Reporte guardado en {report_path}")

    def plot_comparison(self):
        """Generar gráfico comparativo"""
        if len(self.results) < 2:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🤖 Comparación de Agentes de Trading', fontsize=16, fontweight='bold')

        # 1. Portfolio Value
        ax1 = axes[0, 0]
        for agent_type, result in self.results.items():
            values = result['portfolio_values']
            ax1.plot(values, label=f'{agent_type} (${values[-1]:.0f})', linewidth=2)
        ax1.set_title('Valor del Portfolio')
        ax1.set_xlabel('Días')
        ax1.set_ylabel('Valor ($)')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. Cumulative Returns
        ax2 = axes[0, 1]
        for agent_type, result in self.results.items():
            returns = [(v - 10000) / 10000 * 100 for v in result['portfolio_values']]
            ax2.plot(returns, label=f'{agent_type} ({returns[-1]:.1f}%)', linewidth=2)
        ax2.set_title('Retorno Acumulado')
        ax2.set_xlabel('Días')
        ax2.set_ylabel('Retorno (%)')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # 3. Sharpe Ratio Comparison
        ax3 = axes[1, 0]
        agents = list(self.results.keys())
        sharpe_ratios = [self.results[a]['sharpe_ratio'] for a in agents]
        bars = ax3.bar(agents, sharpe_ratios, color=['blue', 'green'])
        ax3.set_title('Sharpe Ratio')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.grid(alpha=0.3, axis='y')

        # Agregar valores sobre las barras
        for bar, value in zip(bars, sharpe_ratios):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

        # 4. Key Metrics Comparison
        ax4 = axes[1, 1]
        metrics = ['total_return_pct', 'win_rate_pct', 'max_drawdown_pct']
        metric_names = ['Retorno Total (%)', 'Win Rate (%)', 'Max Drawdown (%)']

        x = np.arange(len(metrics))
        width = 0.35

        rl_values = [self.results['RL'][m] for m in metrics]
        ga_values = [self.results['GA'][m] for m in metrics]

        ax4.bar(x - width/2, rl_values, width, label='RL', color='blue', alpha=0.8)
        ax4.bar(x + width/2, ga_values, width, label='GA', color='green', alpha=0.8)

        ax4.set_title('Métricas Clave')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metric_names, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(alpha=0.3, axis='y')

        # Agregar valores sobre las barras
        for i, (rl_val, ga_val) in enumerate(zip(rl_values, ga_values)):
            ax4.text(i - width/2, rl_val + max(rl_values + ga_values) * 0.02,
                    f'{rl_val:.1f}', ha='center', va='bottom', fontsize=9)
            ax4.text(i + width/2, ga_val + max(rl_values + ga_values) * 0.02,
                    f'{ga_val:.1f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # Guardar
        plot_path = 'results/figures/competition_comparison.png'
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico comparativo guardado en {plot_path}")
        plt.close()


def main():
    """Funcion principal"""
    competition = TradingCompetition()
    competition.run_competition()


if __name__ == "__main__":
    main()