#!/usr/bin/env python3
"""
COMPETENCIA FINAL: RL vs GA en Trading con Moon Dev Risk Management

Script completo para comparar el rendimiento de agentes RL (PPO) y GA (Genetic Algorithm)
envueltos en SafeTradingWrapper con gestión de riesgo profesional estilo Moon Dev AI Agents.

Características:
- Agentes envueltos en SafeTradingWrapper para control de riesgo
- Validación de trades antes de ejecución
- Stop losses dinámicos y position sizing
- Logging detallado de decisiones de riesgo
- Comparación justa con y sin risk management

Autor: Sistema de Trading IA + Moon Dev AI Agents
Fecha: 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Agregar directorios al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'trading_competition'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

from environments.trading_env import TradingEnv

# Importar Moon Dev AI Agents
from safe_trading_wrapper import SafeTradingWrapper
from agent_adapters import RLAgentAdapter, GAAgentAdapter

# Importar clases necesarias para pickle
from agents.train_ga_agent import GAStrategy


def load_models(use_risk_management=True):
    """Cargar modelos entrenados y opcionalmente envolverlos con risk management"""
    print("🤖 Cargando modelos entrenados con adaptadores Moon Dev...")

    # Configuración de riesgo conservadora para SPY
    risk_config = {
        'max_drawdown': 0.12,  # 12% max drawdown (conservador para stocks)
        'max_portfolio_heat': 0.25,  # 25% portfolio heat
        'max_risk_per_trade': 0.04,  # 4% risk per trade
        'volatility_lookback': 15,
        'initial_balance': 10000.0
    }

    if use_risk_management:
        print("🛡️ Creando agentes con Moon Dev Risk Management...")

        # Crear adaptadores con risk management
        rl_agent = RLAgentAdapter(
            model_path='models/ppo_trading_agent.zip',
            initial_balance=risk_config['initial_balance'],
            name="RL_PPO_Adapter"
        )

        ga_agent = GAAgentAdapter(
            model_path='models/ga_advanced_strategy.pkl',
            initial_balance=risk_config['initial_balance'],
            name="GA_Evolutionary_Adapter"
        )

        # Envolver con SafeTradingWrapper
        rl_wrapped = SafeTradingWrapper(
            trading_agent=rl_agent,
            risk_config=risk_config,
            fallback_mode=True,
            name="RL_with_MoonDev_Risk"
        )

        ga_wrapped = SafeTradingWrapper(
            trading_agent=ga_agent,
            risk_config=risk_config,
            fallback_mode=True,
            name="GA_with_MoonDev_Risk"
        )

        print("✅ Agentes RL y GA envueltos con gestión de riesgo")
        return rl_wrapped, ga_wrapped

    else:
        print("⚠️ Creando agentes SIN gestión de riesgo (solo para comparación)...")

        # Crear adaptadores sin risk management
        rl_agent = RLAgentAdapter(
            model_path='models/ppo_trading_agent.zip',
            initial_balance=risk_config['initial_balance'],
            name="RL_PPO_Adapter"
        )

        ga_agent = GAAgentAdapter(
            model_path='models/ga_advanced_strategy.pkl',
            initial_balance=risk_config['initial_balance'],
            name="GA_Evolutionary_Adapter"
        )

        print("✅ Agentes RL y GA creados sin gestión de riesgo")
        return rl_agent, ga_agent


def backtest_wrapped_agent(wrapped_agent, df, initial_balance=10000, agent_name="Wrapped Agent"):
    """Backtest de un agente envuelto en SafeTradingWrapper"""

    balance = initial_balance
    shares = 0
    net_worth = initial_balance
    history = []

    # Estado del portfolio para el wrapper
    portfolio_state = {
        'balance': balance,
        'shares': shares,
        'net_worth': net_worth,
        'current_drawdown': 0.0,
        'peak_value': initial_balance
    }

    wrapped_agent.reset()  # Resetear el wrapper

    for i in range(len(df)):
        row = df.iloc[i]

        # Preparar market data
        market_data = {
            'Open': row['Open'],
            'High': row['High'],
            'Low': row['Low'],
            'Close': row['Close'],
            'Volume': row.get('Volume', 0),
            'ATR': row.get('ATR', row['High'] - row['Low'])  # Fallback ATR
        }

        # Agregar indicadores técnicos si existen
        for col in df.columns:
            if col not in market_data:
                market_data[col] = row[col]

        # Wrapper decide acción con risk management
        action_dict = wrapped_agent.get_action(portfolio_state, market_data)

        action = action_dict['action']
        size = action_dict.get('size', 0)
        risk_approved = action_dict.get('risk_approved', True)
        risk_reason = action_dict.get('risk_reason', 'Approved')

        current_price = row['Close']

        # Ejecutar acción si fue aprobada por risk management
        if risk_approved:
            if action == 1 and balance > 0:  # BUY
                cost_per_share = current_price * 1.001  # Spread
                shares_to_buy = min(size, balance / cost_per_share)
                cost = shares_to_buy * cost_per_share

                if cost <= balance:
                    shares += shares_to_buy
                    balance -= cost

            elif action == 2 and shares > 0:  # SELL
                shares_to_sell = min(size, shares) if size > 0 else shares
                proceeds = shares_to_sell * current_price * 0.999  # Spread
                balance += proceeds
                shares -= shares_to_sell

        # Actualizar net worth
        net_worth = balance + shares * current_price

        # Actualizar portfolio state para el wrapper
        portfolio_state.update({
            'balance': balance,
            'shares': shares,
            'net_worth': net_worth
        })

        # Calcular drawdown
        if net_worth > portfolio_state['peak_value']:
            portfolio_state['peak_value'] = net_worth
        portfolio_state['current_drawdown'] = (portfolio_state['peak_value'] - net_worth) / portfolio_state['peak_value']

        # Guardar historial
        history.append({
            'step': i,
            'action': action,
            'size': size,
            'balance': balance,
            'shares': shares,
            'net_worth': net_worth,
            'risk_approved': risk_approved,
            'risk_reason': risk_reason,
            'stop_loss': action_dict.get('stop_loss'),
            'take_profit': action_dict.get('take_profit'),
            'risk_score': action_dict.get('risk_score', 0)
        })

    return pd.DataFrame(history)


def backtest_agent(agent, df, initial_balance=10000, agent_name="Trading Agent"):
    """
    Backtest genérico para cualquier agente (wrapped o puro).
    Maneja tanto SafeTradingWrapper como agentes directos.
    """

    # Si es un wrapper, usar su lógica especial
    if isinstance(agent, SafeTradingWrapper):
        return backtest_wrapped_agent(agent, df, initial_balance, agent_name)

    # Para agentes directos (adaptadores), usar lógica simplificada
    balance = initial_balance
    shares = 0
    net_worth = initial_balance
    history = []

    # Estado del portfolio
    portfolio_state = {
        'balance': balance,
        'shares': shares,
        'net_worth': net_worth,
        'current_drawdown': 0.0,
        'peak_value': initial_balance
    }

    agent.reset()  # Resetear agente si tiene método reset

    for i in range(len(df)):
        row = df.iloc[i]

        # Preparar market data
        market_data = {
            'Open': row['Open'],
            'High': row['High'],
            'Low': row['Low'],
            'Close': row['Close'],
            'Volume': row.get('Volume', 0),
            'ATR': row.get('ATR', row['High'] - row['Low'])
        }

        # Agregar indicadores técnicos si existen
        for col in df.columns:
            if col not in market_data:
                market_data[col] = row[col]

        # Agente decide acción
        action_dict = agent.get_action(portfolio_state, market_data)

        action = action_dict['action']
        size = action_dict.get('size', 0)

        current_price = row['Close']

        # Ejecutar acción
        if action == 1 and balance > 0:  # BUY
            cost_per_share = current_price * 1.001  # Spread
            shares_to_buy = min(size, balance / cost_per_share)
            cost = shares_to_buy * cost_per_share

            if cost <= balance:
                shares += shares_to_buy
                balance -= cost
                net_worth = balance + (shares * current_price)

        elif action == 2 and shares > 0:  # SELL
            shares_to_sell = min(size, shares)
            revenue = shares_to_sell * current_price * 0.999  # Spread

            shares -= shares_to_sell
            balance += revenue
            net_worth = balance + (shares * current_price)

        # Actualizar estado del portfolio
        portfolio_state.update({
            'balance': balance,
            'shares': shares,
            'net_worth': net_worth
        })

        # Calcular drawdown
        portfolio_state['peak_value'] = max(portfolio_state['peak_value'], net_worth)
        portfolio_state['current_drawdown'] = (portfolio_state['peak_value'] - net_worth) / portfolio_state['peak_value']

        # Guardar historial
        history.append({
            'step': i,
            'action': action,
            'balance': balance,
            'shares': shares,
            'net_worth': net_worth,
            'risk_approved': True,  # No risk management aplicado
            'risk_reason': 'Direct agent execution',
            'stop_loss': None,
            'take_profit': None,
            'risk_score': 0.0
        })

    return pd.DataFrame(history)


def calculate_metrics(history_df, initial_balance=10000):
    """Calcular métricas de performance"""

    final_value = history_df['net_worth'].iloc[-1]

    # Return total
    total_return = ((final_value - initial_balance) / initial_balance) * 100

    # Sharpe Ratio
    returns = history_df['net_worth'].pct_change().dropna()
    if len(returns) > 0 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0

    # Maximum Drawdown
    cumulative = history_df['net_worth']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    # Win Rate (basado en acciones)
    trades = history_df[history_df['action'].isin([1, 2])]
    if len(trades) > 0:
        # Simplificado: contar trades que aumentaron net worth
        trade_returns = history_df['net_worth'].pct_change()
        wins = (trade_returns > 0).sum()
        total_trades = len(trade_returns)
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    else:
        win_rate = 0

    # Total trades
    buy_actions = (history_df['action'] == 1).sum()
    sell_actions = (history_df['action'] == 2).sum()
    total_trades = buy_actions + sell_actions

    # Calmar Ratio
    calmar = abs(total_return / max_drawdown) if max_drawdown != 0 else 0

    return {
        'Total Return (%)': total_return,
        'Sharpe Ratio': sharpe,
        'Max Drawdown (%)': max_drawdown,
        'Win Rate (%)': win_rate,
        'Total Trades': total_trades,
        'Calmar Ratio': calmar,
        'Final Value ($)': final_value
    }


def plot_comparison(rl_history, ga_history, rl_metrics, ga_metrics,
                   df_test, save_path='results/figures/competition_results.png'):
    """Crear visualización completa de la competencia"""

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Equity Curves
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(rl_history['step'], rl_history['net_worth'],
            'b-', linewidth=2, label='RL (PPO)', alpha=0.8)
    ax1.plot(ga_history['step'], ga_history['net_worth'],
            'r-', linewidth=2, label='GA (Genetic)', alpha=0.8)
    ax1.axhline(y=10000, color='gray', linestyle='--',
               linewidth=1, label='Initial Balance')
    ax1.set_xlabel('Trading Days', fontsize=11)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax1.set_title('Portfolio Performance Over Time', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Plot 2: Drawdown Comparison
    ax2 = fig.add_subplot(gs[1, 0])
    rl_cummax = rl_history['net_worth'].expanding().max()
    rl_dd = ((rl_history['net_worth'] - rl_cummax) / rl_cummax) * 100
    ga_cummax = ga_history['net_worth'].expanding().max()
    ga_dd = ((ga_history['net_worth'] - ga_cummax) / ga_cummax) * 100

    ax2.fill_between(rl_history['step'], rl_dd, 0, alpha=0.3, color='blue', label='RL')
    ax2.fill_between(ga_history['step'], ga_dd, 0, alpha=0.3, color='red', label='GA')
    ax2.set_xlabel('Trading Days', fontsize=10)
    ax2.set_ylabel('Drawdown (%)', fontsize=10)
    ax2.set_title('Drawdown Over Time', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Plot 3: Actions Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    rl_actions = rl_history['action'].value_counts().sort_index()
    ga_actions = ga_history['action'].value_counts().sort_index()

    x = np.arange(3)
    width = 0.35
    ax3.bar(x - width/2, [rl_actions.get(i, 0) for i in range(3)],
           width, label='RL', color='blue', alpha=0.7)
    ax3.bar(x + width/2, [ga_actions.get(i, 0) for i in range(3)],
           width, label='GA', color='red', alpha=0.7)
    ax3.set_xlabel('Action', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Action Distribution', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['HOLD', 'BUY', 'SELL'])
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3, axis='y')

    # Plot 4: Metrics Comparison (Radar Chart simplificado como bars)
    ax4 = fig.add_subplot(gs[2, :])

    metrics_names = ['Return (%)', 'Sharpe Ratio', 'Win Rate (%)', 'Calmar Ratio']
    rl_values = [
        rl_metrics['Total Return (%)'],
        rl_metrics['Sharpe Ratio'] * 10,  # Escalar para visualización
        rl_metrics['Win Rate (%)'],
        rl_metrics['Calmar Ratio'] * 10
    ]
    ga_values = [
        ga_metrics['Total Return (%)'],
        ga_metrics['Sharpe Ratio'] * 10,
        ga_metrics['Win Rate (%)'],
        ga_metrics['Calmar Ratio'] * 10
    ]

    x = np.arange(len(metrics_names))
    width = 0.35
    ax4.bar(x - width/2, rl_values, width, label='RL', color='blue', alpha=0.7)
    ax4.bar(x + width/2, ga_values, width, label='GA', color='red', alpha=0.7)
    ax4.set_ylabel('Value (scaled)', fontsize=10)
    ax4.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_names, rotation=15, ha='right')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3, axis='y')

    plt.suptitle('RL vs GA Trading Competition Results',
                fontsize=16, fontweight='bold', y=0.995)

    # Guardar
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado en {save_path}\n")
    plt.close()


def print_results(rl_metrics, ga_metrics):
    """Imprimir resultados de la competencia"""

    print("\n" + "="*70)
    print("🏆 RESULTADOS FINALES DE LA COMPETENCIA")
    print("="*70)

    print("\n📊 MÉTRICAS DETALLADAS:\n")

    # Crear tabla comparativa
    print(f"{'Métrica':<25} {'RL (PPO)':<20} {'GA (Genetic)':<20}")
    print("-" * 70)

    for key in rl_metrics.keys():
        rl_val = rl_metrics[key]
        ga_val = ga_metrics[key]

        # Formatear según tipo
        if 'Return' in key or 'Drawdown' in key or 'Rate' in key:
            rl_str = f"{rl_val:>8.2f}%"
            ga_str = f"{ga_val:>8.2f}%"
        elif 'Ratio' in key:
            rl_str = f"{rl_val:>8.4f}"
            ga_str = f"{ga_val:>8.4f}"
        elif 'Trades' in key:
            rl_str = f"{int(rl_val):>8d}"
            ga_str = f"{int(ga_val):>8d}"
        elif 'Value' in key:
            rl_str = f"${rl_val:>8.2f}"
            ga_str = f"${ga_val:>8.2f}"
        else:
            rl_str = f"{rl_val:>8.2f}"
            ga_str = f"{ga_val:>8.2f}"

        # Marcar ganador
        if rl_val > ga_val:
            rl_str += " 🏆"
        elif ga_val > rl_val:
            ga_str += " 🏆"

        print(f"{key:<25} {rl_str:<20} {ga_str:<20}")

    print("="*70)

    # Determinar ganador general
    rl_score = 0
    ga_score = 0

    # Puntos por cada métrica
    if rl_metrics['Total Return (%)'] > ga_metrics['Total Return (%)']:
        rl_score += 1
    else:
        ga_score += 1

    if rl_metrics['Sharpe Ratio'] > ga_metrics['Sharpe Ratio']:
        rl_score += 1
    else:
        ga_score += 1

    if rl_metrics['Max Drawdown (%)'] > ga_metrics['Max Drawdown (%)']: # Menor es mejor
        ga_score += 1
    else:
        rl_score += 1

    if rl_metrics['Win Rate (%)'] > ga_metrics['Win Rate (%)']:
        rl_score += 1
    else:
        ga_score += 1

    print("\n🎯 PUNTUACIÓN FINAL:")
    print(f"   RL (PPO): {rl_score} puntos")
    print(f"   GA (Genetic): {ga_score} puntos")

    if rl_score > ga_score:
        winner = "🤖 RL (PPO) - REINFORCEMENT LEARNING"
        margin = rl_score - ga_score
    elif ga_score > rl_score:
        winner = "🧬 GA (GENETIC ALGORITHM)"
        margin = ga_score - rl_score
    else:
        winner = "🤝 EMPATE"
        margin = 0

    print("\n" + "="*70)
    print(f"🏆 GANADOR: {winner}")
    if margin > 0:
        print(f"   Margen de victoria: {margin} puntos")
    print("="*70)


def main(use_risk_management=True):
    print("\n" + "="*70)
    if use_risk_management:
        print("🛡️ COMPETENCIA FINAL: RL vs GA con Moon Dev Risk Management")
    else:
        print("🥊 COMPETENCIA FINAL: RL vs GA (Sin Risk Management)")
    print("="*70 + "\n")

    # Cargar datos
    data_path = 'data/processed/SPY_with_indicators.csv'
    print(f"Cargando datos de {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"✓ Datos cargados: {len(df)} filas\n")

    # Split datos (usar solo TEST set - 20% final)
    train_size = int(len(df) * 0.6)
    val_size = int(len(df) * 0.2)
    df_test = df.iloc[train_size + val_size:].copy().reset_index(drop=True)

    print("Usando datos de TEST:")
    print(f"  {len(df_test)} días ({df.index[train_size + val_size]} a {df.index[-1]})")
    print("  Capital inicial: $10,000 para cada agente\n")

    # Cargar modelos
    rl_agent, ga_agent = load_models(use_risk_management)

    # Backtest RL
    print("="*70)
    agent_type = "RL con Risk Management" if use_risk_management else "RL (PPO)"
    print(f"🤖 Ejecutando backtest del agente {agent_type}...")
    print("="*70)
    rl_history = backtest_agent(rl_agent, df_test, agent_name="RL (PPO)")
    rl_metrics = calculate_metrics(rl_history)
    print(f"✓ {agent_type} completado: ${rl_metrics['Final Value ($)']:.2f} final\n")

    # Backtest GA
    print("="*70)
    agent_type = "GA con Risk Management" if use_risk_management else "GA (Genetic)"
    print(f"🧬 Ejecutando backtest del agente {agent_type}...")
    print("="*70)
    ga_history = backtest_agent(ga_agent, df_test, agent_name="GA (Genetic)")
    ga_metrics = calculate_metrics(ga_history)
    print(f"✓ {agent_type} completado: ${ga_metrics['Final Value ($)']:.2f} final\n")

    # Mostrar estadísticas de risk management si se usó
    if use_risk_management:
        print("="*70)
        print("📊 ESTADÍSTICAS DE RISK MANAGEMENT")
        print("="*70)

        # Solo mostrar stats si son wrappers
        if hasattr(rl_agent, 'get_wrapper_stats'):
            rl_stats = rl_agent.get_wrapper_stats()
            print("RL Agent Risk Stats:")
            print(f"  Trades procesados: {rl_stats['total_trades_processed']}")
            print(f"  Trades bloqueados: {rl_stats['trades_blocked_by_risk']} ({rl_stats['block_rate']:.1%})")
            print(f"  Trades ajustados: {rl_stats['trades_adjusted_by_risk']} ({rl_stats['adjustment_rate']:.1%})")
            print(f"  Errores de evaluación: {rl_stats['risk_evaluation_errors']}")
        else:
            print("RL Agent: Sin estadísticas de risk management disponibles")

        print()

        if hasattr(ga_agent, 'get_wrapper_stats'):
            ga_stats = ga_agent.get_wrapper_stats()
            print("GA Agent Risk Stats:")
            print(f"  Trades procesados: {ga_stats['total_trades_processed']}")
            print(f"  Trades bloqueados: {ga_stats['trades_blocked_by_risk']} ({ga_stats['block_rate']:.1%})")
            print(f"  Trades ajustados: {ga_stats['trades_adjusted_by_risk']} ({ga_stats['adjustment_rate']:.1%})")
            print(f"  Errores de evaluación: {ga_stats['risk_evaluation_errors']}")
        else:
            print("GA Agent: Sin estadísticas de risk management disponibles")

        print()

    # Resultados
    print_results(rl_metrics, ga_metrics)

    # Graficar
    print("\n📊 Generando visualizaciones...")
    plot_comparison(rl_history, ga_history, rl_metrics, ga_metrics, df_test)

    # Guardar resultados
    suffix = "_with_risk" if use_risk_management else "_no_risk"
    results_path = f'results/competition_results{suffix}.csv'
    results_df = pd.DataFrame({
        'Agent': ['RL (PPO)', 'GA (Genetic)'],
        **{k: [rl_metrics[k], ga_metrics[k]] for k in rl_metrics.keys()}
    })
    results_df.to_csv(results_path, index=False)
    print(f"✓ Resultados guardados en {results_path}")

    print("\n" + "="*70)
    print("✅ COMPETENCIA COMPLETADA")
    print("="*70)

    print("\nArchivos generados:")
    print(f"  - results/figures/competition_results{suffix}.png")
    print(f"  - results/competition_results{suffix}.csv")

    if use_risk_management:
        print("\n🛡️ Risk Management aplicado con parámetros conservadores:")
        print("  - Max Drawdown: 12%")
        print("  - Max Risk por Trade: 4%")
        print("  - Portfolio Heat Limit: 25%")

    print("\n🎉 ¡Felicitaciones! Has completado el proyecto de trading con IA")

    return rl_metrics, ga_metrics, rl_history, ga_history


if __name__ == "__main__":
    import sys

    # Verificar si se especificó usar risk management
    use_risk = len(sys.argv) > 1 and sys.argv[1].lower() in ['true', '1', 'yes', 'risk']

    if use_risk:
        print("Ejecutando competencia CON Moon Dev Risk Management...")
        main(use_risk_management=True)
    else:
        print("Ejecutando competencia SIN Risk Management...")
        print("(Para usar risk management, ejecuta: python compete_agents.py risk)")
        main(use_risk_management=False)