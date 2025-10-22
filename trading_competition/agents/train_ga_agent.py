#!/usr/bin/env python3
"""
Script completo para evolucionar agente GA (Genetic Algorithm) para trading.

Este script usa DEAP para evolucionar estrategias de trading basadas en reglas
genéticas, optimizando parámetros como RSI, MACD, stop loss, etc.
"""

import os
import sys
import pandas as pd
import numpy as np
import random
import pickle
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# Import del environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from environments.trading_env import TradingEnv  # No usado en GA


# Configuración DEAP (solo si no existen)
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


class GAStrategy:
    """Estrategia de trading basada en genes"""

    def __init__(self, genes):
        self.genes = genes
        self.decode_genes()

    def decode_genes(self):
        """Decodificar genes a parámetros de estrategia"""
        # 15 genes: [RSI_buy, RSI_sell, MACD_thresh, stop_loss, take_profit,
        #            position_size, ATR_mult, trend_days, min_hold, volume_filter,
        #            squeeze_momentum_thresh, use_ifvg_filter, ema_alignment_required,
        #            squeeze_release_lookback, ifvg_proximity_threshold]

        self.rsi_buy = 20 + self.genes[0] * 20  # [20, 40]
        self.rsi_sell = 60 + self.genes[1] * 20  # [60, 80]
        self.macd_threshold = self.genes[2] * 3  # [0, 3]
        self.stop_loss = 0.01 + self.genes[3] * 0.09  # [1%, 10%]
        self.take_profit = 0.02 + self.genes[4] * 0.18  # [2%, 20%]
        self.position_size = 0.5 + self.genes[5] * 0.5  # [50%, 100%]
        self.atr_multiplier = 1.0 + self.genes[6] * 3.0  # [1, 4]
        self.trend_days = int(20 + self.genes[7] * 80)  # [20, 100]
        self.min_holding_period = int(1 + self.genes[8] * 9)  # [1, 10]
        self.volume_filter = 0.5 + self.genes[9] * 1.5  # [0.5, 2.0]

        # Nuevos genes para estrategia avanzada
        self.squeeze_momentum_threshold = self.genes[10] * 5  # [0, 5]
        self.use_ifvg_filter = self.genes[11] > 0.5  # [True/False]
        self.ema_alignment_required = self.genes[12] > 0.5  # [True/False]
        self.squeeze_release_lookback = int(1 + self.genes[13] * 9)  # [1, 10]
        self.ifvg_proximity_threshold = 0.01 + self.genes[14] * 0.04  # [1%, 5%]

    def decide(self, row, position_info):
        """
        Decidir acción basada en reglas genéticas con indicadores avanzados

        Args:
            row: Fila actual del DataFrame con indicadores
            position_info: dict con info de posición actual
                {'has_position': bool, 'entry_price': float,
                 'entry_step': int, 'current_step': int}

        Returns:
            action: 0=HOLD, 1=BUY, 2=SELL
        """

        # Extraer indicadores tradicionales
        rsi = row.get('RSI_14', 50)
        macd = row.get('MACD_12_26_9', 0)
        atr = row.get('ATR_14', 0)
        close = row['Close']

        # Extraer indicadores avanzados
        squeeze_momentum = row.get('squeeze_momentum', 0)
        squeeze_momentum_delta = row.get('squeeze_momentum_delta', 0)
        ifvg_bullish_nearby = row.get('ifvg_bullish_nearby', 0)
        ema_alignment = row.get('ema_alignment', 0)
        price_above_ema200 = row.get('price_above_ema200', 0)
        price_above_ema55 = row.get('price_above_ema55', 0)

        # FILTROS OBLIGATORIOS (deben cumplirse siempre)
        # 1. No operar contra macro (precio debe estar sobre EMA200)
        if price_above_ema200 == 0:
            return 0  # HOLD - no operar contra tendencia macro

        # 2. Si se requiere alineación EMA, verificar
        if self.ema_alignment_required and ema_alignment != 1:
            return 0  # HOLD - EMAs no alineadas bullish

        # 3. Si se usa filtro IFVG, verificar que haya IFVG bullish cercano
        if self.use_ifvg_filter and ifvg_bullish_nearby == 0:
            return 0  # HOLD - no hay IFVG bullish cercano

        # Calcular tendencia
        sma_ratio = row.get('SMA_20_50_ratio', 1.0)
        trend_up = sma_ratio > 1.0

        # Volumen (simplificado)
        volume_ratio = 2.0

        # Si NO tiene posición -> evaluar COMPRA
        if not position_info['has_position']:
            # Sistema de puntuación para compra (más flexible)
            score = 0

            # Indicadores tradicionales (+1 punto cada uno)
            if rsi < self.rsi_buy:
                score += 1
            if macd > self.macd_threshold:
                score += 1
            if trend_up:
                score += 1
            if volume_ratio > self.volume_filter:
                score += 1

            # Indicadores avanzados (+1 punto cada uno)
            if squeeze_momentum > self.squeeze_momentum_threshold:
                score += 1
            if squeeze_momentum_delta > 0:  # Momentum aumentando
                score += 1
            if ema_alignment == 1:
                score += 1

            # Comprar si al menos 3 condiciones se cumplen (más restrictivo con indicadores avanzados)
            if score >= 3:
                return 1  # BUY
            else:
                return 0  # HOLD

        # Si TIENE posición -> evaluar VENTA
        else:
            entry_price = position_info['entry_price']
            current_price = close
            days_held = position_info['current_step'] - position_info['entry_step']

            # Calcular P&L
            pnl_pct = (current_price - entry_price) / entry_price

            # EXIT RULES (ordenadas por prioridad)

            # 1. Stop loss tradicional
            if pnl_pct < -self.stop_loss:
                return 2  # SELL (stop loss)

            # 2. Take profit
            if pnl_pct > self.take_profit:
                return 2  # SELL (take profit)

            # 3. Stop loss dinámico con ATR
            atr_stop = entry_price - (atr * self.atr_multiplier)
            if current_price < atr_stop:
                return 2  # SELL (ATR stop)

            # 4. Salidas técnicas avanzadas
            if squeeze_momentum < 0:  # Momentum negativo
                return 2  # SELL

            if price_above_ema55 == 0:  # Precio bajo EMA55
                return 2  # SELL

            # 5. Señales técnicas tradicionales (con período mínimo)
            if days_held >= self.min_holding_period:
                if rsi > self.rsi_sell or macd < 0:
                    return 2  # SELL (señal técnica)

            return 0  # HOLD


def evaluate_individual(individual, df, initial_balance=10000):
    """
    Evaluar fitness de un individuo (estrategia)

    Returns:
        tuple: (fitness_score,) - debe ser tupla para DEAP
    """
    try:
        # Crear estrategia desde genes
        strategy = GAStrategy(individual)

        # Simular trading
        balance = initial_balance
        shares = 0
        trades = []

        entry_price = 0
        entry_step = 0

        for i in range(len(df)):
            row = df.iloc[i]

            # Info de posición actual
            position_info = {
                'has_position': shares > 0,
                'entry_price': entry_price,
                'entry_step': entry_step,
                'current_step': i
            }

            # Decidir acción
            action = strategy.decide(row, position_info)

            current_price = row['Close']

            # Ejecutar acción
            if action == 1 and shares == 0 and balance > 0:  # BUY
                # Comprar con position_size% del balance
                cost_per_share = current_price * 1.001  # Comisión 0.1%
                shares_to_buy = (balance * strategy.position_size) / cost_per_share
                cost = shares_to_buy * cost_per_share

                if cost <= balance:
                    shares = shares_to_buy
                    balance -= cost
                    entry_price = current_price
                    entry_step = i
                    trades.append({'type': 'BUY', 'price': current_price, 'step': i})

            elif action == 2 and shares > 0:  # SELL
                # Vender todas las shares
                proceeds = shares * current_price * 0.999  # Comisión 0.1%
                balance += proceeds
                profit = (current_price - entry_price) / entry_price
                trades.append({'type': 'SELL', 'price': current_price,
                              'profit': profit, 'step': i})
                shares = 0

        # Cerrar posición final si está abierta
        if shares > 0:
            final_price = df.iloc[-1]['Close']
            balance += shares * final_price * 0.999
            shares = 0

        # Calcular métricas
        final_value = balance
        total_return = (final_value - initial_balance) / initial_balance

        # Penalizaciones
        if len(trades) < 5:  # Muy pocos trades
            return (-10.0,)

        if len(trades) > 200:  # Overtrading
            return (-5.0,)

        # Calcular Sharpe Ratio simplificado
        if len(trades) >= 10:
            profits = [t['profit'] for t in trades if 'profit' in t]
            if len(profits) > 0:
                sharpe = np.mean(profits) / (np.std(profits) + 1e-8) * np.sqrt(252)
            else:
                sharpe = 0
        else:
            sharpe = total_return

        # Fitness = Sharpe Ratio (o retorno si pocos trades)
        fitness = sharpe

        return (fitness,)

    except Exception as e:
        print(f"Error evaluando individuo: {e}")
        return (-999.0,)


def setup_ga():
    """Configurar toolbox de DEAP"""
    toolbox = base.Toolbox()

    # Generador de genes (floats entre 0 y 1)
    toolbox.register("attr_float", random.random)

    # Individuo: lista de 15 genes
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                    toolbox.attr_float, n=15)

    # Población
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operadores genéticos
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def evolve_ga(df, population_size=100, n_generations=50):
    """Evolucionar población de estrategias"""

    print("="*60)
    print("EVOLUCIÓN DE AGENTE GA (GENETIC ALGORITHM)")
    print("="*60)

    # Setup
    toolbox = setup_ga()

    # Registrar función de evaluación
    toolbox.register("evaluate", evaluate_individual, df=df)

    # Crear población inicial
    print(f"\n1. Creando población inicial de {population_size} individuos...")
    pop = toolbox.population(n=population_size)
    print("[SUCCESS] Poblacion creada\n")

    # Estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Hall of fame (mejores individuos)
    hof = tools.HallOfFame(1)

    print(f"2. Evolucionando por {n_generations} generaciones...")
    print("   (Esto toma ~5-10 minutos)\n")
    print("="*60)

    # Evolucionar
    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.8,  # Probabilidad de crossover
        mutpb=0.2,  # Probabilidad de mutación
        ngen=n_generations,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    print("\n" + "="*60)
    print("✅ EVOLUCIÓN COMPLETADA")
    print("="*60)

    return pop, hof, logbook


def save_best_individual(best_individual, save_path='models/ga_best_individual.pkl'):
    """Guardar mejor individuo"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(best_individual, f)

    print(f"\n[SUCCESS] Mejor individuo guardado en {save_path}")


def plot_evolution(logbook, save_path='results/figures/ga_evolution.png'):
    """Graficar evolución del GA"""

    gen = logbook.select("gen")
    avg_fitness = logbook.select("avg")
    max_fitness = logbook.select("max")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(gen, avg_fitness, 'b-', label='Average Fitness', linewidth=2)
    ax.plot(gen, max_fitness, 'r-', label='Best Fitness', linewidth=2)
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness (Sharpe Ratio)', fontsize=12)
    ax.set_title('GA Evolution Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Grafico guardado en {save_path}\n")
    plt.close()


def main():
    print("\n" + "="*60)
    print("ENTRENAMIENTO DE AGENTE GA PARA TRADING")
    print("="*60 + "\n")

    # Cargar datos
    data_path = 'data/processed/SPY_with_indicators.csv'
    print(f"Cargando datos de {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"[SUCCESS] Datos cargados: {len(df)} filas\n")

    # Split train (60%)
    train_size = int(len(df) * 0.6)
    df_train = df.iloc[:train_size].copy()
    print(f"Usando {len(df_train)} días para entrenamiento\n")

    # Evolucionar
    pop, hof, logbook = evolve_ga(
        df_train,
        population_size=100,
        n_generations=50
    )

    # Mejor individuo
    best = hof[0]
    best_fitness = best.fitness.values[0]

    print("\n" + "="*60)
    print("MEJOR INDIVIDUO ENCONTRADO")
    print("="*60)
    print(f"Fitness (Sharpe Ratio): {best_fitness:.4f}")
    print("\nGenes decodificados:")

    strategy = GAStrategy(best)
    print(f"  - RSI Buy threshold: {strategy.rsi_buy:.1f}")
    print(f"  - RSI Sell threshold: {strategy.rsi_sell:.1f}")
    print(f"  - MACD threshold: {strategy.macd_threshold:.2f}")
    print(f"  - Stop loss: {strategy.stop_loss*100:.1f}%")
    print(f"  - Take profit: {strategy.take_profit*100:.1f}%")
    print(f"  - Position size: {strategy.position_size*100:.1f}%")
    print(f"  - ATR multiplier: {strategy.atr_multiplier:.2f}")
    print(f"  - Trend filter days: {strategy.trend_days}")
    print(f"  - Min holding period: {strategy.min_holding_period} días")
    print(f"  - Volume filter: {strategy.volume_filter:.2f}x")
    print("="*60)

    # Guardar
    save_best_individual(best)

    # Graficar
    plot_evolution(logbook)

    print("\n✅ Proceso completado exitosamente!")
    print("\nArchivos generados:")
    print("  - models/ga_best_individual.pkl (mejor estrategia)")
    print("  - results/figures/ga_evolution.png (gráfico evolución)")

    print("\nPróximos pasos:")
    print("  1. Revisar gráfico de evolución")
    print("  2. Competir RL vs GA en datos de test")
    print("  3. Analizar qué estrategia funcionó mejor")


if __name__ == "__main__":
    main()