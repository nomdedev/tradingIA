#!/usr/bin/env python3
"""
Agente de Algoritmos Genéticos para trading usando DEAP.
Evoluciona parámetros de estrategias de trading técnico.
"""

import os
import sys
sys.path.append('.')
import random
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from rich.console import Console
from functools import partial

console = Console()

def load_data():
    """Cargar datos procesados con indicadores."""
    data_path = "data/processed/SPY_with_indicators.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el archivo de datos: {data_path}")

    df = pd.read_csv(data_path, index_col=0)
    console.print(f"📥 Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
    return df

def calculate_fitness(individual, data):
    """
    Función de fitness: calcula el retorno total de una estrategia.

    Individual: [rsi_overbought, rsi_oversold, macd_signal_threshold, bb_width_threshold]
    """
    rsi_overbought, rsi_oversold, macd_signal_threshold, bb_width_threshold = individual

    # Copiar datos para no modificar el original
    df = data.copy()

    # Señales de trading basadas en indicadores
    signals = pd.Series(0, index=df.index)  # 0: hold, 1: buy, -1: sell

    # RSI strategy
    rsi = df['RSI_14']
    signals[(rsi < rsi_oversold)] = 1  # Buy when oversold
    signals[(rsi > rsi_overbought)] = -1  # Sell when overbought

    # MACD strategy
    macd = df['MACD_12_26_9']
    macd_signal = df['MACDs_12_26_9']
    macd_hist = df['MACDh_12_26_9']

    # Buy when MACD crosses above signal and histogram is positive
    macd_buy = (macd > macd_signal) & (macd_hist > macd_signal_threshold)
    # Sell when MACD crosses below signal and histogram is negative
    macd_sell = (macd < macd_signal) & (macd_hist < -macd_signal_threshold)

    signals[macd_buy] = 1
    signals[macd_sell] = -1

    # Bollinger Bands strategy
    bb_upper = df['BBU_20_2.0']
    bb_lower = df['BBL_20_2.0']
    bb_middle = df['BBM_20_2.0']
    close = df['Close']

    # Buy when price touches lower band and bands are wide
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_buy = (close <= bb_lower) & (bb_width > bb_width_threshold)
    bb_sell = (close >= bb_upper) & (bb_width > bb_width_threshold)

    signals[bb_buy] = 1
    signals[bb_sell] = -1

    # Simular trading
    position = 0  # 0: no position, 1: long
    entry_price = 0
    total_return = 0
    trades = 0

    for i in range(len(signals)):
        signal = signals.iloc[i]
        price = close.iloc[i]

        if signal == 1 and position == 0:  # Buy signal
            position = 1
            entry_price = price
            trades += 1
        elif signal == -1 and position == 1:  # Sell signal
            # Calculate return
            trade_return = (price - entry_price) / entry_price
            total_return += trade_return
            position = 0
            trades += 1

    # If still in position at end, close it
    if position == 1:
        final_price = close.iloc[-1]
        trade_return = (final_price - entry_price) / entry_price
        total_return += trade_return

    # Penalizar estrategias con muy pocos trades (overfitting)
    if trades < 5:
        total_return -= 1.0  # Penalización

    return total_return,

def create_toolbox():
    """Crear toolbox de DEAP para el algoritmo genético."""
    # Crear tipos
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Toolbox
    toolbox = base.Toolbox()

    # Atributos: RSI overbought/oversold (30-70), MACD threshold (0-0.001), BB width (0.01-0.05)
    toolbox.register("attr_rsi_overbought", random.uniform, 60, 80)
    toolbox.register("attr_rsi_oversold", random.uniform, 20, 40)
    toolbox.register("attr_macd_threshold", random.uniform, 0, 0.002)
    toolbox.register("attr_bb_width", random.uniform, 0.01, 0.06)

    # Individuo
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.attr_rsi_overbought, toolbox.attr_rsi_oversold,
                     toolbox.attr_macd_threshold, toolbox.attr_bb_width), n=1)

    # Población
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operadores
    toolbox.register("evaluate", calculate_fitness)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

def evolve_ga_agent(data, n_generations=50, population_size=100):
    """Evolucionar el agente de algoritmos genéticos."""
    console.print("🧬 Iniciando evolución del agente GA...")

    toolbox = create_toolbox()

    # Registrar función de evaluación con datos
    toolbox.register("evaluate", partial(calculate_fitness, data=data))

    # Crear población inicial
    pop = toolbox.population(n=population_size)

    # Evaluación inicial
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Algoritmo evolutivo
    final_pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2,
                                           ngen=n_generations, stats=stats, verbose=True)

    # Mejor individuo
    best_ind = tools.selBest(final_pop, 1)[0]
    best_fitness = best_ind.fitness.values[0]

    console.print(f"🏆 Mejor individuo: {best_ind}")
    console.print(f"🎯 Mejor fitness: {best_fitness:.4f}")

    return best_ind, logbook

def plot_evolution(logbook):
    """Graficar la evolución del fitness."""
    gen = logbook.select("gen")
    fit_max = logbook.select("max")
    fit_avg = logbook.select("avg")

    plt.figure(figsize=(10, 6))
    plt.plot(gen, fit_max, label="Mejor fitness")
    plt.plot(gen, fit_avg, label="Fitness promedio")
    plt.xlabel("Generación")
    plt.ylabel("Fitness (Retorno total)")
    plt.title("Evolución del Agente Genético")
    plt.legend()
    plt.grid(True)

    # Crear directorio si no existe
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/ga_evolution.png", dpi=300, bbox_inches='tight')
    plt.show()

def save_ga_agent(best_individual, filename="models/ga_agent_params.txt"):
    """Guardar los parámetros del mejor agente GA."""
    os.makedirs("models", exist_ok=True)

    with open(filename, 'w') as f:
        f.write("# Parámetros del Agente Genético\n")
        f.write(f"RSI_Overbought: {best_individual[0]:.2f}\n")
        f.write(f"RSI_Oversold: {best_individual[1]:.2f}\n")
        f.write(f"MACD_Threshold: {best_individual[2]:.6f}\n")
        f.write(f"BB_Width_Threshold: {best_individual[3]:.4f}\n")

    console.print(f"💾 Parámetros guardados en: {filename}")

def main():
    """Función principal."""
    console.print("🚀 Iniciando entrenamiento del agente GA para trading")

    try:
        # Cargar datos
        data = load_data()

        # Evolucionar agente
        best_agent, logbook = evolve_ga_agent(data, n_generations=30, population_size=50)

        # Graficar evolución
        plot_evolution(logbook)

        # Guardar agente
        save_ga_agent(best_agent)

        console.print("✅ Entrenamiento GA completado!")

    except Exception as e:
        console.print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()