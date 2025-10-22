#!/usr/bin/env python3
"""
Script de diagnóstico para entender por qué el GA no genera trades.
"""

import os
import sys
import pandas as pd
# import numpy as np  # No usado
# import random  # No usado
# from deap import base, creator, tools  # No usados

# Import del environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.train_ga_agent import GAStrategy, evaluate_individual


def diagnose_ga_trading():
    """Diagnosticar por qué el GA no genera trades"""
    print("Diagnóstico del GA Trading")
    print("="*50)

    # Cargar datos
    data_path = 'data/processed/SPY_with_indicators.csv'
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df_train = df.iloc[:int(len(df)*0.6)].copy()

    print(f"Datos de entrenamiento: {len(df_train)} filas")
    print(f"Columnas disponibles: {list(df_train.columns)}")
    print()

    # Verificar indicadores clave
    print("Verificación de indicadores:")
    sample_row = df_train.iloc[100]  # Fila de muestra
    print(f"  RSI_14: {sample_row.get('RSI_14', 'MISSING')}")
    print(f"  MACD_12_26_9: {sample_row.get('MACD_12_26_9', 'MISSING')}")
    print(f"  ATR_14: {sample_row.get('ATR_14', 'MISSING')}")
    print(f"  SMA_20_50_ratio: {sample_row.get('SMA_20_50_ratio', 'MISSING')}")
    print(f"  Volume: {sample_row.get('Volume', 'MISSING')}")
    print(f"  OBV: {sample_row.get('OBV', 'MISSING')}")
    print()

    # Crear estrategia de prueba (genes medios)
    test_genes = [0.5] * 10  # Valores medios
    strategy = GAStrategy(test_genes)

    print("Estrategia de prueba (genes medios):")
    print(f"  RSI Buy: {strategy.rsi_buy:.1f}")
    print(f"  RSI Sell: {strategy.rsi_sell:.1f}")
    print(f"  MACD threshold: {strategy.macd_threshold:.2f}")
    print("  Trend up required: SMA_20_50_ratio > 1.0")
    print(f"  Volume filter: {strategy.volume_filter:.2f}x")
    print()

    # Probar decisiones en algunas filas
    print("Probando decisiones en filas de muestra:")
    for i in [100, 200, 300, 400]:
        if i < len(df_train):
            row = df_train.iloc[i]

            position_info = {
                'has_position': False,
                'entry_price': 0,
                'entry_step': 0,
                'current_step': i
            }

            action = strategy.decide(row, position_info)
            action_name = ['HOLD', 'BUY', 'SELL'][action]

            rsi = row.get('RSI_14', 50)
            macd = row.get('MACD_12_26_9', 0)
            sma_ratio = row.get('SMA_20_50_ratio', 1.0)
            volume_ratio = 2.0  # Simplificado para diagnóstico

            print(f"Fila {i}: RSI={rsi:.1f}, MACD={macd:.2f}, SMA_ratio={sma_ratio:.2f}, Vol_ratio={volume_ratio:.2f} -> {action_name}")
    print()

    # Evaluar individuo de prueba
    print("Evaluando individuo de prueba:")
    fitness = evaluate_individual(test_genes, df_train)
    print(f"Fitness: {fitness[0]:.4f}")
    print()

    # Probar estrategia agresiva (condiciones más laxas)
    print("Probando estrategia agresiva:")
    aggressive_genes = [0.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]  # Más agresiva
    aggressive_strategy = GAStrategy(aggressive_genes)

    print(f"  RSI Buy: {aggressive_strategy.rsi_buy:.1f} (muy bajo)")
    print(f"  RSI Sell: {aggressive_strategy.rsi_sell:.1f} (muy alto)")
    print(f"  MACD threshold: {aggressive_strategy.macd_threshold:.2f} (muy bajo)")
    print(f"  Volume filter: {aggressive_strategy.volume_filter:.2f}x (muy bajo)")

    fitness_aggressive = evaluate_individual(aggressive_genes, df_train)
    print(f"Fitness agresivo: {fitness_aggressive[0]:.4f}")
    print()

    # Verificar distribución de indicadores
    print("Distribución de indicadores clave:")
    print(f"  RSI_14 - Min: {df_train['RSI_14'].min():.1f}, Max: {df_train['RSI_14'].max():.1f}, Mean: {df_train['RSI_14'].mean():.1f}")
    print(f"  MACD_12_26_9 - Min: {df_train['MACD_12_26_9'].min():.2f}, Max: {df_train['MACD_12_26_9'].max():.2f}, Mean: {df_train['MACD_12_26_9'].mean():.2f}")
    print(f"  SMA_20_50_ratio - Min: {df_train['SMA_20_50_ratio'].min():.2f}, Max: {df_train['SMA_20_50_ratio'].max():.2f}, Mean: {df_train['SMA_20_50_ratio'].mean():.2f}")


def main():
    diagnose_ga_trading()


if __name__ == "__main__":
    main()