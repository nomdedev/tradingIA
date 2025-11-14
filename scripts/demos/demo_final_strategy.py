"""
Demo Script para BTC Final Backtest

Este script demuestra cómo usar la estrategia final híbrida avanzada
que combina todas las mejores características implementadas.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Agregar src al path
sys.path.append(str(Path(__file__).parent / 'src'))

def load_sample_btc_data():
    """Cargar datos de muestra BTC para demo"""
    # Crear datos sintéticos realistas para BTC
    np.random.seed(42)

    # Generar fechas (últimos 2 años)
    dates = pd.date_range('2022-01-01', '2024-01-01', freq='1H')

    # Parámetros realistas para BTC
    base_price = 20000
    volatility = 0.02  # 2% volatilidad por hora
    drift = 0.0001     # Drift positivo pequeño

    prices = [float(base_price)]
    volumes = []

    for i in range(1, len(dates)):
        # Movimiento browniano con drift
        shock = np.random.normal(drift, volatility)
        new_price = prices[-1] * (1 + shock)

        # Evitar precios negativos
        new_price = max(new_price, 1000)

        prices.append(new_price)

        # Volumen realista (correlacionado con volatilidad)
        vol_shock = abs(shock) * 1000000 + np.random.normal(0, 500000)
        volumes.append(max(vol_shock, 100000))

    # Crear OHLCV
    df = pd.DataFrame({
        'Open': prices[:-1],
        'High': [max(o, c + abs(c-o)*np.random.uniform(0, 0.02)) for o, c in zip(prices[:-1], prices[1:])],
        'Low': [min(o, c - abs(c-o)*np.random.uniform(0, 0.02)) for o, c in zip(prices[:-1], prices[1:])],
        'Close': prices[1:],
        'Volume': volumes
    }, index=dates[1:])

    # Ajustar High/Low para ser consistentes
    df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, len(df)))
    df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, len(df)))

    return df

def main():
    """Ejecutar demo simplificado"""
    print("🚀 DEMO: BTC Final Backtest - Estrategia Híbrida Avanzada")
    print("=" * 70)

    # Cargar datos de muestra
    print("📊 Generando datos BTC sintéticos...")
    df_btc = load_sample_btc_data()
    print(f"✅ Datos generados: {len(df_btc)} velas de {df_btc.index[0]} a {df_btc.index[-1]}")
    print(f"Precio inicial: ${df_btc['Close'].iloc[0]:.2f}")
    print(f"Precio final: ${df_btc['Close'].iloc[-1]:.2f}")
    print()

    # Ejecutar backtest simplificado (sin walk-forward para demo)
    print("🎯 Ejecutando backtest simplificado...")
    from backtesting import Backtest
    from src.btc_final_backtest import BTCFinalStrategy

    # Usar datos recientes para demo más rápido
    df_recent = df_btc.tail(1000)  # Últimas 1000 velas

    bt = Backtest(df_recent, BTCFinalStrategy, cash=10000, commission=0.0003)
    result = bt.run()

    print("✅ Backtest completado exitosamente")
    print()

    # Mostrar resultados básicos
    print("📊 RESULTADOS DEL BACKTEST:")
    print("=" * 40)
    print(f"Retorno Total: {result['Return [%]']:.2f}%")
    print(f"Win Rate: {result['Win Rate [%]']:.1f}%")
    print(f"Max Drawdown: {result['Max. Drawdown [%]']:.2f}%")
    print(f"Sharpe Ratio: {result['Sharpe Ratio']:.2f}")
    print(f"Total Trades: {result['# Trades']}")
    print()

    print("🎉 DEMO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print("La estrategia final combina:")
    print("• LSTM para predicciones de precio")
    print("• Kalman VMA para filtros de momentum")
    print("• Risk parity sizing")
    print("• HFT optimizations (slippage, latency)")
    print("• Ensemble approach")
    print()
    print("Para testing completo con walk-forward, usar:")
    print("python -c \"from src.btc_final_backtest import run_final_backtest; run_final_backtest(df_btc)\"")
    print()
    print("📁 El sistema completo está listo para implementación!")

if __name__ == "__main__":
    main()