#!/usr/bin/env python3
"""
Script de ejemplo para cargar y usar datos de BTC/USD en backtesting.

Este script muestra cómo:
1. Cargar datos descargados desde Alpaca
2. Preparar datos para backtesting
3. Ejecutar una estrategia de ejemplo
4. Calcular métricas básicas de rendimiento
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

# Agregar el directorio raíz al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies import load_strategy

def load_btc_data(timeframe='1d', start_date=None, end_date=None):
    """
    Carga datos de BTC/USD desde archivos CSV.

    Args:
        timeframe (str): Timeframe deseado ('1m', '5m', '1h', '1d')
        start_date (str): Fecha de inicio (YYYY-MM-DD)
        end_date (str): Fecha de fin (YYYY-MM-DD)

    Returns:
        pd.DataFrame: Datos OHLCV con datetime index
    """
    # Mapear timeframes a nombres de archivo
    timeframe_map = {
        '1m': '1m', '5m': '5m', '15m': '15min', '1h': '1h', '1d': '1d'
    }

    filename = f"data/raw/btc_usd_{timeframe_map.get(timeframe, '1d')}.csv"

    if not os.path.exists(filename):
        print(f"Archivo de datos no encontrado: {filename}")
        print("Ejecuta primero: python scripts/download_btc_data.py --start-date 2020-01-01 --end-date 2024-01-01 --timeframe 1Day")
        return None

    # Cargar datos
    df = pd.read_csv(filename, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)

    # Filtrar por fechas si se especifican
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    # Verificar que tenemos las columnas necesarias
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Columnas requeridas faltantes en {filename}")
        print(f"Columnas encontradas: {list(df.columns)}")
        print(f"Columnas requeridas: {required_columns}")
        return None

    print(f"Datos cargados: {len(df)} registros de {df.index.min()} a {df.index.max()}")
    return df

def run_backtest(strategy_name, data, params=None):
    """
    Ejecuta un backtest con una estrategia específica.

    Args:
        strategy_name (str): Nombre de la estrategia
        data (pd.DataFrame): Datos OHLCV
        params (dict): Parámetros de la estrategia

    Returns:
        dict: Resultados del backtest
    """
    # Cargar estrategia
    strategy_class = load_strategy(strategy_name)
    if not strategy_class:
        print(f"Error: No se pudo cargar la estrategia '{strategy_name}'")
        return None

    # Usar parámetros por defecto si no se especifican
    if params is None:
        if hasattr(strategy_class, 'get_default_params'):
            params = strategy_class.get_default_params()
        else:
            print(f"Error: La estrategia '{strategy_name}' no tiene parámetros por defecto")
            return None

    # Crear instancia de estrategia
    strategy = strategy_class(**params)

    # Preparar datos en el formato esperado
    data_dict = {'main': data}  # La estrategia espera un dict con timeframes

    # Generar señales
    signals = strategy.generate_signals(data_dict)

    # Simular trading simple (compra en señal BUY, venta en señal SELL)
    capital = 10000  # Capital inicial
    position = 0  # 0 = sin posición, 1 = comprado
    trades = []
    equity = [capital]

    for i, (timestamp, row) in enumerate(data.iterrows()):
        signal = signals.get('main', {}).get(timestamp, 0)

        # Lógica de trading simple
        if signal > 0 and position == 0:  # Señal de compra
            position = 1
            entry_price = row['close']
            trades.append({
                'type': 'BUY',
                'timestamp': timestamp,
                'price': entry_price,
                'quantity': capital / entry_price
            })
            print(f"BUY at {timestamp}: ${entry_price:.2f}")

        elif signal < 0 and position == 1:  # Señal de venta
            position = 0
            exit_price = row['close']
            last_trade = trades[-1]
            pnl = (exit_price - last_trade['price']) * last_trade['quantity']
            capital += pnl
            trades.append({
                'type': 'SELL',
                'timestamp': timestamp,
                'price': exit_price,
                'pnl': pnl
            })
            print(f"SELL at {timestamp}: ${exit_price:.2f}, PnL: ${pnl:.2f}")
            equity.append(capital)

    # Calcular métricas
    if trades:
        total_trades = len([t for t in trades if t['type'] == 'SELL'])
        winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in trades if t.get('pnl', 0) < 0])

        total_pnl = sum(t.get('pnl', 0) for t in trades if 'pnl' in t)
        win_rate = winning_trades / max(total_trades, 1) * 100

        # Sharpe ratio (simplificado)
        returns = pd.Series(equity).pct_change().dropna()
        if len(returns) > 1:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Asumiendo datos diarios
        else:
            sharpe = 0

        results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe,
            'final_capital': capital,
            'return_pct': (capital - 10000) / 10000 * 100,
            'trades': trades
        }
    else:
        results = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'sharpe_ratio': 0,
            'final_capital': capital,
            'return_pct': 0,
            'trades': []
        }

    return results

def plot_results(data, signals, results):
    """Grafica los resultados del backtest"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Gráfico de precios con señales
    ax1.plot(data.index, data['close'], label='BTC/USD', alpha=0.7)

    # Señales de compra
    buy_signals = [ts for ts, sig in signals.get('main', {}).items() if sig > 0]
    if buy_signals:
        buy_prices = [data.loc[ts, 'close'] for ts in buy_signals if ts in data.index]
        ax1.scatter(buy_signals[:len(buy_prices)], buy_prices, color='green', marker='^', s=100, label='BUY')

    # Señales de venta
    sell_signals = [ts for ts, sig in signals.get('main', {}).items() if sig < 0]
    if sell_signals:
        sell_prices = [data.loc[ts, 'close'] for ts in sell_signals if ts in data.index]
        ax1.scatter(sell_signals[:len(sell_prices)], sell_prices, color='red', marker='v', s=100, label='SELL')

    ax1.set_title('BTC/USD con Señales de Trading')
    ax1.set_ylabel('Precio (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gráfico de equity
    if 'equity' in results and results['equity']:
        ax2.plot(results['equity'], label='Capital')
        ax2.set_title('Evolución del Capital')
        ax2.set_ylabel('Capital (USD)')
        ax2.set_xlabel('Trades')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    """Función principal"""
    print("=== Backtesting con Datos BTC/USD ===\n")

    # Seleccionar timeframe
    timeframes = ['5m', '15m', '1h', '4h']
    print("Timeframes disponibles:")
    for i, tf in enumerate(timeframes, 1):
        print(f"  {i}. {tf}")

    # Por defecto usar 1h si existe, sino el primero disponible
    selected_tf = '1h'
    for tf in ['1h', '4h', '15m', '5m']:
        if os.path.exists(f"data/raw/btc_usd_{tf}.csv"):
            selected_tf = tf
            break

    print(f"Usando timeframe: {selected_tf}")

    # Cargar datos
    print("1. Cargando datos...")
    data = load_btc_data(selected_tf, '2023-01-01', '2024-01-01')
    if data is None:
        print(f"No se encontraron datos para {selected_tf}. Ejecuta primero:")
        print(f"python scripts/download_btc_data.py --start-date 2020-01-01 --end-date 2024-01-01 --timeframe {selected_tf.replace('m', 'Min').replace('h', 'Hour')}")
        return

    # Seleccionar estrategia
    available_strategies = [
        "RSI Mean Reversion",
        "MACD Momentum",
        "Bollinger Bands",
        "MA Crossover",
        "Volume Breakout"
    ]

    print("\n2. Estrategias disponibles:")
    for i, strategy in enumerate(available_strategies, 1):
        print(f"   {i}. {strategy}")

    # Usar RSI Mean Reversion como ejemplo
    strategy_name = "RSI Mean Reversion"
    print(f"\n3. Ejecutando backtest con: {strategy_name}")

    # Ejecutar backtest
    results = run_backtest(strategy_name, data)

    if results:
        print("\n4. Resultados del Backtest:")
        print(f"   Total de trades: {results['total_trades']}")
        print(f"   Trades ganadores: {results['winning_trades']}")
        print(f"   Trades perdedores: {results['losing_trades']}")
        print(".1f")
        print(".2f")
        print(".2f")
        print(".2f")

        # Mostrar trades
        if results['trades']:
            print("\n5. Últimos 5 trades:")
            for trade in results['trades'][-10:]:  # Últimos 10 trades
                if 'pnl' in trade:
                    print(f"   {trade['type']} {trade['timestamp'].strftime('%Y-%m-%d')} @ ${trade['price']:.2f} | PnL: ${trade['pnl']:.2f}")
                else:
                    print(f"   {trade['type']} {trade['timestamp'].strftime('%Y-%m-%d')} @ ${trade['price']:.2f}")

        print("\n=== Backtest completado ===")
    else:
        print("Error ejecutando el backtest")

if __name__ == "__main__":
    main()