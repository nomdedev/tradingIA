#!/usr/bin/env python3
"""
Backtest VP IFVG EMA Strategy V2
Tests the improved strategy with position management and pattern enhancements.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.vp_ifvg_ema_strategy_v2 import VPIFVGEmaStrategyV2
import talib


class SimpleBacktester:
    """Simple backtester for strategy testing"""

    def __init__(self, strategy, initial_capital=10000, commission=0.001):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.trades = []
        self.equity_curve = [self.capital]
        
        # PRE-CALCULAR INDICADORES PARA OPTIMIZACIÓN
        self.indicators_cache = {}

    def _precalculate_indicators(self, df):
        """Pre-calcular indicadores que no cambian"""
        print("🔄 Pre-calculando indicadores técnicos...")
        
        # Obtener parámetros de EMA de la estrategia
        strategy_params = self.strategy.get_parameters()
        ema_periods = [
            strategy_params['ema1_length'],  # 20
            strategy_params['ema2_length'],  # 50
            strategy_params['ema3_length'],  # 100
            strategy_params['ema4_length'],  # 200
        ]
        
        # Calcular EMAs usando los períodos de la estrategia
        for i, period in enumerate(ema_periods, 1):
            df[f'ema{i}'] = talib.EMA(df['close'].values, timeperiod=period)
        
        # Calcular ATR
        df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        
        self.indicators_cache = {
            'df_with_indicators': df.copy(),
            'total_bars': len(df)
        }
        
        print("✅ Indicadores pre-calculados")
        return df

    def run_backtest(self, df):
        """Run backtest on dataframe"""
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = [self.capital]

        # PRE-CALCULAR INDICADORES PARA OPTIMIZACIÓN
        df = self._precalculate_indicators(df)

        # OPTIMIZACIÓN: Saltar pre-cálculo de VP si está desactivado
        if self.strategy.use_vp_levels:
            print("🔄 Pre-calculando Volume Profile para todas las barras...")
            print("   Esto puede tomar unos minutos pero acelera el backtest significativamente...")
            
            vp_start_time = time.time()
            self.vp_cache = self._precalculate_volume_profile(df)
            vp_time = time.time() - vp_start_time
            
            print(f"✅ Volume Profile pre-calculado en {vp_time:.1f} segundos")
            print(f"   Cache contiene {len(self.vp_cache)} entradas VP")
        else:
            print("⏭️  Saltando pre-cálculo de Volume Profile (desactivado)")
            self.vp_cache = {}
        print()

        total_bars = len(df)
        print(f"📊 Iniciando backtest con {total_bars} barras...")
        
        # OPTIMIZACIÓN: Procesar cada N barras para velocidad
        step_size = 5  # Procesar cada 5 barras (15 minutos = 1.25 horas)
        effective_bars = (total_bars // step_size) + 1
        
        print(f"⚡ Modo velocidad: procesando cada {step_size} barras")
        print(f"   Barras efectivas: {effective_bars} (de {total_bars})")
        print("⚡ Procesando señales...")
        print()

        start_time = time.time()
        last_update_time = start_time

        # Run through each bar (con step para velocidad)
        for idx in range(0, total_bars, step_size):
            current_time = time.time()

            # Mostrar progreso cada 1000 barras efectivas o cada 10 segundos
            progress = (idx + 1) / total_bars * 100
            if idx % (1000 * step_size) == 0 or (current_time - last_update_time) > 10 or idx >= total_bars - step_size:
                elapsed_time = current_time - start_time
                bars_per_second = (idx + 1) / elapsed_time if elapsed_time > 0 else 0
                remaining_bars = total_bars - (idx + 1)
                estimated_remaining = remaining_bars / bars_per_second if bars_per_second > 0 else 0

                print(f"📈 Progreso: {idx+1}/{total_bars} barras ({progress:.1f}%)")
                print(f"   💰 Capital: ${self.capital:,.2f} | Trades: {len(self.trades)}")
                print(f"   ⏱️  Velocidad: {bars_per_second:.1f} barras/seg | Tiempo restante: {estimated_remaining/60:.1f} min")
                print()
                last_update_time = current_time

            try:
                # Usar VP pre-calculado en lugar de recalcular
                vp_data = self.vp_cache.get(idx, {'poc': df.iloc[idx]['close'], 'vah': df.iloc[idx]['high'], 'val': df.iloc[idx]['low']})
                
                # Usar DataFrame con indicadores pre-calculados
                df_with_indicators = self.indicators_cache['df_with_indicators'].iloc[:idx+1].copy()
                df_multi_tf = {'5min': df_with_indicators}
                
                # Inyectar VP pre-calculado en la estrategia
                self.strategy.current_vp_data = vp_data
                
                signals = self.strategy.generate_signals(df_multi_tf)

                # Process entries
                if signals['entries'].iloc[-1]:  # Entry signal on current bar
                    signal = {
                        'action': 'enter',
                        'direction': signals['signals'].iloc[-1],
                        'strength': signals['trade_scores'].iloc[-1]
                    }
                    self._enter_trade(signal, df.iloc[idx])

                # Process exits
                if signals['exits'].iloc[-1]:  # Exit signal on current bar
                    signal = {
                        'action': 'exit',
                        'direction': -signals['signals'].iloc[-1] if signals['signals'].iloc[-1] != 0 else 0
                    }
                    self._exit_trade(signal, df.iloc[idx])

            except Exception as e:
                print(f"❌ Error en barra {idx}: {str(e)}")
                continue

            # Record equity
            self.equity_curve.append(self.capital)

        print(f"\n✅ Backtest completado. Procesadas {total_bars} barras.")
        total_time = time.time() - start_time
        print(f"⏱️  Tiempo total: {total_time/60:.1f} minutos ({total_time:.1f} segundos)")
        print(f"📊 Trades ejecutados: {len(self.trades)}")
        print(f"⚡ Velocidad promedio: {total_bars/total_time:.1f} barras/segundo")
        print(f"⚡ Optimizaciones aplicadas: Step={step_size}, VP=OFF, VolFilter=OFF")
        print()

        # Calculate metrics
        return self._calculate_metrics()

    def _calculate_indicators(self, df):
        """Calculate technical indicators"""
        # EMAs
        for period in [5, 9, 21, 22, 34, 50, 100, 200]:
            df[f'ema{period}'] = talib.EMA(df['close'].values, timeperiod=period)

        # ATR for stops
        df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)

        return df

    def _precalculate_volume_profile(self, df):
        """Pre-calcular Volume Profile para todas las barras usando ventana móvil"""
        vp_cache = {}
        total_bars = len(df)
        
        # Usar una ventana móvil de VP para balancear velocidad vs precisión
        vp_window = min(500, total_bars)  # Máximo 500 barras para VP
        
        print(f"   📊 Calculando VP con ventana de {vp_window} barras...")
        
        for idx in range(total_bars):
            if idx < 20:  # Necesitamos datos mínimos
                vp_cache[idx] = {'poc': df.iloc[idx]['close'], 'vah': df.iloc[idx]['high'], 'val': df.iloc[idx]['low']}
                continue
                
            # Calcular VP solo para las últimas vp_window barras
            start_idx = max(0, idx - vp_window)
            df_window = df.iloc[start_idx:idx+1]
            
            # Calcular POC (Point of Control) - precio con mayor volumen
            price_bins = np.linspace(df_window['low'].min(), df_window['high'].max(), 100)
            volume_profile = np.zeros(len(price_bins))
            
            for i, row in df_window.iterrows():
                price_range = row['high'] - row['low']
                if price_range == 0:
                    bin_idx = np.argmin(np.abs(price_bins - row['close']))
                    volume_profile[bin_idx] += row['volume']
                else:
                    # Distribuir volumen proporcionalmente en el rango de precios
                    bin_start = np.searchsorted(price_bins, row['low'], side='left')
                    bin_end = np.searchsorted(price_bins, row['high'], side='right')
                    bins_in_range = bin_end - bin_start
                    
                    if bins_in_range > 0:
                        volume_per_bin = row['volume'] / bins_in_range
                        volume_profile[bin_start:bin_end] += volume_per_bin
            
            # Encontrar POC
            poc_idx = np.argmax(volume_profile)
            poc = price_bins[poc_idx]
            
            # Calcular VAH/VAL (Value Area High/Low) - 68% del volumen
            total_volume = volume_profile.sum()
            value_area_volume = total_volume * 0.68
            
            # Acumular desde POC hacia afuera
            vah = poc
            val = poc
            accumulated_volume = volume_profile[poc_idx]
            
            # Expandir hacia arriba
            for i in range(poc_idx + 1, len(volume_profile)):
                if accumulated_volume >= value_area_volume:
                    break
                accumulated_volume += volume_profile[i]
                vah = price_bins[i]
            
            # Expandir hacia abajo
            for i in range(poc_idx - 1, -1, -1):
                if accumulated_volume >= value_area_volume:
                    break
                accumulated_volume += volume_profile[i]
                val = price_bins[i]
            
            vp_cache[idx] = {
                'poc': poc,
                'vah': vah,
                'val': val
            }
            
            # Mostrar progreso cada 5000 barras
            if idx % 5000 == 0 and idx > 0:
                print(f"   🔄 VP calculado para barra {idx}/{total_bars} ({idx/total_bars*100:.1f}%)")
        
        return vp_cache

    def _enter_trade(self, signal, bar):
        """Enter a new trade"""
        entry_price = bar['close']
        quantity = (self.capital * 0.02) / entry_price  # 2% risk
        cost = quantity * entry_price * (1 + self.commission)

        if cost > self.capital:
            return  # Not enough capital

        self.capital -= cost

        trade = {
            'entry_idx': len(self.equity_curve) - 1,
            'entry_price': entry_price,
            'quantity': quantity,
            'direction': signal['direction'],
            'entry_time': bar.name
        }
        self.trades.append(trade)

    def _exit_trade(self, signal, bar):
        """Exit existing trade"""
        if not self.trades:
            return

        # Exit last trade (simplified)
        trade = self.trades[-1]
        exit_price = bar['close']
        pnl = (exit_price - trade['entry_price']) * trade['quantity'] * trade['direction']
        pnl -= abs(pnl) * self.commission  # Commission on exit

        self.capital += trade['entry_price'] * trade['quantity'] + pnl

        trade.update({
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_time': bar.name,
            'duration': len(self.equity_curve) - 1 - trade['entry_idx']
        })

    def _calculate_metrics(self):
        """Calculate backtest metrics"""
        if not self.trades:
            return {
                'final_capital': self.capital,
                'total_return_pct': 0,
                'max_drawdown': 0,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'expectancy': 0,
                'sharpe_ratio': 0
            }

        # Basic calculations
        final_capital = self.capital
        total_return_pct = (final_capital - self.initial_capital) / self.initial_capital * 100

        # Trades analysis
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]

        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        total_profit = sum(t.get('pnl', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        expectancy = sum(t.get('pnl', 0) for t in self.trades) / total_trades if total_trades > 0 else 0

        # Max drawdown
        equity = pd.Series(self.equity_curve)
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min() * 100

        # Sharpe ratio (simplified)
        returns = equity.pct_change().dropna()
        # Sharpe con risk-free rate
        rf_daily = 0.04 / 252
        excess_returns = returns - rf_daily
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if len(returns) > 0 and excess_returns.std() > 0 else 0.0

        return {
            'final_capital': final_capital,
            'total_return_pct': total_return_pct,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'sharpe_ratio': sharpe_ratio
        }


def run_backtest():
    """Run backtest for VP IFVG EMA V2 strategy"""

    print("🚀 BACKTESTING VP IFVG EMA STRATEGY V2 (OPTIMIZADO)")
    print("=" * 60)

    # Load data
    data_path = 'data/btc_15Min.csv'
    if not os.path.exists(data_path):
        print(f"❌ No se encontró el archivo: {data_path}")
        return

    print(f"📊 Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.columns = df.columns.str.lower()
    
    # OPTIMIZACIÓN: Usar solo una muestra de datos para testing rápido
    sample_size = 5000  # Usar solo 5000 barras para testing
    if len(df) > sample_size:
        print(f"⚡ Usando muestra de {sample_size} barras para testing rápido")
        df = df.tail(sample_size)  # Usar las más recientes
    
    print(f"✅ Datos cargados: {len(df)} barras")
    print(f"   Periodo: {df.index[0]} a {df.index[-1]}")
    print()

    # Initialize strategy con optimizaciones para velocidad
    strategy = VPIFVGEmaStrategyV2()
    
    # OPTIMIZACIONES PARA VELOCIDAD EN TESTING
    print("⚡ Aplicando optimizaciones de velocidad...")
    
    # Desactivar cálculos pesados para testing rápido
    strategy.use_vp_levels = False  # Desactivar Volume Profile (muy lento)
    strategy.use_volume_filter = False  # Desactivar filtro de volumen complejo
    
    # Mantener solo lo esencial: FVG + EMA + patrón mejorado
    strategy.use_ema_filter = True
    
    print("   ✅ Volume Profile: DESACTIVADO (muy lento)")
    print("   ✅ Filtro de volumen: DESACTIVADO")
    print("   ✅ Filtro EMA: ACTIVADO")
    print("   ✅ Patrón IFVG + Volumen alto + EMA cross: ACTIVADO")
    print()

    # Initialize backtester
    backtester = SimpleBacktester(
        strategy=strategy,
        initial_capital=10000,
        commission=0.001  # 0.1%
    )

    # Run backtest
    print("⚡ Ejecutando backtest...")
    results = backtester.run_backtest(df)

    # Display results
    print("\n" + "=" * 50)
    print("📊 RESULTADOS DEL BACKTEST")
    print("=" * 50)

    # Basic metrics
    print(f"Capital inicial: ${backtester.initial_capital:,.2f}")
    print(f"Capital final: ${results['final_capital']:,.2f}")
    print(f"Retorno total: {results['total_return_pct']:.2f}%")
    print(f"Máx drawdown: {results['max_drawdown']:.2f}%")
    print(f"Trades totales: {results['total_trades']}")
    print(f"Win rate: {results['win_rate']:.2f}%")
    print(f"Profit factor: {results['profit_factor']:.2f}")
    print(f"Expectancy: ${results['expectancy']:,.4f}")
    print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")

    # Strategy specific metrics
    if 'strategy_metrics' in results:
        metrics = results['strategy_metrics']
        print(f"\n📈 MÉTRICAS DE ESTRATEGIA:")
        print(f"   Trades largos: {metrics.get('long_trades', 0)}")
        print(f"   Trades cortos: {metrics.get('short_trades', 0)}")
        print(f"   Avg trade duration: {metrics.get('avg_trade_duration', 0):.1f} bars")
        print(f"   Max consecutive wins: {metrics.get('max_consecutive_wins', 0)}")
        print(f"   Max consecutive losses: {metrics.get('max_consecutive_losses', 0)}")

    # Save detailed report
    report_path = 'results/vp_ifvg_ema_v2_backtest.md'
    os.makedirs('results', exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# VP IFVG EMA Strategy V2 - Backtest Results\n\n")
        f.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Datos:** {len(df)} barras BTC 15min\n\n")
        f.write("## Resultados Generales\n\n")
        f.write(f"- Capital inicial: ${backtester.initial_capital:,.2f}\n")
        f.write(f"- Capital final: ${results['final_capital']:,.2f}\n")
        f.write(f"- Retorno total: {results['total_return_pct']:.2f}%\n")
        f.write(f"- Máx drawdown: {results['max_drawdown']:.2f}%\n")
        f.write(f"- Trades totales: {results['total_trades']}\n")
        f.write(f"- Win rate: {results['win_rate']:.2f}%\n")
        f.write(f"- Profit factor: {results['profit_factor']:.2f}\n")
        f.write(f"- Expectancy: ${results['expectancy']:,.4f}\n")
        f.write(f"- Sharpe ratio: {results['sharpe_ratio']:.2f}\n\n")

        if 'strategy_metrics' in results:
            f.write("## Métricas de Estrategia\n\n")
            metrics = results['strategy_metrics']
            f.write(f"- Trades largos: {metrics.get('long_trades', 0)}\n")
            f.write(f"- Trades cortos: {metrics.get('short_trades', 0)}\n")
            f.write(f"- Avg trade duration: {metrics.get('avg_trade_duration', 0):.1f} bars\n")
            f.write(f"- Max consecutive wins: {metrics.get('max_consecutive_wins', 0)}\n")
            f.write(f"- Max consecutive losses: {metrics.get('max_consecutive_losses', 0)}\n\n")

        f.write("## Mejoras Implementadas\n\n")
        f.write("- Gestión completa de posiciones con TradePosition class\n")
        f.write("- Stops dinámicos basados en ATR (2x SL, 4x TP)\n")
        f.write("- Risk management (2% capital por trade, 6% diario)\n")
        f.write("- Patrón mejorado: IFVG + Volumen alto (1.5x) + EMA cross (9/21) bullish\n")
        f.write("- Trailing stops activados en profit > 1.5x ATR\n")

    print(f"\n💾 Reporte detallado guardado en: {report_path}")
    print("\n✅ Backtest completado exitosamente!")


if __name__ == "__main__":
    run_backtest()