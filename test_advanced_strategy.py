#!/usr/bin/env python3
"""
Script de prueba para estrategia avanzada con Squeeze Momentum, IFVG y EMAs.

Este script demuestra el funcionamiento de la nueva estrategia GA con indicadores
profesionales, mostrando decisiones de trading con explicaciones detalladas.
"""

import os
import sys
import pandas as pd
import numpy as np

# Agregar paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'trading_competition'))

# Imports
from trading_competition.agents.train_ga_agent import GAStrategy


def create_manual_ga_strategy():
    """
    Crear estrategia GA con genes optimizados manualmente para estrategia avanzada.

    Genes optimizados basados en análisis técnico profesional:
    - RSI: conservador (30/70)
    - MACD: sensible (0.5)
    - Squeeze: agresivo (2.0 threshold)
    - IFVG: activado
    - EMA alignment: requerido
    """
    # 15 genes optimizados manualmente
    genes = [
        0.5,   # RSI_buy = 30 (0.5 * 20 + 20)
        0.5,   # RSI_sell = 70 (0.5 * 20 + 60)
        0.17,  # MACD_threshold = 0.5 (0.17 * 3)
        0.11,  # stop_loss = 2% (0.11 * 0.09 + 0.01)
        0.22,  # take_profit = 6% (0.22 * 0.18 + 0.02)
        0.8,   # position_size = 90% (0.8 * 0.5 + 0.5)
        0.5,   # ATR_multiplier = 2.5 (0.5 * 3.0 + 1.0)
        0.25,  # trend_days = 40 (0.25 * 80 + 20)
        0.22,  # min_holding_period = 3 días (0.22 * 9 + 1)
        0.67,  # volume_filter = 1.5 (0.67 * 1.5 + 0.5)
        0.4,   # squeeze_momentum_threshold = 2.0 (0.4 * 5)
        0.8,   # use_ifvg_filter = True (0.8 > 0.5)
        0.9,   # ema_alignment_required = True (0.9 > 0.5)
        0.33,  # squeeze_release_lookback = 4 (0.33 * 9 + 1)
        0.25   # ifvg_proximity_threshold = 2% (0.25 * 0.04 + 0.01)
    ]

    return GAStrategy(genes)


def simulate_trading_detailed(strategy, df, days=100):
    """
    Simular trading con explicaciones detalladas de cada decisión.

    Args:
        strategy: Instancia de GAStrategy
        df: DataFrame con indicadores
        days: Número de días a simular
    """
    print("\n" + "="*80)
    print("SIMULACIÓN DETALLADA DE TRADING - ESTRATEGIA AVANZADA")
    print("="*80)

    # Estado inicial
    balance = 10000
    shares = 0
    entry_price = 0
    entry_step = 0
    trades = []

    print(f"Capital inicial: ${balance:,.2f}")
    print(f"Días a simular: {min(days, len(df))}")
    print("Estrategia: Squeeze Momentum + IFVG + EMAs institucionales")
    print()

    # Simular día por día
    for i in range(min(days, len(df))):
        row = df.iloc[i]
        current_price = row['Close']

        # Información de posición
        position_info = {
            'has_position': shares > 0,
            'entry_price': entry_price,
            'entry_step': entry_step,
            'current_step': i
        }

        # Mostrar estado actual
        print(f"\n--- DÍA {i+1} ---")
        print(f"Precio: ${current_price:.2f}")

        # Mostrar indicadores clave
        squeeze_on = row.get('squeeze_on', 0)
        squeeze_momentum = row.get('squeeze_momentum', 0)
        squeeze_momentum_delta = row.get('squeeze_momentum_delta', 0)
        ifvg_bullish = row.get('ifvg_bullish_nearby', 0)
        ema_alignment = row.get('ema_alignment', 0)
        price_above_ema200 = row.get('price_above_ema200', 0)
        price_above_ema55 = row.get('price_above_ema55', 0)

        print(f"Squeeze: {'ON' if squeeze_on else 'OFF'} | Momentum: {squeeze_momentum:.2f} | Delta: {squeeze_momentum_delta:.2f}")
        print(f"IFVG Bullish: {'YES' if ifvg_bullish else 'NO'} | EMA Alignment: {ema_alignment} | Above EMA200: {'YES' if price_above_ema200 else 'NO'}")

        # Verificar filtros obligatorios
        filters_passed = True
        filter_reasons = []

        if price_above_ema200 == 0:
            filters_passed = False
            filter_reasons.append("Precio bajo EMA200 (contra tendencia macro)")

        if strategy.ema_alignment_required and ema_alignment != 1:
            filters_passed = False
            filter_reasons.append("EMAs no alineadas bullish")

        if strategy.use_ifvg_filter and ifvg_bullish == 0:
            filters_passed = False
            filter_reasons.append("No hay IFVG bullish cercano")

        # Decidir acción
        if not filters_passed:
            action = 0  # HOLD
            action_reason = f"FILTROS BLOQUEANDO: {'; '.join(filter_reasons)}"
        else:
            action = strategy.decide(row, position_info)

            # Explicar decisión
            if not position_info['has_position']:
                # Lógica de compra
                score = 0
                reasons = []

                rsi = row.get('RSI_14', 50)
                macd = row.get('MACD_12_26_9', 0)
                sma_ratio = row.get('SMA_20_50_ratio', 1.0)
                volume_ratio = 2.0  # Simplificado

                if rsi < strategy.rsi_buy:
                    score += 1
                    reasons.append(f"RSI {rsi:.1f} < {strategy.rsi_buy:.1f}")
                if macd > strategy.macd_threshold:
                    score += 1
                    reasons.append(f"MACD {macd:.3f} > {strategy.macd_threshold:.3f}")
                if sma_ratio > 1.0:
                    score += 1
                    reasons.append(f"Tendencia alcista (SMA ratio {sma_ratio:.2f})")
                if volume_ratio > strategy.volume_filter:
                    score += 1
                    reasons.append(f"Volumen alto {volume_ratio:.1f} > {strategy.volume_filter:.1f}")
                if squeeze_momentum > strategy.squeeze_momentum_threshold:
                    score += 1
                    reasons.append(f"Squeeze momentum {squeeze_momentum:.2f} > {strategy.squeeze_momentum_threshold:.2f}")
                if squeeze_momentum_delta > 0:
                    score += 1
                    reasons.append(f"Momentum aumentando {squeeze_momentum_delta:.2f}")
                if ema_alignment == 1:
                    score += 1
                    reasons.append("EMAs alineadas bullish")

                if action == 1:
                    action_reason = f"COMPRA (Score: {score}/7) - {'; '.join(reasons)}"
                else:
                    action_reason = f"HOLD (Score: {score}/7 insuficiente) - {'; '.join(reasons)}"
            else:
                # Lógica de venta
                pnl_pct = (current_price - entry_price) / entry_price
                days_held = i - entry_step

                if action == 2:
                    if pnl_pct < -strategy.stop_loss:
                        action_reason = f"VENTA - Stop Loss: {pnl_pct:.1%} < -{strategy.stop_loss:.1%}"
                    elif pnl_pct > strategy.take_profit:
                        action_reason = f"VENTA - Take Profit: {pnl_pct:.1%} > {strategy.take_profit:.1%}"
                        action_reason = "VENTA - ATR Stop: precio bajo nivel ATR"
                    elif squeeze_momentum < 0:
                        action_reason = f"VENTA - Squeeze momentum negativo: {squeeze_momentum:.2f}"
                    elif price_above_ema55 == 0:
                        action_reason = "VENTA - Precio bajo EMA55"
                    else:
                        action_reason = f"VENTA - Señal técnica (días held: {days_held})"
                else:
                    action_reason = "HOLD - Posición abierta, sin señales de venta"

        # Mostrar decisión
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        print(f"Decisión: {action_names[action]}")
        print(f"Razón: {action_reason}")

        # Ejecutar acción
        if action == 1 and shares == 0 and balance > 0:  # BUY
            cost_per_share = current_price * 1.001
            shares_to_buy = (balance * strategy.position_size) / cost_per_share
            cost = shares_to_buy * cost_per_share

            if cost <= balance:
                shares = shares_to_buy
                balance -= cost
                entry_price = current_price
                entry_step = i
                trades.append({'type': 'BUY', 'price': current_price, 'shares': shares, 'step': i})
                print(f"Ejecutado: COMPRA de {shares:.0f} acciones a ${current_price:.2f}")

        elif action == 2 and shares > 0:  # SELL
            proceeds = shares * current_price * 0.999
            profit = proceeds - (shares * entry_price)
            balance += proceeds

            trades.append({
                'type': 'SELL',
                'price': current_price,
                'shares': shares,
                'profit': profit,
                'step': i
            })
            shares = 0
            print(f"Ejecutado: VENTA de acciones a ${current_price:.2f} | Profit: ${profit:.2f}")

        # Mostrar portfolio
        portfolio_value = balance + shares * current_price
        print(f"Portfolio: ${portfolio_value:.2f} (Balance: ${balance:.2f} + {shares:.0f} acciones)")

        if i >= days - 1 or (i >= 10 and len(trades) >= 3):  # Mostrar algunos días o cuando hay suficientes trades
            break

    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)

    final_value = balance + shares * current_price
    total_return = ((final_value - 10000) / 10000) * 100

    print(f"Valor final: ${final_value:.2f}")
    print(f"Retorno total: {total_return:.2f}%")
    print(f"Total trades: {len(trades)}")

    if trades:
        profits = [t.get('profit', 0) for t in trades if t['type'] == 'SELL']
        if profits:
            win_rate = len([p for p in profits if p > 0]) / len(profits) * 100
            avg_profit = np.mean(profits)
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Profit promedio por trade: ${avg_profit:.2f}")

    print("\nDetalle de trades:")
    for i, trade in enumerate(trades):
        if trade['type'] == 'BUY':
            print(f"  {i+1}. BUY: {trade['shares']:.0f} acciones @ ${trade['price']:.2f}")
        else:
            print(f"  {i+1}. SELL: {trade['shares']:.0f} acciones @ ${trade['price']:.2f} | P&L: ${trade.get('profit', 0):.2f}")


def main():
    """Función principal"""
    print("🧬 PRUEBA DE ESTRATEGIA AVANZADA GA")
    print("Indicadores: Squeeze Momentum + IFVG + EMAs Institucionales")

    # Cargar datos
    data_path = 'data/processed/SPY_with_indicators.csv'
    if not os.path.exists(data_path):
        print(f"❌ Datos no encontrados: {data_path}")
        print("Ejecutando cálculo de indicadores...")
        # Aquí iría el código para calcular indicadores si no existen
        return

    print(f"Cargando datos de {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"✓ Datos cargados: {len(df)} filas")

    # Verificar que tiene indicadores avanzados
    advanced_cols = ['squeeze_on', 'squeeze_momentum', 'ifvg_bullish_nearby', 'ema_alignment']
    missing_cols = [col for col in advanced_cols if col not in df.columns]

    if missing_cols:
        print(f"❌ Faltan indicadores avanzados: {missing_cols}")
        print("Ejecutando cálculo de indicadores avanzados...")
        from trading_competition.utils.advanced_indicators import add_all_advanced_indicators
        df = add_all_advanced_indicators(df)
        print("✓ Indicadores avanzados calculados")

    # Crear estrategia
    print("\nCreando estrategia GA con genes optimizados...")
    strategy = create_manual_ga_strategy()

    print("Parámetros de la estrategia:")
    print(f"  RSI: Buy={strategy.rsi_buy:.1f}, Sell={strategy.rsi_sell:.1f}")
    print(f"  MACD threshold: {strategy.macd_threshold:.2f}")
    print(f"  Squeeze momentum threshold: {strategy.squeeze_momentum_threshold:.2f}")
    print(f"  Use IFVG filter: {strategy.use_ifvg_filter}")
    print(f"  EMA alignment required: {strategy.ema_alignment_required}")
    print(f"  Stop loss: {strategy.stop_loss:.1%}, Take profit: {strategy.take_profit:.1%}")

    # Simular trading
    simulate_trading_detailed(strategy, df, days=50)

    print("\n✅ Prueba completada exitosamente!")


if __name__ == "__main__":
    main()