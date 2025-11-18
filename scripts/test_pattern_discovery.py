"""
Test Pattern Discovery Analyzer
Ejecuta el análisis de patrones predictivos con datos de muestra
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.pattern_discovery_analyzer import run_pattern_discovery


def generate_realistic_data(bars: int = 3000) -> pd.DataFrame:
    """
    Genera datos realistas con diferentes regímenes de mercado
    """
    np.random.seed(42)
    
    start_date = datetime(2024, 1, 1)
    dates = pd.date_range(start=start_date, periods=bars, freq='5min')
    
    # Generar precio con tendencias y volatilidad variable
    base_price = 50000
    prices = [base_price]
    
    # Crear diferentes regímenes
    for i in range(1, bars):
        # Régimen alcista
        if i < bars * 0.3:
            trend = 0.0003
            volatility = 0.003
        # Régimen lateral
        elif i < bars * 0.6:
            trend = 0.0
            volatility = 0.002
        # Régimen bajista
        else:
            trend = -0.0002
            volatility = 0.004
        
        # Añadir ruido y tendencia
        change = np.random.normal(trend, volatility)
        
        # Ocasionalmente crear gaps (IFVG)
        if np.random.random() < 0.05:  # 5% de probabilidad de gap
            change *= 2
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Generar OHLCV
    high = prices * (1 + np.abs(np.random.normal(0, 0.005, bars)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.005, bars)))
    open_price = np.roll(prices, 1)
    open_price[0] = prices[0]
    
    # Volumen con picos ocasionales
    base_volume = 1000000
    volume = base_volume * (1 + np.abs(np.random.normal(0, 0.5, bars)))
    
    # Aumentar volumen en movimientos grandes
    price_change = np.abs(np.diff(prices, prepend=prices[0]))
    volume += price_change * 10000000
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    }, index=dates)
    
    return df


def main():
    print("=" * 70)
    print("ANÁLISIS DE PATRONES PREDICTIVOS")
    print("=" * 70)
    print()
    print("Este análisis identifica qué condiciones predicen mejor los movimientos")
    print("de precio, respondiendo preguntas como:")
    print()
    print("  • ¿Qué tan efectivo es operar cerca de EMAs?")
    print("  • ¿El volumen y POC realmente importan?")
    print("  • ¿Los IFVG son predictivos?")
    print("  • ¿El Squeeze Momentum funciona?")
    print("  • ¿Las tendencias en 15m/1h ayudan en 5m?")
    print()
    print("=" * 70)
    
    # Generar datos
    print("\n1. Generando datos de mercado realistas...")
    df = generate_realistic_data(3000)
    print(f"   ✓ Generados {len(df)} bars (5 minutos)")
    print(f"   ✓ Período: {df.index[0]} a {df.index[-1]}")
    print(f"   ✓ Precio inicial: ${df['close'].iloc[0]:.2f}")
    print(f"   ✓ Precio final: ${df['close'].iloc[-1]:.2f}")
    print(f"   ✓ Cambio total: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")
    
    # Ejecutar análisis
    print("\n2. Ejecutando análisis de patrones...")
    print("   (Esto puede tardar 1-2 minutos)")
    print()
    
    try:
        all_results, report = run_pattern_discovery(
            df, 
            min_occurrences=15,  # Mínimo 15 casos para considerar un patrón
            output_file="pattern_discovery_results.md"
        )
        
        print("\n" + "=" * 70)
        print("✓ ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("=" * 70)
        print()
        print("Archivos generados:")
        print("  • pattern_discovery_results.md - Reporte completo")
        print()
        print("Próximos pasos:")
        print("  1. Revisa el reporte para ver los patrones más rentables")
        print("  2. Identifica los patrones con mayor Profit Factor (>2.0)")
        print("  3. Verifica la confianza (más ⭐ es mejor)")
        print("  4. Implementa los patrones en tu estrategia")
        print()
        
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
