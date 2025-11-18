"""
Test Script: Evaluación de Patrones Específicos
Este script prueba los casos específicos mencionados por el usuario
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from scripts.conditional_pattern_evaluator import (
    ConditionalPatternEvaluator, Pattern, Condition, ConditionType
)


def test_specific_patterns():
    """
    Probar los patrones específicos mencionados:
    1. Precio toca EMA22 + Squeeze pendiente negativa + Debajo POC → ¿Se aleja contrario?
    2. Volumen alto + Movimiento importante → ¿Qué parámetros se repiten?
    3. IFVG + Multi-timeframe + Volumen → ¿Probabilidad de éxito?
    """
    
    print("=" * 80)
    print("EVALUACIÓN DE PATRONES CONDICIONALES ESPECÍFICOS")
    print("=" * 80)
    print()
    
    # Cargar datos
    data_path = 'data/btc_15Min.csv'
    if not os.path.exists(data_path):
        print(f"❌ No se encontró el archivo: {data_path}")
        return
    
    print(f"📊 Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"✅ Datos cargados: {len(df)} barras")
    print(f"   Periodo: {df.index[0]} a {df.index[-1]}")
    print()
    
    # Normalizar nombres de columnas a minúsculas
    df.columns = df.columns.str.lower()
    
    # Crear evaluador
    evaluator = ConditionalPatternEvaluator(
        df=df,
        forward_bars=10,  # 10 barras = 2.5 horas en 15min
        profit_threshold=0.01  # 1% profit
    )
    
    print("🔧 Evaluador inicializado")
    print(f"   Forward bars: 10 (2.5 horas)")
    print(f"   Profit threshold: 1.0%")
    print()
    
    # === PATRÓN 1: EMA22 Touch + Squeeze Neg + Below POC ===
    print("=" * 80)
    print("PATRÓN 1: Precio toca EMA22 + Squeeze pendiente negativa + Debajo POC")
    print("Hipótesis: Precio se aleja en dirección contraria (rebote alcista)")
    print("=" * 80)
    print()
    
    pattern1_variants = []
    
    # Variante A: Tolerancia 0.5%
    pattern1_variants.append(Pattern(
        name="EMA22_Touch_0.5%_SqueezeNeg_BelowPOC",
        conditions=[
            Condition(ConditionType.PRICE_NEAR_EMA, {'period': 22, 'tolerance_pct': 0.5}),
            Condition(ConditionType.SQUEEZE_MOMENTUM_SLOPE, {'direction': 'negative'}),
            Condition(ConditionType.PRICE_VS_POC, {'position': 'below'})
        ],
        require_all=True
    ))
    
    # Variante B: Tolerancia 1.0%
    pattern1_variants.append(Pattern(
        name="EMA22_Touch_1.0%_SqueezeNeg_BelowPOC",
        conditions=[
            Condition(ConditionType.PRICE_NEAR_EMA, {'period': 22, 'tolerance_pct': 1.0}),
            Condition(ConditionType.SQUEEZE_MOMENTUM_SLOPE, {'direction': 'negative'}),
            Condition(ConditionType.PRICE_VS_POC, {'position': 'below'})
        ],
        require_all=True
    ))
    
    # Variante C: Solo EMA22 + Squeeze (sin POC)
    pattern1_variants.append(Pattern(
        name="EMA22_Touch_SqueezeNeg_Only",
        conditions=[
            Condition(ConditionType.PRICE_NEAR_EMA, {'period': 22, 'tolerance_pct': 0.5}),
            Condition(ConditionType.SQUEEZE_MOMENTUM_SLOPE, {'direction': 'negative'})
        ],
        require_all=True
    ))
    
    results1 = evaluator.evaluate_patterns(pattern1_variants)
    
    print("📊 RESULTADOS PATRÓN 1:\n")
    for result in results1:
        print(f"Variante: {result.pattern.name}")
        print(f"  Ocurrencias: {result.occurrences}")
        print(f"  Win Rate: {result.win_rate:.2%}")
        print(f"  Expectancy: {result.expectancy:.4f}")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"  Avg Profit: {result.avg_profit:.4f} | Avg Loss: {result.avg_loss:.4f}")
        print()
    
    # === PATRÓN 2: Volumen Alto + Movimiento Grande ===
    print("=" * 80)
    print("PATRÓN 2: Volumen alto + Movimiento importante del precio")
    print("Objetivo: Identificar qué parámetros se repiten en estos casos")
    print("=" * 80)
    print()
    
    pattern2_variants = []
    
    # Variante A: Volume 2x, Movement 2%
    pattern2_variants.append(Pattern(
        name="HighVol_2x_LargeMove_2%",
        conditions=[
            Condition(ConditionType.VOLUME_HIGH, {'multiplier': 2.0}),
            Condition(ConditionType.PRICE_MOVEMENT_LARGE, {'threshold_pct': 2.0})
        ],
        require_all=True
    ))
    
    # Variante B: Volume 1.5x, Movement 1.5%
    pattern2_variants.append(Pattern(
        name="HighVol_1.5x_LargeMove_1.5%",
        conditions=[
            Condition(ConditionType.VOLUME_HIGH, {'multiplier': 1.5}),
            Condition(ConditionType.PRICE_MOVEMENT_LARGE, {'threshold_pct': 1.5})
        ],
        require_all=True
    ))
    
    # Variante C: Volume 3x, Movement 3%
    pattern2_variants.append(Pattern(
        name="HighVol_3x_LargeMove_3%",
        conditions=[
            Condition(ConditionType.VOLUME_HIGH, {'multiplier': 3.0}),
            Condition(ConditionType.PRICE_MOVEMENT_LARGE, {'threshold_pct': 3.0})
        ],
        require_all=True
    ))
    
    # Variante D: Solo volumen alto (para comparar)
    pattern2_variants.append(Pattern(
        name="HighVol_2x_Only",
        conditions=[
            Condition(ConditionType.VOLUME_HIGH, {'multiplier': 2.0})
        ],
        require_all=True
    ))
    
    # Variante E: Solo movimiento grande (para comparar)
    pattern2_variants.append(Pattern(
        name="LargeMove_2%_Only",
        conditions=[
            Condition(ConditionType.PRICE_MOVEMENT_LARGE, {'threshold_pct': 2.0})
        ],
        require_all=True
    ))
    
    results2 = evaluator.evaluate_patterns(pattern2_variants)
    
    print("📊 RESULTADOS PATRÓN 2:\n")
    for result in results2:
        print(f"Variante: {result.pattern.name}")
        print(f"  Ocurrencias: {result.occurrences}")
        print(f"  Win Rate: {result.win_rate:.2%}")
        print(f"  Expectancy: {result.expectancy:.4f}")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        
        if result.best_params:
            print(f"  Parámetros comunes en trades ganadores:")
            print(f"    - Avg forward return: {result.best_params.get('avg_forward_return', 0):.4f}")
            print(f"    - Max forward return: {result.best_params.get('max_forward_return', 0):.4f}")
        print()
    
    # === PATRÓN 3: IFVG + Volumen ===
    print("=" * 80)
    print("PATRÓN 3: IFVG + Volumen alto")
    print("Objetivo: Evaluar probabilidad de éxito de combinación IFVG + Confirmación")
    print("=" * 80)
    print()
    
    pattern3_variants = []
    
    # Variante A: IFVG Bullish + High Volume
    pattern3_variants.append(Pattern(
        name="IFVG_Bullish_HighVol",
        conditions=[
            Condition(ConditionType.IFVG_PRESENT, {'direction': 'bullish'}),
            Condition(ConditionType.VOLUME_HIGH, {'multiplier': 1.5})
        ],
        require_all=True
    ))
    
    # Variante B: IFVG Bearish + High Volume
    pattern3_variants.append(Pattern(
        name="IFVG_Bearish_HighVol",
        conditions=[
            Condition(ConditionType.IFVG_PRESENT, {'direction': 'bearish'}),
            Condition(ConditionType.VOLUME_HIGH, {'multiplier': 1.5})
        ],
        require_all=True
    ))
    
    # Variante C: IFVG Any + High Volume
    pattern3_variants.append(Pattern(
        name="IFVG_Any_HighVol",
        conditions=[
            Condition(ConditionType.IFVG_PRESENT, {'direction': 'any'}),
            Condition(ConditionType.VOLUME_HIGH, {'multiplier': 1.5})
        ],
        require_all=True
    ))
    
    # Variante D: IFVG + High Volume + EMA Alignment
    pattern3_variants.append(Pattern(
        name="IFVG_HighVol_EMA_Bullish",
        conditions=[
            Condition(ConditionType.IFVG_PRESENT, {'direction': 'bullish'}),
            Condition(ConditionType.VOLUME_HIGH, {'multiplier': 1.5}),
            Condition(ConditionType.EMA_CROSS, {'fast': 9, 'slow': 21, 'direction': 'bullish'})
        ],
        require_all=True
    ))
    
    # Variante E: IFVG Solo (para comparar)
    pattern3_variants.append(Pattern(
        name="IFVG_Only",
        conditions=[
            Condition(ConditionType.IFVG_PRESENT, {'direction': 'any'})
        ],
        require_all=True
    ))
    
    results3 = evaluator.evaluate_patterns(pattern3_variants)
    
    print("📊 RESULTADOS PATRÓN 3:\n")
    for result in results3:
        print(f"Variante: {result.pattern.name}")
        print(f"  Ocurrencias: {result.occurrences}")
        print(f"  Win Rate: {result.win_rate:.2%}")
        print(f"  Expectancy: {result.expectancy:.4f}")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print()
    
    # === RESUMEN COMPARATIVO ===
    print("=" * 80)
    print("📊 RESUMEN COMPARATIVO - MEJORES PATRONES")
    print("=" * 80)
    print()
    
    all_results = results1 + results2 + results3
    all_results.sort(key=lambda x: x.expectancy, reverse=True)
    
    print("Top 10 patrones por Expectancy:\n")
    for i, result in enumerate(all_results[:10], 1):
        print(f"{i}. {result.pattern.name}")
        print(f"   Win Rate: {result.win_rate:.2%} | Expectancy: {result.expectancy:.4f} | PF: {result.profit_factor:.2f}")
        print(f"   Ocurrencias: {result.occurrences}")
        print()
    
    # === ANÁLISIS DE INSIGHTS ===
    print("=" * 80)
    print("💡 INSIGHTS Y RECOMENDACIONES")
    print("=" * 80)
    print()
    
    # Insight 1: EMA22 Touch
    best_ema_pattern = max(results1, key=lambda x: x.expectancy)
    print("1️⃣ PATRÓN EMA22 TOUCH:")
    print(f"   Mejor variante: {best_ema_pattern.pattern.name}")
    print(f"   Win Rate: {best_ema_pattern.win_rate:.2%}")
    if best_ema_pattern.win_rate > 0.55:
        print("   ✅ CONFIRMADO: El precio tiende a alejarse cuando toca EMA22 con squeeze negativo")
        print("   💡 Recomendación: Implementar entrada en rebote desde EMA22")
    else:
        print("   ⚠️ Sin edge significativo: Probar con otros timeframes o EMAs")
    print()
    
    # Insight 2: Volume + Movement
    best_vol_pattern = max(results2, key=lambda x: x.expectancy)
    print("2️⃣ PATRÓN VOLUMEN ALTO + MOVIMIENTO:")
    print(f"   Mejor variante: {best_vol_pattern.pattern.name}")
    print(f"   Win Rate: {best_vol_pattern.win_rate:.2%}")
    print(f"   Ocurrencias: {best_vol_pattern.occurrences}")
    if best_vol_pattern.best_params:
        print(f"   Parámetros óptimos identificados:")
        for key, val in best_vol_pattern.best_params.items():
            if isinstance(val, float):
                print(f"     - {key}: {val:.4f}")
    print()
    
    # Insight 3: IFVG
    best_ifvg_pattern = max(results3, key=lambda x: x.expectancy)
    print("3️⃣ PATRÓN IFVG:")
    print(f"   Mejor variante: {best_ifvg_pattern.pattern.name}")
    print(f"   Win Rate: {best_ifvg_pattern.win_rate:.2%}")
    if best_ifvg_pattern.win_rate > 0.55:
        print("   ✅ IFVG + Confirmación tiene edge predictivo")
        print("   💡 Recomendación: Usar IFVG con filtro de volumen")
    else:
        print("   ⚠️ IFVG solo no es suficiente: Requiere confirmación adicional")
    print()
    
    # Generar reporte completo
    print("=" * 80)
    print("💾 GENERANDO REPORTE COMPLETO...")
    print("=" * 80)
    print()
    
    report_file = evaluator.generate_report(
        all_results, 
        filename="reports/specific_patterns_evaluation.md"
    )
    
    print(f"✅ Reporte guardado en: {report_file}")
    print()
    print("🎯 SIGUIENTE PASO:")
    print("   1. Revisar reporte detallado")
    print("   2. Implementar mejores patrones en estrategia")
    print("   3. Backtestear con stops/targets apropiados")
    print()


if __name__ == "__main__":
    test_specific_patterns()
