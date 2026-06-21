"""
Test ÁREA 3: Kelly Criterion con Ajuste por Régimen
===================================================
Valida que Kelly ahora:
1. Ajusta por régimen de mercado (bull/bear/chop)
2. Penaliza rachas (correlación serial)
3. Usa lookback adaptativo
4. Calcula estadísticas correctamente
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import pandas as pd


def read_kelly_source():
    """Leer código fuente sin importar"""
    filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                           'core', 'risk', 'kelly_sizer.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def test_kelly_has_regime_adjustment():
    """Test 1: KellySizer tiene método calculate_regime_adjusted_kelly"""
    print("Test 1: Método calculate_regime_adjusted_kelly existe...")
    
    source = read_kelly_source()
    
    assert 'def calculate_regime_adjusted_kelly' in source, \
        "No existe calculate_regime_adjusted_kelly"
    assert 'regime_multiplier' in source, \
        "No hay régimen multiplier"
    assert 'REGIME_MULTIPLIERS' in source, \
        "No hay diccionario de multiplicadores"
    
    print("  ✅ Método de ajuste por régimen implementado")
    return True


def test_kelly_has_streak_penalty():
    """Test 2: Kelly penaliza rachas (correlación serial)"""
    print("Test 2: Penalización por rachas (correlación serial)...")
    
    source = read_kelly_source()
    
    assert 'streak_penalty' in source, "No hay streak_penalty"
    assert 'STREAK_PENALTIES' in source, "No hay diccionario de penalizaciones"
    assert '_count_consecutive_wins' in source, "No cuenta wins consecutivos"
    assert 'consecutive_wins' in source, "No rastrea wins consecutivos"
    
    print("  ✅ Penalización por rachas implementada")
    return True


def test_kelly_has_adaptive_lookback():
    """Test 3: Kelly usa lookback adaptativo"""
    print("Test 3: Lookback adaptativo...")
    
    source = read_kelly_source()
    
    assert 'calculate_adaptive_lookback' in source, \
        "No existe calculate_adaptive_lookback"
    assert 'get_statistics_with_adaptive_lookback' in source, \
        "No existe get_statistics_with_adaptive_lookback"
    assert 'pnl_volatility' in source or 'volatility' in source, \
        "No calcula volatilidad para adaptar"
    
    print("  ✅ Lookback adaptativo implementado")
    return True


def test_regime_multipliers_correct():
    """Test 4: Multiplicadores de régimen son correctos"""
    print("Test 4: Multiplicadores de régimen...")
    
    source = read_kelly_source()
    
    # Verificar valores específicos
    assert "'bull': 1.0" in source, "Bull no es 1.0"
    assert "'bear': 0.5" in source, "Bear no es 0.5"
    assert "'chop': 0.3" in source or "'sideways': 0.3" in source, \
        "Sideways/chop no es 0.3"
    
    # Verificar que bear < bull
    # (esto lo verificamos por los valores literales arriba)
    
    print("  ✅ Multiplicadores: bull=1.0, bear=0.5, chop=0.3")
    return True


def test_kelly_integrates_analysis_engines():
    """Test 5: Kelly integra AnalysisEngines para régimen"""
    print("Test 5: Integración con AnalysisEngines...")
    
    source = read_kelly_source()
    
    assert 'from src.analysis_engines import AnalysisEngines' in source, \
        "No importa AnalysisEngines"
    assert 'ANALYSIS_ENGINES_AVAILABLE' in source, \
        "No tiene flag de disponibilidad"
    assert 'detect_regime_hmm' in source, \
        "No usa detect_regime_hmm"
    
    print("  ✅ Integra AnalysisEngines.detect_regime_hmm()")
    return True


def test_kelly_sizer_basic_functionality():
    """Test 6: KellyPositionSizer funciona básicamente"""
    print("Test 6: Funcionalidad básica de KellyPositionSizer...")
    
    from core.risk.kelly_sizer import KellyPositionSizer, REGIME_MULTIPLIERS, STREAK_PENALTIES
    
    # Crear instancia
    sizer = KellyPositionSizer(kelly_fraction=0.5)
    
    # Test básico
    result = sizer.calculate_kelly_fraction(
        win_rate=0.55,
        win_loss_ratio=1.5
    )
    
    assert result.kelly_fraction > 0, "Kelly fraction debería ser > 0"
    assert result.kelly_full > result.kelly_fraction, "Full > fraction con 0.5"
    
    # Verificar constantes
    assert math.isclose(REGIME_MULTIPLIERS['bull'], 1.0)
    assert math.isclose(REGIME_MULTIPLIERS['bear'], 0.5)
    assert math.isclose(STREAK_PENALTIES[5], 0.5)
    
    print(f"  ✅ Kelly base funciona: fraction={result.kelly_fraction:.3f}")
    return True


def test_kelly_regime_adjusted():
    """Test 7: calculate_regime_adjusted_kelly funciona"""
    print("Test 7: Kelly ajustado por régimen...")
    
    from core.risk.kelly_sizer import KellyPositionSizer
    
    sizer = KellyPositionSizer(kelly_fraction=0.5)
    
    # Crear trade history con racha de wins
    trade_history = pd.DataFrame({
        'pnl': [100, 50, 75, 80, 60],  # 5 wins seguidos
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='h')
    })
    
    # Sin datos de precio (régimen unknown)
    result = sizer.calculate_regime_adjusted_kelly(
        win_rate=0.55,
        win_loss_ratio=1.5,
        price_data=None,
        trade_history=trade_history
    )
    
    assert 'kelly_final' in result, "Falta kelly_final"
    assert 'regime' in result, "Falta regime"
    assert 'streak_penalty' in result, "Falta streak_penalty"
    assert 'consecutive_wins' in result, "Falta consecutive_wins"
    
    # Verificar que detectó la racha
    assert result['consecutive_wins'] == 5, f"Debería ser 5 wins, es {result['consecutive_wins']}"
    assert result['streak_penalty'] > 0, "Debería haber penalización por racha"
    
    # Kelly final debería ser menor que kelly base por la penalización
    assert result['kelly_final'] < result['kelly_base'], \
        "Kelly final debería ser menor por penalización de racha"
    
    print(f"  ✅ Racha detectada: {result['consecutive_wins']} wins")
    print(f"  ✅ Penalización: {result['streak_penalty']*100:.0f}%")
    print(f"  ✅ Kelly: {result['kelly_base']:.3f} → {result['kelly_final']:.3f}")
    return True


def test_kelly_adaptive_lookback():
    """Test 8: Lookback adaptativo funciona"""
    print("Test 8: Lookback adaptativo...")
    
    from core.risk.kelly_sizer import KellyPositionSizer
    
    sizer = KellyPositionSizer()
    
    # Trade history con alta volatilidad
    high_vol_trades = pd.DataFrame({
        'pnl': [100, -80, 150, -120, 90, -100, 200, -150, 50, -60] * 5
    })
    
    # Trade history con baja volatilidad
    low_vol_trades = pd.DataFrame({
        'pnl': [10, 12, 11, 13, 10, 11, 12, 10, 11, 12] * 5
    })
    
    lookback_high = sizer.calculate_adaptive_lookback(high_vol_trades)
    lookback_low = sizer.calculate_adaptive_lookback(low_vol_trades)
    
    # Alta volatilidad debería usar lookback más corto
    # Baja volatilidad debería usar lookback más largo
    assert lookback_high <= lookback_low, \
        f"Alta vol ({lookback_high}) debería tener lookback ≤ baja vol ({lookback_low})"
    
    print(f"  ✅ Alta volatilidad: lookback={lookback_high}")
    print(f"  ✅ Baja volatilidad: lookback={lookback_low}")
    return True


def run_all_tests():
    """Ejecutar todos los tests"""
    print("=" * 60)
    print("ÁREA 3: Kelly Criterion - Tests de Validación")
    print("=" * 60)
    print()
    
    tests = [
        test_kelly_has_regime_adjustment,
        test_kelly_has_streak_penalty,
        test_kelly_has_adaptive_lookback,
        test_regime_multipliers_correct,
        test_kelly_integrates_analysis_engines,
        test_kelly_sizer_basic_functionality,
        test_kelly_regime_adjusted,
        test_kelly_adaptive_lookback,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"  ❌ FALLÓ: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {type(e).__name__}: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    if failed == 0:
        print(f"✅ ÁREA 3 COMPLETADA - {passed}/{passed + failed} tests pasaron")
    else:
        print(f"❌ ÁREA 3 INCOMPLETA - {passed}/{passed + failed} tests pasaron")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
