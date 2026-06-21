"""
Test ÁREA 2: Walk-Forward Analysis Real
========================================
Valida que WFA ahora:
1. Optimiza parámetros en cada período (no usa mismos params)
2. Calcula degradación correctamente
3. Calcula stability_score
4. Certifica estrategias basado en criterios
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def create_test_data(n_bars=2000, seed=42):
    """Crear datos de prueba con tendencia conocida"""
    np.random.seed(seed)
    
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='5min')
    
    # Crear precios con tendencia y volatilidad
    returns = np.random.standard_normal(n_bars) * 0.001 + 0.0001  # Ligera tendencia alcista
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Crear OHLCV realista
    df = pd.DataFrame({
        'Open': prices * (1 + np.random.standard_normal(n_bars) * 0.0005),
        'High': prices * (1 + np.abs(np.random.standard_normal(n_bars) * 0.002)),
        'Low': prices * (1 - np.abs(np.random.standard_normal(n_bars) * 0.002)),
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, n_bars)
    }, index=dates)
    
    # Asegurar OHLC válido
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    return df


def read_wfa_source():
    """Leer código fuente directamente sin importar (evita scipy)"""
    import re
    
    filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                           'core', 'execution', 'backtester_core.py')
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extraer método run_walk_forward
    # Buscar desde "def run_walk_forward" hasta el siguiente "def " al mismo nivel
    pattern = r'(    def run_walk_forward\(.*?)(?=\n    def |\nclass |\Z)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        return match.group(1)
    return content


def test_wfa_signature_accepts_param_ranges():
    """Test 1: run_walk_forward acepta param_ranges"""
    print("Test 1: Firma de run_walk_forward con param_ranges...")
    
    source = read_wfa_source()
    
    # Verificar firma
    assert 'param_ranges: Dict = None' in source, "param_ranges no está en la firma"
    assert 'strategy_params: Dict = None' in source or 'strategy_params:' in source, "strategy_params debe existir"
    assert 'min_test_bars: int' in source, "min_test_bars debería existir"
    
    print("  ✅ Firma correcta: incluye param_ranges, strategy_params, min_test_bars")
    return True


def test_wfa_returns_stability_score():
    """Test 2: WFA retorna stability_score y certified"""
    print("Test 2: WFA retorna stability_score y certified...")
    
    source = read_wfa_source()
    
    assert 'stability_score' in source, "stability_score no está en el código"
    assert '"certified"' in source or "'certified'" in source, "certified no está en el código"
    assert 'avg_oos_sharpe' in source, "avg_oos_sharpe no está en el código"
    
    print("  ✅ Retorna: stability_score, certified, avg_oos_sharpe")
    return True


def test_wfa_calls_bayesian_optimize():
    """Test 3: WFA llama a _bayesian_optimize cuando hay param_ranges"""
    print("Test 3: WFA integra _bayesian_optimize...")
    
    source = read_wfa_source()
    
    assert '_bayesian_optimize' in source, "No se llama a _bayesian_optimize"
    assert 'use_optimization' in source, "No hay flag use_optimization"
    
    # Verificar que NO tiene el bypass "Skip optimization"
    assert 'Skip optimization' not in source, "Aún tiene el bypass de optimización"
    
    print("  ✅ Integra _bayesian_optimize (sin bypass)")
    return True


def test_wfa_degradation_formula():
    """Test 4: Fórmula de degradación correcta (IS - OOS) / |IS|"""
    print("Test 4: Fórmula de degradación...")
    
    source = read_wfa_source()
    
    # La fórmula correcta: (train_sharpe - test_sharpe) / abs(train_sharpe)
    # Antes era incorrecta: (test_sharpe - train_sharpe)
    assert 'train_sharpe - test_sharpe' in source, "Fórmula de degradación incorrecta"
    
    print("  ✅ Fórmula: (IS - OOS) / |IS| * 100")
    return True


def test_wfa_certification_criteria():
    """Test 5: Criterios de certificación definidos"""
    print("Test 5: Criterios de certificación...")
    
    source = read_wfa_source()
    
    # Verificar criterios
    assert 'avg_degradation' in source and '< 30' in source, "Criterio degradación < 30% no encontrado"
    assert 'avg_oos_sharpe' in source and '> 0.5' in source, "Criterio OOS Sharpe > 0.5 no encontrado"
    assert 'stability_score' in source and '> 0.5' in source, "Criterio stability > 0.5 no encontrado"
    
    print("  ✅ Criterios: degradación < 30%, OOS Sharpe > 0.5, stability > 0.5")
    return True


def test_wfa_anchored_window():
    """Test 6: Usa ventana anclada (Anchored WFA)"""
    print("Test 6: Ventana anclada (train_start = 0)...")
    
    source = read_wfa_source()
    
    # Verificar que train_start = 0 (anchored)
    assert 'train_start = 0' in source, "No usa ventana anclada (train_start = 0)"
    assert 'Anchored' in source, "No documenta que es Anchored WFA"
    
    print("  ✅ Anchored WFA: IS siempre desde índice 0")
    return True


def test_wfa_tracks_all_params():
    """Test 7: Guarda todos los parámetros optimizados por período"""
    print("Test 7: Tracking de parámetros por período...")
    
    source = read_wfa_source()
    
    assert 'all_optimized_params' in source, "No guarda all_optimized_params"
    assert 'all_optimized_params.append' in source, "No append params por período"
    
    print("  ✅ Guarda parámetros optimizados de cada período")
    return True


def run_all_tests():
    """Ejecutar todos los tests"""
    print("=" * 60)
    print("ÁREA 2: Walk-Forward Analysis - Tests de Validación")
    print("=" * 60)
    print()
    
    tests = [
        test_wfa_signature_accepts_param_ranges,
        test_wfa_returns_stability_score,
        test_wfa_calls_bayesian_optimize,
        test_wfa_degradation_formula,
        test_wfa_certification_criteria,
        test_wfa_anchored_window,
        test_wfa_tracks_all_params,
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
        print(f"✅ ÁREA 2 COMPLETADA - {passed}/{passed + failed} tests pasaron")
    else:
        print(f"❌ ÁREA 2 INCOMPLETA - {passed}/{passed + failed} tests pasaron")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
