"""
Test de ÁREA 7: Data Validation Pipeline

Verifica:
1. DataValidator funciona correctamente
2. get_council_context_from_validation() genera contexto válido
3. Council Data Oracle puede usar este contexto

Fecha: 12 de Enero 2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np


def test_data_validator_basic():
    """Test básico de DataValidator."""
    print("Test 1: DataValidator básico...")
    
    from core.data.data_validator import DataValidator, ValidationSeverity
    
    # Crear datos de prueba válidos
    df = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [101, 102, 103, 104, 105],
        'low': [99, 100, 101, 102, 103],
        'close': [100.5, 101.5, 102.5, 103.5, 104.5],
        'volume': [1000, 1100, 1200, 1300, 1400],
        'timestamp': pd.date_range('2025-01-01', periods=5, freq='5min')
    })
    
    validator = DataValidator()
    results = validator.run_all_validations(df)
    summary = validator.get_summary()
    
    assert summary['total_checks'] > 0
    print(f"  ✅ Ejecutó {summary['total_checks']} validaciones")
    print(f"  ✅ Status: {summary['status']}")


def test_council_context_generation():
    """Test de generación de contexto para Council."""
    print("\nTest 2: Council context generation...")
    
    from core.data.data_validator import (
        DataValidator, 
        get_council_context_from_validation
    )
    
    # Datos con algunos problemas
    df = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [101, 102, 103, 104, 105],
        'low': [99, 100, 101, 102, 103],
        'close': [100.5, 101.5, 102.5, 103.5, 104.5],
        'volume': [1000, 0, 1200, 1300, 1400],  # Un volumen 0
        'timestamp': pd.date_range('2025-01-01', periods=5, freq='5min')
    })
    
    validator = DataValidator()
    validator.run_all_validations(df)
    summary = validator.get_summary()
    
    # Generar contexto para Council
    council_context = get_council_context_from_validation(summary)
    
    assert 'data_quality' in council_context
    assert 'score' in council_context['data_quality']
    assert 'validated' in council_context['data_quality']
    
    score = council_context['data_quality']['score']
    print(f"  ✅ Score de calidad: {score:.2f}")
    print(f"  ✅ Issues detectados: {len(council_context['data_quality']['issues'])}")


def test_council_data_quality_rule():
    """Test de la regla de Data Oracle en Council."""
    print("\nTest 3: Council Data Oracle rule...")
    
    from core.council import Council
    
    council = Council()
    council.register_standard_experts()
    
    # Verificar que la regla data_quality existe
    assert 'data_quality' in council.rules
    print("  ✅ Regla 'data_quality' registrada en Council")
    
    # Test con datos de buena calidad
    context_good = {
        'signal': 1,
        'current_equity': 10000,
        'current_dd': 0.02,
        'strategy_id': 'test',
        'data_quality': {
            'validated': True,
            'score': 0.9,
            'has_gaps': False,
            'volume_ok': True,
            'issues': []
        }
    }
    
    decision_good = council.decide(context_good)
    print(f"  ✅ Decisión con datos buenos: {decision_good['decision']}")
    
    # Test con datos de mala calidad
    context_bad = {
        'signal': 1,
        'current_equity': 10000,
        'current_dd': 0.02,
        'strategy_id': 'test',
        'data_quality': {
            'validated': True,
            'score': 0.2,
            'has_gaps': True,
            'volume_ok': False,
            'issues': ['[ERROR] time_gaps: Found gaps', '[CRITICAL] volume: Zero volume']
        }
    }
    
    decision_bad = council.decide(context_bad)
    print(f"  ✅ Decisión con datos malos: {decision_bad['decision']}")


def test_data_validator_with_invalid_ohlc():
    """Test de validación con OHLC inválido."""
    print("\nTest 4: Validación de OHLC inválido...")
    
    from core.data.data_validator import DataValidator
    
    # Datos con OHLC inválido (High < Low)
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [99, 102, 103],  # High < Low en primera fila
        'low': [100, 100, 101],
        'close': [100.5, 101.5, 102.5],
        'volume': [1000, 1100, 1200],
        'timestamp': pd.date_range('2025-01-01', periods=3, freq='5min')
    })
    
    validator = DataValidator()
    result = validator.validate_ohlc_relationships(df)
    
    assert not result.passed
    print(f"  ✅ Detectó OHLC inválido: {result.message}")


def test_data_validator_with_gaps():
    """Test de detección de gaps temporales."""
    print("\nTest 5: Detección de gaps temporales...")
    
    from core.data.data_validator import DataValidator
    
    # Datos con gap (falta una barra)
    timestamps = pd.to_datetime([
        '2025-01-01 00:00:00',
        '2025-01-01 00:05:00',
        '2025-01-01 00:10:00',
        # Falta 00:15:00
        '2025-01-01 00:20:00',
        '2025-01-01 00:25:00',
    ])
    
    df = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [101, 102, 103, 104, 105],
        'low': [99, 100, 101, 102, 103],
        'close': [100.5, 101.5, 102.5, 103.5, 104.5],
        'volume': [1000, 1100, 1200, 1300, 1400],
        'timestamp': timestamps
    })
    
    validator = DataValidator()
    result = validator.detect_time_gaps(df, expected_freq='5min')
    
    assert not result.passed
    print(f"  ✅ Detectó gap temporal: {result.message}")


if __name__ == "__main__":
    results = []
    
    try:
        test_data_validator_basic()
        results.append(("DataValidator básico", True))
    except Exception as e:
        results.append(("DataValidator básico", False))
        print(f"Error: {e}")
    
    try:
        test_council_context_generation()
        results.append(("Council context generation", True))
    except Exception as e:
        results.append(("Council context generation", False))
        print(f"Error: {e}")
    
    try:
        test_council_data_quality_rule()
        results.append(("Council Data Oracle rule", True))
    except Exception as e:
        results.append(("Council Data Oracle rule", False))
        print(f"Error: {e}")
    
    try:
        test_data_validator_with_invalid_ohlc()
        results.append(("Validación OHLC inválido", True))
    except Exception as e:
        results.append(("Validación OHLC inválido", False))
        print(f"Error: {e}")
    
    try:
        test_data_validator_with_gaps()
        results.append(("Detección de gaps", True))
    except Exception as e:
        results.append(("Detección de gaps", False))
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("RESUMEN ÁREA 7 - Data Validation")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ÁREA 7 COMPLETADA - Data Validation Pipeline OK")
    else:
        print("❌ ALGUNOS TESTS FALLARON")
    print("=" * 60)
