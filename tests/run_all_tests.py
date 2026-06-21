"""
Ejecutar todos los tests y mostrar resumen.

Test Suites para las 8 ÁREAS CRÍTICAS:
- ÁREA 1: Look-Ahead Bias Fix
- ÁREA 2: Walk-Forward Analysis  
- ÁREA 3: Kelly con Régimen
- ÁREA 4: Council (validado en comparison)
- ÁREA 5: Market Impact Crypto
- ÁREA 6: Risk Manager Mejorado
- ÁREA 7: Data Validation
- ÁREA 8: TradingSignal Standard
"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

tests = [
    ('tests/test_no_lookahead_simple.py', 'ÁREA 1: Look-Ahead Bias'),
    ('tests/test_area2_wfa.py', 'ÁREA 2: Walk-Forward Analysis'),
    ('tests/test_area3_kelly.py', 'ÁREA 3: Kelly con Régimen'),
    ('tests/test_area5_market_impact_crypto.py', 'ÁREA 5: Market Impact Crypto'),
    ('tests/test_area6_risk_manager.py', 'ÁREA 6: Risk Manager Mejorado'),
    ('tests/test_area7_data_validation.py', 'ÁREA 7: Data Validation'),
    ('tests/test_area8_trading_signal.py', 'ÁREA 8: TradingSignal Standard'),
    ('tests/test_comparison_backtest.py', 'Comparison (valida ÁREA 4)'),
]

print("="*70)
print("   TRADING IA - TEST SUITE COMPLETO (8 ÁREAS CRÍTICAS)")
print("="*70)

results = []
for test_path, desc in tests:
    print(f"\n>>> {desc}")
    result = subprocess.run([sys.executable, test_path], capture_output=False)
    results.append((desc, result.returncode))

print("\n" + "="*70)
print("   RESUMEN FINAL")
print("="*70)

for desc, rc in results:
    status = "✅ PASS" if rc == 0 else "❌ FAIL"
    print(f"  {status} - {desc}")

passed = sum(1 for _, r in results if r == 0)
failed = len(results) - passed

print(f"\n  Total: {len(results)} suites | Passed: {passed} | Failed: {failed}")

if failed == 0:
    print("\n" + "🎉"*20)
    print("   ¡TODAS LAS 8 ÁREAS CRÍTICAS VALIDADAS!")
    print("🎉"*20)
else:
    print("\n❌ Hay tests fallidos. Revisar output arriba.")

print("="*70)
