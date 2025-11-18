"""
Quick script to check documentation completeness
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from strategy_loader import StrategyLoader

loader = StrategyLoader()
strategies = loader.list_strategies()

print("\n" + "=" * 80)
print("VERIFICACIÓN DE DOCUMENTACIÓN DE ESTRATEGIAS")
print("=" * 80)

missing_docs = []
complete_docs = []

for name in strategies:
    try:
        strategy = loader.get_strategy(name)
        info = strategy.get_detailed_info()
        
        # Check if documentation is complete
        has_description = info.get('description') and info['description'] != 'Not specified'
        has_buy_signals = info.get('buy_signals') and info['buy_signals'] != 'Not specified'
        has_sell_signals = info.get('sell_signals') and info['sell_signals'] != 'Not specified'
        has_indicators = info.get('indicators') and len(info['indicators']) > 0
        
        if has_description and has_buy_signals and has_sell_signals and has_indicators:
            complete_docs.append(name)
            print(f"✅ {name:<35} - COMPLETA")
        else:
            missing_docs.append(name)
            print(f"⚠️  {name:<35} - INCOMPLETA")
            if not has_description:
                print(f"     Falta: descripción")
            if not has_buy_signals:
                print(f"     Falta: señales de compra")
            if not has_sell_signals:
                print(f"     Falta: señales de venta")
            if not has_indicators:
                print(f"     Falta: indicadores")
    except Exception as e:
        print(f"❌ {name:<35} - ERROR: {str(e)}")
        missing_docs.append(name)

print("\n" + "=" * 80)
print(f"RESUMEN:")
print(f"  Total de estrategias: {len(strategies)}")
print(f"  Documentación completa: {len(complete_docs)}")
print(f"  Documentación incompleta: {len(missing_docs)}")
print("=" * 80)

if len(complete_docs) == len(strategies):
    print("\n🎉 ¡TODAS LAS ESTRATEGIAS TIENEN DOCUMENTACIÓN COMPLETA!")
else:
    print(f"\n⚠️  {len(missing_docs)} estrategias necesitan documentación adicional")

print()
