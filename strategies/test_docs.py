"""Quick verification of strategy documentation"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from strategy_loader import StrategyLoader

print('=== VERIFICACIÓN FINAL ===\n')

loader = StrategyLoader()
strategies = loader.list_strategies()

print(f'Total estrategias: {len(strategies)}\n')

for name in sorted(strategies):
    try:
        strategy = loader.get_strategy(name)
        info = strategy.get_detailed_info()
        print(f'✅ {name}')
        print(f'   Riesgo: {info["risk_level"]} | Timeframe: {info["timeframe"]} | Indicadores: {len(info["indicators"])}')
    except Exception as e:
        print(f'❌ {name} - Error: {str(e)}')

print('\n=== FIN ===')
