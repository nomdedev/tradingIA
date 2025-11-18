"""
Strategy Documentation Utility
Provides easy access to all strategy documentation
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from strategy_loader import StrategyLoader


def print_all_strategies_info():
    """Print detailed information for all available strategies"""
    loader = StrategyLoader()
    strategies = loader.list_strategies()
    
    print("=" * 80)
    print("TRADING STRATEGIES DOCUMENTATION")
    print("=" * 80)
    print()
    
    for strategy_name in sorted(strategies):
        try:
            strategy = loader.get_strategy(strategy_name)
            info = strategy.get_detailed_info()
            
            print(f"\n{'=' * 80}")
            print(f"📊 {info['name']}")
            print(f"{'=' * 80}")
            print()
            
            print(f"📝 Description:")
            print(f"   {info['description']}")
            print()
            
            print(f"📈 Buy Signals:")
            for line in info['buy_signals'].split('\n'):
                print(f"   {line}")
            print()
            
            print(f"📉 Sell Signals:")
            for line in info['sell_signals'].split('\n'):
                print(f"   {line}")
            print()
            
            print(f"🎯 Risk Level: {info['risk_level']}")
            print(f"⏰ Timeframe: {info['timeframe']}")
            print(f"📊 Indicators: {', '.join(info['indicators'])}")
            print()
            
            if info.get('parameters'):
                print(f"⚙️  Parameters:")
                if isinstance(info['parameters'], dict):
                    for param, value in info['parameters'].items():
                        print(f"   • {param}: {value}")
                print()
            
        except Exception as e:
            print(f"Error loading {strategy_name}: {e}")
            print()


def get_strategy_info(strategy_name: str):
    """Get detailed information for a specific strategy"""
    loader = StrategyLoader()
    
    try:
        strategy = loader.get_strategy(strategy_name)
        return strategy.get_detailed_info()
    except Exception as e:
        return {
            'error': str(e),
            'name': strategy_name,
            'description': 'Strategy not found or error loading'
        }


def list_strategies_by_type():
    """List strategies organized by type"""
    loader = StrategyLoader()
    strategies = loader.list_strategies()
    
    categorized = {
        'Reversión a la Media': [],
        'Seguimiento de Tendencia': [],
        'Breakout': [],
        'Avanzadas': []
    }
    
    type_mapping = {
        'bollinger_bands': 'Reversión a la Media',
        'rsi_mean_reversion': 'Reversión a la Media',
        'macd_momentum': 'Seguimiento de Tendencia',
        'ma_crossover': 'Seguimiento de Tendencia',
        'volume_breakout': 'Breakout',
        'oracle_numeris_safeguard': 'Avanzadas',
        'squeeze_adx_ttm': 'Avanzadas',
        'squeeze_adx_ttm_ema15m50': 'Avanzadas',
        'vp_ifvg_ema_ema15m50': 'Avanzadas'
    }
    
    for strategy_name in strategies:
        category = type_mapping.get(strategy_name, 'Avanzadas')
        categorized[category].append(strategy_name)
    
    print("\n" + "=" * 80)
    print("ESTRATEGIAS POR CATEGORÍA")
    print("=" * 80)
    
    for category, strats in categorized.items():
        if strats:
            print(f"\n{category}:")
            for strat in sorted(strats):
                print(f"  • {strat}")
    print()


def compare_strategies(strategy_names: list):
    """Compare multiple strategies side by side"""
    loader = StrategyLoader()
    
    print("\n" + "=" * 100)
    print("COMPARACIÓN DE ESTRATEGIAS")
    print("=" * 100)
    
    comparison_data = []
    for strategy_name in strategy_names:
        try:
            strategy = loader.get_strategy(strategy_name)
            info = strategy.get_detailed_info()
            comparison_data.append(info)
        except Exception as e:
            print(f"Error loading {strategy_name}: {e}")
    
    if not comparison_data:
        print("No strategies to compare")
        return
    
    # Print comparison table
    print(f"\n{'Strategy':<30} {'Risk Level':<15} {'Timeframe':<10} {'Indicators':<50}")
    print("-" * 105)
    
    for info in comparison_data:
        indicators_str = ', '.join(info['indicators'][:3])
        if len(info['indicators']) > 3:
            indicators_str += '...'
        
        print(f"{info['name']:<30} {info['risk_level']:<15} {info['timeframe']:<10} {indicators_str:<50}")
    
    print()


def show_readme_location():
    """Show location of README documentation"""
    readme_path = os.path.join(os.path.dirname(__file__), 'presets', 'README.md')
    
    print("\n" + "=" * 80)
    print("DOCUMENTACIÓN COMPLETA")
    print("=" * 80)
    print()
    print(f"📚 README completo disponible en:")
    print(f"   {readme_path}")
    print()
    print("📖 Para ver la documentación completa, abre el archivo README.md")
    print("   Contiene:")
    print("   • Descripción detallada de cada estrategia")
    print("   • Ejemplos de uso")
    print("   • Mejores condiciones de mercado")
    print("   • Comparación de estrategias")
    print("   • Guía de desarrollo de estrategias personalizadas")
    print()
    
    if os.path.exists(readme_path):
        print("✅ Archivo encontrado")
    else:
        print("⚠️  Archivo no encontrado")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Strategy Documentation Utility')
    parser.add_argument('--all', action='store_true', help='Show all strategies info')
    parser.add_argument('--list', action='store_true', help='List strategies by type')
    parser.add_argument('--strategy', type=str, help='Show info for specific strategy')
    parser.add_argument('--compare', nargs='+', help='Compare multiple strategies')
    parser.add_argument('--readme', action='store_true', help='Show README location')
    
    args = parser.parse_args()
    
    if args.all:
        print_all_strategies_info()
    elif args.list:
        list_strategies_by_type()
    elif args.strategy:
        info = get_strategy_info(args.strategy)
        if 'error' in info:
            print(f"Error: {info['error']}")
        else:
            print(f"\n{info['name']}")
            print("=" * 80)
            print(f"\n{info['description']}")
            print(f"\n{info['buy_signals']}")
            print(f"\n{info['sell_signals']}")
    elif args.compare:
        compare_strategies(args.compare)
    elif args.readme:
        show_readme_location()
    else:
        # Default: show menu
        print("\n" + "=" * 80)
        print("ESTRATEGIAS DE TRADING - UTILIDAD DE DOCUMENTACIÓN")
        print("=" * 80)
        print()
        print("Opciones disponibles:")
        print()
        print("  --all              Mostrar información de todas las estrategias")
        print("  --list             Listar estrategias por categoría")
        print("  --strategy NAME    Mostrar info de estrategia específica")
        print("  --compare A B C    Comparar múltiples estrategias")
        print("  --readme           Mostrar ubicación del README completo")
        print()
        print("Ejemplos:")
        print("  python strategy_docs.py --all")
        print("  python strategy_docs.py --list")
        print("  python strategy_docs.py --strategy bollinger_bands")
        print("  python strategy_docs.py --compare ma_crossover rsi_mean_reversion")
        print("  python strategy_docs.py --readme")
        print()
        
        # Show quick overview
        show_readme_location()
        list_strategies_by_type()
