"""
BTC Strategy Tester - Sistema de Comparación de Estrategias

Sistema completo para comparar y rankear todas las estrategias de trading implementadas.
Realiza análisis estadístico, ranking de performance, y genera recomendaciones finales.

Estrategias comparadas:
1. Mean Reversion IBS + BB
2. Momentum MACD + ADX
3. Pairs Trading Cointegration
4. HFT Momentum Kalman VMA
5. LSTM ML Mean Reversion

Características:
- Ranking basado en múltiples métricas (Sharpe, robustez, win rate)
- Análisis estadístico de significancia
- Matriz de correlación de retornos
- Ensemble strategy optimization
- Reportes y visualizaciones completas
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuración
RESULTS_DIR = Path("results/strategy_comparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Estrategias disponibles
STRATEGIES = {
    'mean_reversion_ibs_bb': {
        'name': 'Mean Reversion IBS + BB',
        'file': 'results/mean_reversion_ibs_bb/metrics.json',
        'type': 'mean_reversion'
    },
    'momentum_macd_adx': {
        'name': 'Momentum MACD + ADX',
        'file': 'results/momentum_macd_adx/metrics.json',
        'type': 'momentum'
    },
    'pairs_trading_cointegration': {
        'name': 'Pairs Trading Cointegration',
        'file': 'results/pairs_trading_cointegration/metrics.json',
        'type': 'pairs_trading'
    },
    'hft_momentum_vma': {
        'name': 'HFT Momentum Kalman VMA',
        'file': 'results/hft_momentum_vma/metrics.json',
        'type': 'hft'
    },
    'lstm_ml_reversion': {
        'name': 'LSTM ML Mean Reversion',
        'file': 'results/lstm_ml_reversion/metrics.json',
        'type': 'ml'
    }
}


class StrategyComparator:
    """
    Clase para comparar y rankear estrategias de trading
    """

    def __init__(self):
        self.strategies_data = {}
        self.ranking_df = None
        self.correlation_matrix = None

    def load_strategy_results(self):
        """Cargar resultados de todas las estrategias"""
        print("Cargando resultados de estrategias...")

        for strategy_key, strategy_info in STRATEGIES.items():
            try:
                with open(strategy_info['file'], 'r') as f:
                    data = json.load(f)
                    self.strategies_data[strategy_key] = {
                        'info': strategy_info,
                        'metrics': data
                    }
                print(f"✓ {strategy_info['name']} cargado")
            except FileNotFoundError:
                print(f"✗ {strategy_info['name']} no encontrado - archivo faltante")
            except Exception as e:
                print(f"✗ Error cargando {strategy_info['name']}: {e}")

        print(f"\nTotal estrategias cargadas: {len(self.strategies_data)}")

    def create_ranking_dataframe(self):
        """Crear DataFrame con métricas de ranking"""
        if not self.strategies_data:
            print("No hay datos de estrategias para rankear")
            return

        ranking_data = []

        for strategy_key, data in self.strategies_data.items():
            metrics = data['metrics']

            # Extraer métricas clave
            row = {
                'strategy': strategy_key,
                'name': data['info']['name'],
                'type': data['info']['type'],
                'sharpe_oos': metrics.get('avg_oos_sharpe', 0),
                'sharpe_std': metrics.get('std_oos_sharpe', 0),
                'win_rate': metrics.get('avg_oos_win_rate', 0),
                'total_return': metrics.get('total_return', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'robustness': 1 if metrics.get('robustness_overall', False) else 0,
                'ab_significant': 1 if metrics.get('ab_test_significant', False) else 0,
                'sensitivity_stable': 1 if metrics.get('sensitivity_stable', False) else 0,
                'overfit_risk': 0 if metrics.get('anti_overfit_risk') == 'low' else 1,
                'recommendation': metrics.get('recommendation', 'Unknown')
            }

            # Métricas específicas por tipo de estrategia
            if data['info']['type'] == 'hft':
                row['avg_holding_period'] = metrics.get('avg_holding_period_min', 0)
                row['avg_slippage'] = metrics.get('avg_slippage_bps', 0)
            elif data['info']['type'] == 'ml':
                row['model_accuracy'] = metrics.get('avg_model_accuracy', 0)

            ranking_data.append(row)

        self.ranking_df = pd.DataFrame(ranking_data)
        return self.ranking_df

    def calculate_composite_score(self):
        """Calcular score compuesto para ranking"""
        if self.ranking_df is None:
            return

        df = self.ranking_df.copy()

        # Normalizar métricas (0-1 scale, higher is better)
        metrics_to_normalize = ['sharpe_oos', 'win_rate', 'profit_factor']
        for metric in metrics_to_normalize:
            if metric in df.columns:
                min_val = df[metric].min()
                max_val = df[metric].max()
                if max_val > min_val:
                    df[f'{metric}_norm'] = (df[metric] - min_val) / (max_val - min_val)
                else:
                    df[f'{metric}_norm'] = 0.5  # Si todos iguales

        # Invertir drawdown (menor es mejor)
        if 'max_drawdown' in df.columns:
            min_dd = df['max_drawdown'].min()
            max_dd = df['max_drawdown'].max()
            if max_dd > min_dd:
                df['drawdown_norm'] = 1 - (df['max_drawdown'] - min_dd) / (max_dd - min_dd)
            else:
                df['drawdown_norm'] = 0.5

        # Pesos para score compuesto
        weights = {
            'sharpe_oos_norm': 0.3,
            'win_rate_norm': 0.2,
            'profit_factor_norm': 0.2,
            'drawdown_norm': 0.15,
            'robustness': 0.1,
            'ab_significant': 0.05
        }

        # Calcular score compuesto
        df['composite_score'] = 0
        for metric, weight in weights.items():
            if metric in df.columns:
                df['composite_score'] += df[metric] * weight

        # Ranking final
        df['rank'] = df['composite_score'].rank(ascending=False).astype(int)

        self.ranking_df = df.sort_values('composite_score', ascending=False)

    def statistical_significance_test(self):
        """Test estadístico de significancia entre estrategias top"""
        if self.ranking_df is None or len(self.ranking_df) < 2:
            return {}

        # Cargar datos de retornos si disponibles
        returns_data = {}
        for strategy_key in self.strategies_data.keys():
            try:
                trades_file = Path(f"results/{strategy_key}/trades.csv")
                if trades_file.exists():
                    trades_df = pd.read_csv(trades_file)
                    if 'ReturnPct' in trades_df.columns:
                        returns_data[strategy_key] = trades_df['ReturnPct'].values
            except Exception:
                continue

        if len(returns_data) < 2:
            return {'error': 'Insuficientes datos de retornos para análisis estadístico'}

        # Test t entre top 2 estrategias
        top_strategies = self.ranking_df.head(2)['strategy'].tolist()
        if len(top_strategies) == 2 and all(s in returns_data for s in top_strategies):
            returns1 = returns_data[top_strategies[0]]
            returns2 = returns_data[top_strategies[1]]

            # t-test
            t_stat, p_value = stats.ttest_ind(returns1, returns2, equal_var=False)

            # Effect size (Cohen's d)
            mean_diff = np.mean(returns1) - np.mean(returns2)
            pooled_std = np.sqrt((np.var(returns1) + np.var(returns2)) / 2)
            cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0

            return {
                'top_strategy_1': STRATEGIES[top_strategies[0]]['name'],
                'top_strategy_2': STRATEGIES[top_strategies[1]]['name'],
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'cohen_d': cohen_d,
                'effect_size': 'large' if abs(cohen_d) > 0.8 else 'medium' if abs(cohen_d) > 0.5 else 'small'
            }

        return {'error': 'No se pudieron comparar las estrategias top'}

    def analyze_strategy_correlations(self):
        """Analizar correlaciones entre estrategias"""
        returns_data = {}

        # Cargar retornos de todas las estrategias
        for strategy_key in self.strategies_data.keys():
            try:
                trades_file = Path(f"results/{strategy_key}/trades.csv")
                if trades_file.exists():
                    trades_df = pd.read_csv(trades_file)
                    if 'ReturnPct' in trades_df.columns and len(trades_df) > 0:
                        returns_data[strategy_key] = trades_df['ReturnPct'].values
            except Exception:
                continue

        if len(returns_data) < 2:
            print("Insuficientes datos para análisis de correlación")
            return

        # Crear DataFrame con retornos alineados
        max_len = max(len(returns) for returns in returns_data.values())
        returns_df = pd.DataFrame()

        for strategy_key, returns in returns_data.items():
            # Pad o truncate para alinear
            if len(returns) < max_len:
                padded_returns = np.pad(returns, (0, max_len - len(returns)), constant_values=0)
            else:
                padded_returns = returns[:max_len]
            returns_df[strategy_key] = padded_returns

        # Calcular matriz de correlación
        self.correlation_matrix = returns_df.corr()

        return self.correlation_matrix

    def create_ensemble_recommendations(self):
        """Crear recomendaciones para ensemble strategy"""
        if self.ranking_df is None:
            return {}

        # Estrategias con buen performance
        good_strategies = self.ranking_df[
            (self.ranking_df['sharpe_oos'] > 1.0) &
            (self.ranking_df['robustness'] == 1) &
            (self.ranking_df['overfit_risk'] == 0)
        ]

        if len(good_strategies) == 0:
            return {'recommendation': 'No suitable strategies for ensemble'}

        # Diversificación por tipo
        # strategy_types = good_strategies['type'].value_counts()

        # Clustering basado en correlación si disponible
        if self.correlation_matrix is not None:
            # Estrategias disponibles en matriz de correlación
            available_strategies = [s for s in good_strategies['strategy']
                                    if s in self.correlation_matrix.columns]

            if len(available_strategies) > 1:
                corr_subset = self.correlation_matrix.loc[available_strategies,
                                                          available_strategies]

                # Clustering jerárquico simple (estrategias con baja correlación)
                low_corr_pairs = []
                for i in range(len(available_strategies)):
                    for j in range(i + 1, len(available_strategies)):
                        corr = corr_subset.iloc[i, j]
                        if abs(corr) < 0.3:  # Baja correlación
                            low_corr_pairs.append(
                                (available_strategies[i], available_strategies[j], corr))

                ensemble_candidates = list(
                    set([pair[0] for pair in low_corr_pairs] + [pair[1] for pair in low_corr_pairs]))
            else:
                ensemble_candidates = available_strategies
        else:
            # Fallback: top 3 estrategias
            ensemble_candidates = good_strategies.head(3)['strategy'].tolist()

        return {
            'ensemble_candidates': ensemble_candidates,
            'diversification_score': len(set(good_strategies['type'])) / len(good_strategies),
            'recommended_weight': 1.0 / len(ensemble_candidates) if ensemble_candidates else 0
        }

    def generate_comparison_report(self):
        """Generar reporte completo de comparación"""
        print("\n" + "=" * 60)
        print("BTC STRATEGY TESTER - REPORTE DE COMPARACIÓN")
        print("=" * 60)

        if self.ranking_df is None:
            print("No hay datos para generar reporte")
            return

        # Ranking de estrategias
        print("\n🏆 RANKING DE ESTRATEGIAS")
        print("-" * 40)

        for idx, row in self.ranking_df.iterrows():
            rank_emoji = "🥇" if row['rank'] == 1 else "🥈" if row['rank'] == 2 else "🥉" if row['rank'] == 3 else "📊"
            status = "✅" if row['recommendation'] == 'Deploy' else "⚠️" if row['recommendation'] == 'Further Testing' else "❌"

            print(f"{rank_emoji} #{int(row['rank'])} {row['name']}")
            print(f"   Sharpe OOS: {row['sharpe_oos']:.2f}")
            print(f"   Win Rate: {row['win_rate']:.1%}")
            print(f"   Max DD: {row['max_drawdown']:.2f}")
            print(
                f"   Robustez: {'✅' if row['robustness'] else '❌'} | A/B: {'✅' if row['ab_significant'] else '❌'} | Overfit: {'🟢' if row['overfit_risk'] == 0 else '🔴'}")
            print(f"   Recomendación: {status} {row['recommendation']}")
            print()

        # Análisis estadístico
        print("\n📈 ANÁLISIS ESTADÍSTICO")
        print("-" * 40)

        stat_results = self.statistical_significance_test()
        if 'error' not in stat_results:
            print(
                f"Top 2 estrategias: {stat_results['top_strategy_1']} vs {stat_results['top_strategy_2']}")
            print(f"t-statistic: {stat_results['t_statistic']:.3f}")
            print(f"p-value: {stat_results['p_value']:.3f}")
            print(
                f"Significancia: {'✅ p < 0.05' if stat_results['significant'] else '❌ p >= 0.05'}")
            print(
                f"Tamaño del efecto: {stat_results['effect_size']} (Cohen's d = {stat_results['cohen_d']:.2f})")
        else:
            print(f"Error en análisis estadístico: {stat_results['error']}")

        # Recomendaciones ensemble
        print("\n🎯 RECOMENDACIONES ENSEMBLE")
        print("-" * 40)

        ensemble_rec = self.create_ensemble_recommendations()
        if 'recommendation' not in ensemble_rec:
            print(f"Estrategias candidatas: {ensemble_rec['ensemble_candidates']}")
            print(f"Score de diversificación: {ensemble_rec['diversification_score']:.1%}")
            print(f"Peso recomendado: {ensemble_rec['recommended_weight']:.2f}")
        else:
            print(ensemble_rec['recommendation'])

        # Estrategia ganadora
        winner = self.ranking_df.iloc[0]
        print("\n🎉 ESTRATEGIA GANADORA")
        print("-" * 40)
        print(f"🥇 {winner['name']}")
        print(f"Score compuesto: {winner['composite_score']:.2f}")
        print(f"Recomendación: {winner['recommendation']}")

        if winner['recommendation'] == 'Deploy':
            print("✅ ¡Lista para implementación en producción!")
        else:
            print("⚠️ Requiere análisis adicional antes de implementación")

    def generate_visualizations(self):
        """Generar visualizaciones de comparación"""
        if self.ranking_df is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Sharpe Ratio comparison
        strategies = self.ranking_df['name']
        sharpes = self.ranking_df['sharpe_oos']
        colors = ['gold' if x == max(sharpes) else 'lightcoral' for x in sharpes]

        axes[0, 0].barh(strategies, sharpes, color=colors)
        axes[0, 0].set_title('Sharpe Ratio OOS por Estrategia', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Sharpe Ratio')
        axes[0, 0].axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Benchmark')
        axes[0, 0].legend()

        # 2. Win Rate comparison
        win_rates = self.ranking_df['win_rate']
        colors = ['gold' if x == max(win_rates) else 'lightblue' for x in win_rates]

        axes[0, 1].barh(strategies, win_rates, color=colors)
        axes[0, 1].set_title('Win Rate OOS por Estrategia', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Win Rate')

        # 3. Robustness radar chart
        robustness_metrics = ['robustness', 'ab_significant', 'sensitivity_stable']
        robustness_data = self.ranking_df[robustness_metrics].values

        angles = np.linspace(0, 2 * np.pi, len(robustness_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Cerrar el círculo

        for i, strategy in enumerate(strategies):
            values = robustness_data[i].tolist()
            values += values[:1]  # Cerrar el círculo
            axes[1, 0].plot(angles, values, 'o-', linewidth=2, label=strategy)
            axes[1, 0].fill(angles, values, alpha=0.25)

        axes[1, 0].set_xticks(angles[:-1])
        axes[1, 0].set_xticklabels(robustness_metrics)
        axes[1, 0].set_title('Robustness Analysis', fontsize=12, fontweight='bold')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 4. Correlation heatmap (si disponible)
        if self.correlation_matrix is not None:
            sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm', center=0,
                        ax=axes[1, 1], square=True)
            axes[1, 1].set_title('Correlation Matrix', fontsize=12, fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'Correlation Matrix\nNot Available',
                            ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Correlation Analysis', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Guardar ranking como CSV
        self.ranking_df.to_csv(RESULTS_DIR / 'strategy_ranking.csv', index=False)

        print(f"\nVisualizaciones guardadas en: {RESULTS_DIR}")


def run_strategy_comparison():
    """
    Ejecutar comparación completa de estrategias
    """
    print("🚀 Iniciando BTC Strategy Tester")
    print("Comparando todas las estrategias implementadas...")

    # Inicializar comparador
    comparator = StrategyComparator()

    # Cargar datos
    comparator.load_strategy_results()

    if not comparator.strategies_data:
        print("❌ No se encontraron resultados de estrategias. Ejecute las estrategias individuales primero.")
        return None

    # Crear ranking
    comparator.create_ranking_dataframe()
    comparator.calculate_composite_score()

    # Análisis adicionales
    comparator.analyze_strategy_correlations()
    stat_results = comparator.statistical_significance_test()
    ensemble_rec = comparator.create_ensemble_recommendations()

    # Generar outputs
    comparator.generate_comparison_report()
    comparator.generate_visualizations()

    # Resultados finales
    results = {
        'ranking': comparator.ranking_df.to_dict('records'),
        'correlation_matrix': comparator.correlation_matrix.to_dict() if comparator.correlation_matrix is not None else None,
        'statistical_test': stat_results,
        'ensemble_recommendations': ensemble_rec,
        'winner': comparator.ranking_df.iloc[0]['name'] if len(
            comparator.ranking_df) > 0 else None}

    # Guardar resultados
    with open(RESULTS_DIR / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📊 Resultados guardados en: {RESULTS_DIR}")
    print("✅ Comparación completada!")

    return results


if __name__ == "__main__":
    # Ejecutar comparación
    results = run_strategy_comparison()

    if results:
        print("\n🎯 Estrategia Ganadora:")
        print(f"🥇 {results['winner']}")
    else:
        print("\n❌ No se pudo completar la comparación")
