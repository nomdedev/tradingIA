import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Añadir el directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.indicators import add_all_indicators
from utils.advanced_indicators import add_all_advanced_indicators
from agents.train_ga_agent import GAStrategy

# Importar agente RL si está disponible
try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("⚠️  Stable Baselines3 no disponible. RL comparison será omitido.")

class StrategyComparator:
    """Compara diferentes estrategias de trading"""

    def __init__(self):
        self.results = {}
        self.data = None
        self.models = {}

    def load_data(self, filepath: str):
        """Carga y prepara datos"""
        print(f"📊 Cargando datos desde {filepath}")
        self.data = pd.read_csv(filepath)

        # Añadir indicadores básicos
        self.data = add_all_indicators(self.data)

        # Añadir indicadores avanzados
        self.data = add_all_advanced_indicators(self.data)

        print(f"✓ Datos preparados: {len(self.data)} filas, {len(self.data.columns)} columnas")
        return self.data

    def load_models(self):
        """Carga todos los modelos disponibles"""
        models_dir = "models"

        # Cargar modelo GA avanzado
        try:
            ga_advanced_path = os.path.join(models_dir, "ga_advanced_strategy.pkl")
            if os.path.exists(ga_advanced_path):
                model_data = joblib.load(ga_advanced_path)
                self.models['GA_Advanced'] = model_data['strategy']
                print("✓ Modelo GA Avanzado cargado")
            else:
                print("⚠️  Modelo GA Avanzado no encontrado")
        except Exception as e:
            print(f"❌ Error cargando GA Avanzado: {e}")

        # Cargar modelo GA básico
        try:
            ga_basic_path = os.path.join(models_dir, "ga_best_individual.pkl")
            if os.path.exists(ga_basic_path):
                model_data = joblib.load(ga_basic_path)
                # Extraer genes del objeto Individual de DEAP
                if hasattr(model_data, '__iter__'):
                    genes = list(model_data)
                    print(f"✓ Modelo GA Básico: {len(genes)} genes encontrados")
                    if len(genes) >= 10:
                        basic_genes = genes[:10]
                        self.models['GA_Basic'] = self._create_basic_strategy(basic_genes)
                        print("✓ Modelo GA Básico cargado")
                    else:
                        print(f"⚠️  Modelo GA Básico tiene solo {len(genes)} genes, necesita al menos 10")
                else:
                    print("⚠️  Formato de modelo GA Básico incompatible")
            else:
                print("⚠️  Modelo GA Básico no encontrado")
        except Exception as e:
            print(f"❌ Error cargando GA Básico: {e}")

        # Cargar modelo RL
        if RL_AVAILABLE:
            try:
                rl_path = os.path.join(models_dir, "ppo_trading_agent.zip")
                if os.path.exists(rl_path):
                    self.models['RL_Agent'] = PPO.load(rl_path)
                    print("✓ Modelo RL cargado")
                else:
                    print("⚠️  Modelo RL no encontrado")
            except Exception as e:
                print(f"❌ Error cargando RL: {e}")

    def _create_basic_strategy(self, genes: List[float]):
        """Crea una estrategia básica con indicadores estándar"""
        class BasicStrategy:
            def __init__(self, genes):
                self.rsi_buy = max(20, min(40, genes[0]))
                self.rsi_sell = max(60, min(80, genes[1]))
                self.macd_threshold = genes[2]
                self.stop_loss = max(1, min(5, genes[3]))
                self.take_profit = max(2, min(8, genes[4]))

            def decide(self, row, position):
                if position > 0:
                    # Check stop loss / take profit
                    entry_price = row.get('entry_price', row['Close'])
                    pnl_pct = (row['Close'] - entry_price) / entry_price * 100
                    if pnl_pct <= -self.stop_loss or pnl_pct >= self.take_profit:
                        return "SELL", f"Exit: P&L {pnl_pct:.1f}%"

                # Basic signals
                signals = 0
                if row.get('RSI', 50) < self.rsi_buy:
                    signals += 1
                if row.get('MACD', 0) > self.macd_threshold:
                    signals += 1
                if row.get('Volume_ratio', 1) > 1.2:
                    signals += 1

                if position == 0 and signals >= 2:
                    return "BUY", f"Entry: {signals} signals"
                elif position > 0 and signals < 1:
                    return "SELL", "Exit: weak signals"

                return "HOLD", "Hold"

        return BasicStrategy(genes)

    def backtest_strategy(self, strategy, name: str, initial_capital: float = 10000) -> Dict:
        """Ejecuta backtest de una estrategia"""
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        portfolio_values = [capital]
        peak_capital = capital
        max_drawdown = 0

        for idx, row in self.data.iterrows():
            decision, reason = strategy.decide(row, position)

            if decision == "BUY" and position == 0:
                shares = int(capital * 0.95 / row['Close'])
                if shares > 0:
                    position = shares
                    entry_price = row['Close']
                    capital -= shares * row['Close']

            elif decision == "SELL" and position > 0:
                capital += position * row['Close']
                pnl = (row['Close'] - entry_price) / entry_price
                trades.append(pnl)
                position = 0
                entry_price = 0

            # Track portfolio value
            portfolio_value = capital + (position * row['Close'] if position > 0 else 0)
            portfolio_values.append(portfolio_value)

            # Calculate drawdown
            peak_capital = max(peak_capital, portfolio_value)
            drawdown = (peak_capital - portfolio_value) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)

        # Calculate metrics
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital * 100
        num_trades = len(trades)

        if num_trades > 0:
            win_rate = sum(1 for t in trades if t > 0) / num_trades
            avg_win = np.mean([t for t in trades if t > 0]) if any(t > 0 for t in trades) else 0
            avg_loss = abs(np.mean([t for t in trades if t < 0])) if any(t < 0 for t in trades) else 0
            profit_factor = (win_rate * avg_win) / ((1 - win_rate) * avg_loss) if avg_loss > 0 else float('inf')

            # Sharpe ratio (simplified)
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            sharpe_ratio = 0

        return {
            'name': name,
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_values': portfolio_values,
            'trades': trades
        }

    def backtest_rl_agent(self, name: str, initial_capital: float = 10000) -> Dict:
        """Backtest del agente RL"""
        if 'RL_Agent' not in self.models:
            return None

        agent = self.models['RL_Agent']

        # Crear entorno simplificado para backtest
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        portfolio_values = [capital]
        peak_capital = capital
        max_drawdown = 0

        for idx, row in self.data.iterrows():
            # Preparar estado para el agente RL
            state = self._prepare_rl_state(row, position, capital)

            # Obtener acción del agente
            action, _ = agent.predict(state, deterministic=True)

            # 0 = HOLD, 1 = BUY, 2 = SELL
            if action == 1 and position == 0:  # BUY
                shares = int(capital * 0.95 / row['Close'])
                if shares > 0:
                    position = shares
                    entry_price = row['Close']
                    capital -= shares * row['Close']

            elif action == 2 and position > 0:  # SELL
                capital += position * row['Close']
                pnl = (row['Close'] - entry_price) / entry_price
                trades.append(pnl)
                position = 0
                entry_price = 0

            # Track portfolio value
            portfolio_value = capital + (position * row['Close'] if position > 0 else 0)
            portfolio_values.append(portfolio_value)

            # Calculate drawdown
            peak_capital = max(peak_capital, portfolio_value)
            drawdown = (peak_capital - portfolio_value) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)

        # Calculate metrics
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital * 100
        num_trades = len(trades)

        if num_trades > 0:
            win_rate = sum(1 for t in trades if t > 0) / num_trades
            avg_win = np.mean([t for t in trades if t > 0]) if any(t > 0 for t in trades) else 0
            avg_loss = abs(np.mean([t for t in trades if t < 0])) if any(t < 0 for t in trades) else 0
            profit_factor = (win_rate * avg_win) / ((1 - win_rate) * avg_loss) if avg_loss > 0 else float('inf')
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            sharpe_ratio = 0

        return {
            'name': name,
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_values': portfolio_values,
            'trades': trades
        }

    def _prepare_rl_state(self, row: pd.Series, position: int, capital: float) -> np.ndarray:
        """Prepara estado para agente RL"""
        # Estado simplificado con indicadores clave
        state = [
            row.get('Close', 0) / 1000,  # Normalizado
            row.get('RSI', 50) / 100,
            row.get('MACD', 0) / 10,
            row.get('Volume_ratio', 1),
            position / 100,  # Normalizado
            capital / 10000  # Normalizado
        ]
        return np.array(state, dtype=np.float32)

    def backtest_buy_and_hold(self, name: str = "Buy_and_Hold", initial_capital: float = 10000) -> Dict:
        """Backtest de estrategia Buy & Hold"""
        initial_price = self.data.iloc[0]['Close']
        final_price = self.data.iloc[-1]['Close']

        shares = int(initial_capital / initial_price)
        final_capital = shares * final_price

        total_return = (final_capital - initial_capital) / initial_capital * 100

        # Portfolio values (simplified)
        portfolio_values = []
        for _, row in self.data.iterrows():
            portfolio_values.append(shares * row['Close'])

        return {
            'name': name,
            'total_return': total_return,
            'num_trades': 1,  # Solo una transacción
            'win_rate': 1.0 if total_return > 0 else 0.0,
            'avg_win': total_return if total_return > 0 else 0,
            'avg_loss': abs(total_return) if total_return < 0 else 0,
            'profit_factor': float('inf') if total_return > 0 else 0,
            'sharpe_ratio': 0,  # No calculable para buy & hold simple
            'max_drawdown': 0,  # Simplificado
            'portfolio_values': portfolio_values,
            'trades': [total_return]
        }

    def run_comparison(self):
        """Ejecuta comparación completa de todas las estrategias"""
        print("\n🏁 INICIANDO COMPARACIÓN DE ESTRATEGIAS")
        print("=" * 60)

        # Cargar modelos
        self.load_models()

        # Ejecutar backtests
        results = []

        # GA Avanzado
        if 'GA_Advanced' in self.models:
            print("🔬 Backtesting GA Avanzado...")
            result = self.backtest_strategy(self.models['GA_Advanced'], 'GA_Advanced')
            results.append(result)

        # GA Básico
        if 'GA_Basic' in self.models:
            print("🔬 Backtesting GA Básico...")
            result = self.backtest_strategy(self.models['GA_Basic'], 'GA_Basic')
            results.append(result)

        # RL Agent (deshabilitado temporalmente por compatibilidad)
        # if RL_AVAILABLE and 'RL_Agent' in self.models:
        #     print("🤖 Backtesting RL Agent...")
        #     result = self.backtest_rl_agent('RL_Agent')
        #     if result:
        #         results.append(result)

        # Buy & Hold
        print("📈 Backtesting Buy & Hold...")
        result = self.backtest_buy_and_hold()
        results.append(result)

        # Mostrar resultados
        self.display_results(results)

        # Generar gráficos
        self.plot_comparison(results)

        return results

    def display_results(self, results: List[Dict]):
        """Muestra tabla comparativa de resultados"""
        print("\n📊 RESULTADOS DE COMPARACIÓN")
        print("=" * 80)

        # Crear tabla
        headers = ['Estrategia', 'Retorno Total', 'Trades', 'Win Rate', 'Profit Factor', 'Max Drawdown', 'Sharpe']
        print(f"{headers[0]:<15} {headers[1]:<12} {headers[2]:<6} {headers[3]:<8} {headers[4]:<12} {headers[5]:<12} {headers[6]:<8}")
        print("-" * 80)

        for result in sorted(results, key=lambda x: x['total_return'], reverse=True):
            print(f"{result['name']:<15} "
                  f"{result['total_return']:>8.1f}% "
                  f"{result['num_trades']:>6} "
                  f"{result['win_rate']:>8.1%} "
                  f"{result['profit_factor']:>12.2f} "
                  f"{result['max_drawdown']:>12.1%} "
                  f"{result['sharpe_ratio']:>8.2f}")

    def plot_comparison(self, results: List[Dict]):
        """Genera gráficos comparativos"""
        try:
            # Configurar estilo
            plt.style.use('default')
            sns.set_palette("husl")

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Comparación de Estrategias de Trading', fontsize=16, fontweight='bold')

            # 1. Evolución del portfolio
            ax1.set_title('Evolución del Portfolio')
            for result in results:
                values = result['portfolio_values']
                if len(values) > 1:
                    ax1.plot(values, label=result['name'], linewidth=2)
            ax1.set_xlabel('Días')
            ax1.set_ylabel('Valor ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Retornos totales
            ax2.set_title('Retorno Total por Estrategia')
            names = [r['name'] for r in results]
            returns = [r['total_return'] for r in results]
            bars = ax2.bar(names, returns)
            ax2.set_ylabel('Retorno (%)')
            ax2.tick_params(axis='x', rotation=45)

            # Colorear barras positivas/negativas
            for bar, ret in zip(bars, returns):
                bar.set_color('green' if ret > 0 else 'red')

            # 3. Win Rate vs Profit Factor
            ax3.set_title('Win Rate vs Profit Factor')
            win_rates = [r['win_rate'] for r in results]
            profit_factors = [min(r['profit_factor'], 5) for r in results]  # Cap for visualization

            scatter = ax3.scatter(win_rates, profit_factors, s=100, alpha=0.7)
            for i, name in enumerate(names):
                ax3.annotate(name, (win_rates[i], profit_factors[i]),
                           xytext=(5, 5), textcoords='offset points')

            ax3.set_xlabel('Win Rate')
            ax3.set_ylabel('Profit Factor (capped at 5)')
            ax3.grid(True, alpha=0.3)

            # 4. Drawdown
            ax4.set_title('Maximum Drawdown')
            drawdowns = [r['max_drawdown'] for r in results]
            bars2 = ax4.bar(names, drawdowns)
            ax4.set_ylabel('Max Drawdown (%)')
            ax4.tick_params(axis='x', rotation=45)

            plt.tight_layout()

            # Guardar gráfico
            os.makedirs('results', exist_ok=True)
            plt.savefig('results/strategy_comparison.png', dpi=300, bbox_inches='tight')
            print("📊 Gráfico guardado en: results/strategy_comparison.png")

            plt.show()

        except Exception as e:
            print(f"⚠️  Error generando gráficos: {e}")


def main():
    """Función principal"""
    print("🧪 COMPARADOR DE ESTRATEGIAS DE TRADING")
    print("Compara: GA Avanzado vs GA Básico vs RL vs Buy & Hold")

    # Inicializar comparador
    comparator = StrategyComparator()

    # Cargar datos
    data_path = "data/processed/SPY_with_indicators.csv"
    comparator.load_data(data_path)

    # Ejecutar comparación
    results = comparator.run_comparison()

    # Análisis final
    print("\n🎯 ANÁLISIS FINAL")
    print("=" * 40)

    if results:
        best_strategy = max(results, key=lambda x: x['total_return'])
        print(f"🥇 Mejor estrategia: {best_strategy['name']}")
        print(f"   Retorno: {best_strategy['total_return']:.1f}%")
        print(f"   Win Rate: {best_strategy['win_rate']:.1%}")
        print(f"   Profit Factor: {best_strategy['profit_factor']:.2f}")

        # Comparación con Buy & Hold
        bh_result = next((r for r in results if r['name'] == 'Buy_and_Hold'), None)
        if bh_result:
            outperformance = best_strategy['total_return'] - bh_result['total_return']
            print(f"   Outperformance vs Buy&Hold: {outperformance:+.1f}%")

    print("\n✅ Comparación completada!")
    print("Resultados guardados en: results/strategy_comparison.png")


if __name__ == "__main__":
    main()