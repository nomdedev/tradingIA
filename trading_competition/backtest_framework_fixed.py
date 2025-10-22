#!/usr/bin/env python3
"""
Backtesting Framework para estrategias de trading.
Integra backtesting.py para validar agentes RL y GA en datos históricos.
"""

import os
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from environments.trading_env import create_trading_env
from rich.console import Console
from rich.table import Table
import warnings
warnings.filterwarnings('ignore')

console = Console()

class RLTradingStrategy(Strategy):
    """
    Estrategia de backtesting basada en agente RL entrenado.
    """

    def init(self):
        """Inicializar estrategia."""
        self.model = None
        self.env = None
        self.current_step = 0
        self.position_size = 0.95  # Usar 95% del capital disponible

        # Cargar modelo RL si existe
        model_path = "models/trading_agent.zip"
        if os.path.exists(model_path):
            try:
                self.model = PPO.load(model_path)
                # Crear entorno para predicciones
                # Usar datos históricos para crear el env
                data_path = "data/processed/SPY_with_indicators.csv"
                if os.path.exists(data_path):
                    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
                    self.env = create_trading_env(df)
                console.print("Modelo RL cargado para backtesting")
            except Exception as e:
                console.print("Error cargando modelo RL: {}".format(e))
                self.model = None
        else:
            console.print("Modelo RL no encontrado, usando estrategia buy & hold")

    def next(self):
        """Lógica de trading para cada timestep."""
        if len(self.data) % 200 == 0:  # Debug cada 200 timesteps
            console.print(f"RL next() called at timestep {len(self.data)}")
        if self.model is None:
            # Estrategia fallback: Buy & Hold
            if len(self.data) > 1 and self.position.size == 0:
                self.buy(size=self.position_size)
            return

        try:
            # Obtener estado actual del mercado
            current_data = {
                'Open': self.data.Open[-1],
                'High': self.data.High[-1],
                'Low': self.data.Low[-1],
                'Close': self.data.Close[-1],
                'Volume': self.data.Volume[-1]
            }

            # Crear observación simplificada (esto es una aproximación)
            # En un entorno real, necesitaríamos todos los indicadores
            obs = self._create_observation_from_data(current_data)

            # Predecir acción
            # Hacer predicción
            action, _ = self.model.predict(obs, deterministic=True)
            if len(self.data) % 100 == 0:
                console.print(f"RL prediction: action={action}, position={self.position.size}")

            # Ejecutar acción
            if action == 1 and self.position.size == 0:  # BUY
                self.buy(size=self.position_size)
            elif action == 2 and self.position.size > 0:  # SELL
                self.sell(size=self.position.size)

        except Exception as e:
            if len(self.data) % 100 == 0:
                console.print(f"RL error at timestep {len(self.data)}: {e}")
            # Ignorar errores en predicción, mantener posición actual
            pass

    def _create_observation_from_data(self, data):
        """Crear observación simplificada desde datos OHLCV."""
        # Esta es una aproximación simplificada
        # En producción, necesitaríamos calcular todos los indicadores
        try:

            # Features más realistas basados en datos actuales
            close_price = data['Close']
            prev_close = self.data.Close[-2] if len(self.data) > 1 else close_price

            # Calcular retornos simples
            returns_1d = (close_price - prev_close) / prev_close if prev_close != 0 else 0

            # RSI aproximado (simplificado)
            if len(self.data) > 14:
                gains = np.maximum(self.data.Close.diff(), 0)
                losses = np.maximum(-self.data.Close.diff(), 0)
                avg_gain = gains.rolling(14).mean()
                avg_loss = losses.rolling(14).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50
            else:
                current_rsi = 50

            obs = np.array([
                (close_price - self.data.Close.mean()) / self.data.Close.std() if self.data.Close.std() != 0 else 0,  # close_normalized
                returns_1d,  # returns_1d
                0.0,  # returns_5d (simplificado)
                current_rsi / 100.0,  # rsi_14 normalizado
                0.0,  # macd_normalized (simplificado)
                0.0,  # macd_signal_normalized (simplificado)
                0.02, # atr_relative (constante por ahora)
                0.0,  # bb_position (simplificado)
                self.data.Close.pct_change().std() if len(self.data) > 20 else 0.02,  # volatility_20d
                1.0,  # sma_ratio (simplificado)
                1.0 if self.position.size > 0 else 0.0  # position_state
            ], dtype=np.float32)

            return obs

        except Exception:
            # Retornar observación neutral en caso de error
            return np.zeros(15, dtype=np.float32)

class GATradingStrategy(Strategy):
    """
    Estrategia de backtesting basada en agente GA optimizado.
    """

    def init(self):
        """Inicializar estrategia GA."""
        self.ga_params = None
        self.position_size = 0.95

        # Cargar parámetros GA
        params_path = "models/ga_agent_params.txt"
        if os.path.exists(params_path):
            try:
                self.ga_params = {}
                with open(params_path, 'r') as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.strip().split(': ')
                            self.ga_params[key] = float(value)
                console.print("🧬 Parámetros GA cargados para backtesting")
            except Exception as e:
                console.print(f"❌ Error cargando parámetros GA: {e}")
                self.ga_params = None
        else:
            console.print("⚠️ Parámetros GA no encontrados, usando estrategia simple")

    def next(self):
        """Lógica de trading GA."""
        if len(self.data) % 200 == 0:  # Debug cada 200 timesteps
            console.print(f"GA next() called at timestep {len(self.data)}")
        if self.ga_params is None:
            # Estrategia fallback simple
            if len(self.data) > 20:
                sma_20 = self.data.Close[-20:].mean()
                if self.data.Close[-1] > sma_20 and self.position.size == 0:
                    self.buy(size=self.position_size)
                elif self.data.Close[-1] < sma_20 and self.position.size > 0:
                    self.sell(size=self.position.size)
            return

        try:
            # Aplicar lógica GA optimizada
            rsi_overbought = self.ga_params['RSI_Overbought']
            rsi_oversold = self.ga_params['RSI_Oversold']
            macd_threshold = self.ga_params['MACD_Threshold']
            bb_width_threshold = self.ga_params['BB_Width_Threshold']

            # Calcular indicadores básicos (aproximación)
            if len(self.data) > 14:
                # RSI aproximado (simplificado)
                gains = np.maximum(self.data.Close.diff(), 0)
                losses = np.maximum(-self.data.Close.diff(), 0)
                avg_gain = gains.rolling(14).mean()
                avg_loss = losses.rolling(14).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50

                # Señales de trading
                signal = 0

                # RSI signals
                if current_rsi < rsi_oversold:
                    signal = 1  # Buy
                    if len(self.data) % 100 == 0:
                        console.print(f"GA RSI BUY signal: {current_rsi:.2f} < {rsi_oversold}")
                elif current_rsi > rsi_overbought:
                    signal = -1  # Sell
                    if len(self.data) % 100 == 0:
                        console.print(f"GA RSI SELL signal: {current_rsi:.2f} > {rsi_overbought}")

                # MACD approximation (muy simplificado)
                if len(self.data) > 26:
                    ema_12 = self.data.Close.ewm(span=12).mean()
                    ema_26 = self.data.Close.ewm(span=26).mean()
                    macd_line = ema_12 - ema_26
                    macd_signal = macd_line.ewm(span=9).mean()
                    macd_hist = macd_line - macd_signal

                    if macd_line.iloc[-1] > macd_signal.iloc[-1] and macd_hist.iloc[-1] > macd_threshold:
                        signal = 1
                    elif macd_line.iloc[-1] < macd_signal.iloc[-1] and macd_hist.iloc[-1] < -macd_threshold:
                        signal = -1

                # Bollinger Bands approximation
                if len(self.data) > 20:
                    sma_20 = self.data.Close.rolling(20).mean()
                    std_20 = self.data.Close.rolling(20).std()
                    bb_upper = sma_20 + 2 * std_20
                    bb_lower = sma_20 - 2 * std_20
                    bb_width = (bb_upper - bb_lower) / sma_20

                    if self.data.Close.iloc[-1] <= bb_lower.iloc[-1] and bb_width.iloc[-1] > bb_width_threshold:
                        signal = 1
                    elif self.data.Close.iloc[-1] >= bb_upper.iloc[-1] and bb_width.iloc[-1] > bb_width_threshold:
                        signal = -1

                # Ejecutar señal
                if signal == 1 and self.position.size == 0:
                    self.buy(size=self.position_size)
                    console.print(f"🧬 GA BUY en {self.data.Close[-1]:.2f}")
                elif signal == -1 and self.position.size > 0:
                    self.sell(size=self.position.size)
                    console.print(f"🧬 GA SELL en {self.data.Close[-1]:.2f}")

        except Exception:
            # En caso de error, mantener posición
            pass

def load_historical_data():
    """Cargar datos históricos para backtesting."""
    try:
        # Cargar datos raw (OHLCV)
        raw_dir = "data/raw"
        spy_files = [f for f in os.listdir(raw_dir) if f.startswith('SPY') and f.endswith('.csv')]

        if not spy_files:
            raise FileNotFoundError("Datos raw no encontrados")

        raw_path = os.path.join(raw_dir, spy_files[0])
        df = pd.read_csv(raw_path, skiprows=3, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'],
                        index_col=0, parse_dates=True)

        # Renombrar columnas para backtesting.py
        df = df.rename(columns={
            'Close': 'Close',
            'High': 'High',
            'Low': 'Low',
            'Open': 'Open',
            'Volume': 'Volume'
        })

        # Asegurar que el índice sea datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        console.print("Datos historicos cargados: {} filas".format(len(df)))
        return df

    except Exception as e:
        console.print(f"❌ Error cargando datos históricos: {e}")
        raise

def run_backtests():
    """Ejecutar backtests para ambas estrategias."""
    console.print("Iniciando backtests...")
    # Cargar datos
    data = load_historical_data()

    # Configuración del backtest
    initial_cash = 10000
    commission = 0.001  # 0.1%

    results = {}

    # Backtest RL Strategy
    console.print("\n🤖 Ejecutando backtest - Estrategia RL...")
    try:
        bt_rl = Backtest(data, RLTradingStrategy, cash=initial_cash, commission=commission)
        results['RL'] = bt_rl.run()
        console.print("✅ Backtest RL completado")
    except Exception as e:
        console.print(f"❌ Error en backtest RL: {e}")
        results['RL'] = None

    # Backtest GA Strategy
    console.print("\n🧬 Ejecutando backtest - Estrategia GA...")
    try:
        bt_ga = Backtest(data, GATradingStrategy, cash=initial_cash, commission=commission)
        results['GA'] = bt_ga.run()
        console.print("✅ Backtest GA completado")
    except Exception as e:
        console.print(f"❌ Error en backtest GA: {e}")
        results['GA'] = None

    return results, data

def display_results(results):
    """Mostrar resultados de los backtests."""
    console.print("\n" + "="*60)
    console.print("📊 RESULTADOS DE BACKTESTING")
    console.print("="*60)

    table = Table(title="Comparación de Estrategias")
    table.add_column("Métrica", style="cyan", no_wrap=True)
    table.add_column("RL Strategy", style="green")
    table.add_column("GA Strategy", style="yellow")
    table.add_column("Buy & Hold", style="magenta")

    metrics = [
        ('Retorno Total', 'Return [%]'),
        ('Retorno Anual', 'Return (Ann.) [%]'),
        ('Volatilidad Anual', 'Volatility (Ann.) [%]'),
        ('Sharpe Ratio', 'Sharpe Ratio'),
        ('Max Drawdown', 'Max. Drawdown [%]'),
        ('Win Rate', 'Win Rate [%]'),
        ('Mejor Trade', 'Best Trade [%]'),
        ('Peor Trade', 'Worst Trade [%]'),
        ('Trades Totales', '# Trades'),
        ('Exposure Time', 'Exposure Time [%]')
    ]

    for metric_name, result_key in metrics:
        rl_value = "N/A"
        ga_value = "N/A"
        bh_value = "N/A"

        if results.get('RL') is not None and result_key in results['RL']:
            rl_value = f"{results['RL'][result_key]:.2f}"

        if results.get('GA') is not None and result_key in results['GA']:
            ga_value = f"{results['GA'][result_key]:.2f}"

        # Buy & Hold aproximado (retorno total del período)
        if results.get('RL') is not None and 'Return [%]' in results['RL']:
            # Calcular buy & hold simple
            try:
                data = load_historical_data()
                initial_price = data['Close'].iloc[0]
                final_price = data['Close'].iloc[-1]
                bh_return = ((final_price - initial_price) / initial_price) * 100
                bh_value = f"{bh_return:.2f}"
            except Exception:
                bh_value = "N/A"

        table.add_row(metric_name, rl_value, ga_value, bh_value)

    console.print(table)

def plot_backtest_results(results, data):
    """Crear visualizaciones de los resultados."""
    try:
        # Crear directorio para plots
        os.makedirs("results/backtests", exist_ok=True)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Equity curves
        ax1.set_title('Curvas de Equity')
        if results.get('RL') is not None:
            ax1.plot(results['RL']['Equity'], label='RL Strategy', alpha=0.8)
        if results.get('GA') is not None:
            ax1.plot(results['GA']['Equity'], label='GA Strategy', alpha=0.8)

        # Buy & Hold
        initial_price = data['Close'].iloc[0]
        bh_equity = (data['Close'] / initial_price) * 10000
        ax1.plot(bh_equity, label='Buy & Hold', alpha=0.6, linestyle='--')
        ax1.set_ylabel('Equity ($)')
        ax1.legend()
        ax1.grid(True)

        # Drawdown
        ax2.set_title('Drawdown')
        if results.get('RL') is not None:
            ax2.fill_between(results['RL']['Equity'].index,
                           results['RL']['Drawdown'], 0, alpha=0.3, label='RL')
        if results.get('GA') is not None:
            ax2.fill_between(results['GA']['Equity'].index,
                           results['GA']['Drawdown'], 0, alpha=0.3, label='GA')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True)

        # Returns distribution
        ax3.set_title('Distribución de Retornos Diarios')

        if results.get('RL') is not None and len(results['RL']['Returns']) > 0:
            rl_returns = results['RL']['Returns'].dropna()
            ax3.hist(rl_returns, bins=50, alpha=0.6, label='RL', density=True)

        if results.get('GA') is not None and len(results['GA']['Returns']) > 0:
            ga_returns = results['GA']['Returns'].dropna()
            ax3.hist(ga_returns, bins=50, alpha=0.6, label='GA', density=True)

        ax3.set_xlabel('Retorno Diario (%)')
        ax3.set_ylabel('Densidad')
        ax3.legend()
        ax3.grid(True)

        # Rolling Sharpe Ratio (30 días)
        ax4.set_title('Sharpe Ratio Rolling (30 días)')
        if results.get('RL') is not None and len(results['RL']['Returns']) > 30:
            rl_sharpe = (results['RL']['Returns'].rolling(30).mean() /
                        results['RL']['Returns'].rolling(30).std()) * np.sqrt(252)
            ax4.plot(rl_sharpe, label='RL Strategy', alpha=0.8)

        if results.get('GA') is not None and len(results['GA']['Returns']) > 30:
            ga_sharpe = (results['GA']['Returns'].rolling(30).mean() /
                        results['GA']['Returns'].rolling(30).std()) * np.sqrt(252)
            ax4.plot(ga_sharpe, label='GA Strategy', alpha=0.8)

        ax4.set_ylabel('Sharpe Ratio')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig("results/backtests/backtest_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()

        console.print("📊 Gráficos guardados en: results/backtests/backtest_comparison.png")

    except Exception as e:
        console.print(f"❌ Error creando visualizaciones: {e}")

def save_detailed_report(results):
    """Guardar reporte detallado en archivo."""
    try:
        os.makedirs("results/backtests", exist_ok=True)

        with open("results/backtests/backtest_report.txt", 'w', encoding='utf-8') as f:
            f.write("REPORTE DETALLADO DE BACKTESTING\n")
            f.write("="*50 + "\n\n")

            for strategy_name, result in results.items():
                if result is not None:
                    f.write(f"ESTRATEGIA: {strategy_name}\n")
                    f.write("-"*30 + "\n")

                    # Métricas principales
                    f.write(f"Retorno Total: {result['Return [%]']:.2f}%\n")
                    f.write(f"Retorno Anualizado: {result['Return (Ann.) [%]']:.2f}%\n")
                    f.write(f"Volatilidad Anual: {result['Volatility (Ann.) [%]']:.2f}%\n")
                    f.write(f"Sharpe Ratio: {result['Sharpe Ratio']:.2f}\n")
                    f.write(f"Max Drawdown: {result['Max. Drawdown [%]']:.2f}%\n")
                    f.write(f"Win Rate: {result['Win Rate [%]']:.2f}%\n")
                    f.write(f"Número de Trades: {result['# Trades']}\n")
                    f.write(f"Exposure Time: {result['Exposure Time [%]']:.2f}%\n\n")

                    # Estadísticas de trades
                    f.write("ESTADÍSTICAS DE TRADES:\n")
                    f.write(f"Mejor Trade: {result['Best Trade [%]']:.2f}%\n")
                    f.write(f"Peor Trade: {result['Worst Trade [%]']:.2f}%\n")
                    f.write(f"Trade Promedio: {result['Avg. Trade [%]']:.2f}%\n")
                    f.write(f"Profit Factor: {result.get('Profit Factor', 'N/A')}\n\n")

                else:
                    f.write(f"ESTRATEGIA: {strategy_name} - ERROR EN BACKTEST\n\n")

        console.print("📄 Reporte detallado guardado en: results/backtests/backtest_report.txt")

    except Exception as e:
        console.print(f"❌ Error guardando reporte: {e}")

def main():
    """Función principal de backtesting."""
    console.print("Iniciando Framework de Backtesting")

    try:
        # Ejecutar backtests
        results, data = run_backtests()

        # Mostrar resultados
        display_results(results)

        # Crear visualizaciones
        plot_backtest_results(results, data)

        # Guardar reporte detallado
        save_detailed_report(results)

        console.print("\n✅ Backtesting completado exitosamente!")

    except Exception as e:
        console.print(f"❌ Error en backtesting: {e}")
        raise

if __name__ == "__main__":
    main()
