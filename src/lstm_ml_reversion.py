"""
LSTM ML Mean Reversion Strategy

Estrategia de mean reversion basada en Machine Learning usando LSTM networks.
Predice retornos futuros y opera cuando el modelo indica reversión.

Características:
- LSTM network para predicción de retornos
- Feature engineering avanzado con indicadores técnicos
- Walk-forward testing con re-training del modelo
- A/B testing vs traditional mean reversion
- Robustness analysis y anti-overfit measures
"""

import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import talib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Configuración
RESULTS_DIR = Path("results/lstm_ml_reversion")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuración de TensorFlow
tf.config.set_visible_devices([], 'GPU')  # Disable GPU for compatibility
tf.random.set_seed(42)
np.random.seed(42)


class LSTMModel:
    """
    LSTM Model para predicción de retornos
    """

    def __init__(self, lookback=20, n_features=10, lstm_units=50, dropout_rate=0.2):
        self.lookback = lookback
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def build_model(self):
        """Construir arquitectura LSTM"""
        model = Sequential([
            LSTM(self.lstm_units, input_shape=(self.lookback, self.n_features),
                 return_sequences=True),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units // 2),
            Dropout(self.dropout_rate),
            Dense(1)  # Predicción de retorno
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='mse',
                      metrics=['mae'])
        self.model = model
        return model

    def prepare_features(self, df):
        """Preparar features para el modelo"""
        features = pd.DataFrame(index=df.index)

        # Price-based features
        features['returns'] = df['Close'].pct_change()
        features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Technical indicators
        features['rsi'] = talib.RSI(df['Close'], timeperiod=14)
        features['macd'], features['macdsignal'], features['macdhist'] = talib.MACD(df['Close'])
        features['bb_upper'], features['bb_middle'], features['bb_lower'] = talib.BBANDS(
            df['Close'])
        features['stoch_k'], features['stoch_d'] = talib.STOCH(df['High'], df['Low'], df['Close'])
        features['willr'] = talib.WILLR(df['High'], df['Low'], df['Close'])
        features['cci'] = talib.CCI(df['High'], df['Low'], df['Close'])
        features['mfi'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'])

        # Volatility features
        features['atr'] = talib.ATR(df['High'], df['Low'], df['Close'])
        features['natr'] = talib.NATR(df['High'], df['Low'], df['Close'])

        # Volume features
        features['volume_sma'] = talib.SMA(df['Volume'], timeperiod=20)
        features['volume_ratio'] = df['Volume'] / features['volume_sma']

        # Momentum features
        features['roc'] = talib.ROC(df['Close'], timeperiod=10)
        features['mom'] = talib.MOM(df['Close'], timeperiod=10)

        # Fill NaN values
        features = features.fillna(method='bfill').fillna(0)

        return features

    def prepare_sequences(self, features, target, lookback):
        """Preparar secuencias para LSTM"""
        X, y = [], []

        for i in range(lookback, len(features)):
            X.append(features.iloc[i - lookback:i].values)
            y.append(target.iloc[i])

        return np.array(X), np.array(y)

    def train(self, df_train, target_horizon=5):
        """Entrenar el modelo LSTM"""
        # Preparar features
        features = self.prepare_features(df_train)

        # Target: retorno futuro
        target = df_train['Close'].pct_change(target_horizon).shift(-target_horizon).fillna(0)

        # Alinear datos
        common_index = features.index.intersection(target.index)
        features = features.loc[common_index]
        target = target.loc[common_index]

        # Escalar features
        scaled_features = self.scaler.fit_transform(features)

        # Preparar secuencias
        X, y = self.prepare_sequences(pd.DataFrame(scaled_features, index=features.index),
                                      target, self.lookback)

        if len(X) == 0:
            print("No hay suficientes datos para entrenar")
            return

        # Build model
        self.build_model()

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train
        history = self.model.fit(
            X, y,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

        self.is_trained = True
        return history

    def predict(self, df_current):
        """Hacer predicciones"""
        if not self.is_trained:
            return np.zeros(len(df_current))

        # Preparar features
        features = self.prepare_features(df_current)

        # Escalar
        scaled_features = self.scaler.transform(features)

        # Preparar secuencias
        X, _ = self.prepare_sequences(pd.DataFrame(scaled_features, index=features.index),
                                      pd.Series(index=features.index), self.lookback)

        if len(X) == 0:
            return np.zeros(len(df_current))

        # Predecir
        predictions = self.model.predict(X, verbose=0).flatten()

        # Pad predictions to match original length
        padded_predictions = np.zeros(len(df_current))
        padded_predictions[self.lookback:] = predictions

        return padded_predictions


class LSTMMReversionStrategy(Strategy):
    """
    LSTM-based Mean Reversion Strategy
    """

    # Parámetros optimizables
    prediction_threshold = 0.02  # Threshold para señal de reversión
    confidence_threshold = 0.7   # Threshold de confianza del modelo
    max_holding_period = 10      # Máximo período de holding
    stop_loss_mult = 1.5         # Multiplicador ATR para SL
    take_profit_mult = 2.0       # Multiplicador para TP

    def init(self):
        """Inicializar modelo LSTM y indicadores"""
        # Modelo LSTM
        self.lstm_model = LSTMModel()

        # Predicciones del modelo
        self.predictions = self.I(self._get_predictions, self.data)

        # Indicadores técnicos auxiliares
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=20
        )

        # Contador de períodos en posición
        self.holding_counter = 0

        # Modelo entrenado flag
        self.model_trained = False

    def _get_predictions(self, data):
        """Obtener predicciones del modelo LSTM"""
        df = pd.DataFrame({
            'Open': data.Open,
            'High': data.High,
            'Low': data.Low,
            'Close': data.Close,
            'Volume': data.Volume
        })

        if not self.model_trained and len(df) > 100:
            # Entrenar modelo con datos históricos disponibles
            try:
                self.lstm_model.train(df)
                self.model_trained = True
                print("Modelo LSTM entrenado")
            except Exception as e:
                print(f"Error entrenando modelo: {e}")
                return np.zeros(len(df))

        # Hacer predicciones
        predictions = self.lstm_model.predict(df)
        return predictions

    def next(self):
        """Lógica de trading basada en LSTM"""
        if len(self.predictions) < 2 or not self.model_trained:
            return

        current_prediction = self.predictions[-1]
        prev_prediction = self.predictions[-2]

        # Condiciones de reversión
        bullish_reversion = (current_prediction > self.prediction_threshold and
                             prev_prediction < -self.prediction_threshold)
        bearish_reversion = (current_prediction < -self.prediction_threshold and
                             prev_prediction > self.prediction_threshold)

        # Filtros adicionales
        rsi_oversold = self.rsi[-1] < 30
        rsi_overbought = self.rsi[-1] > 70
        price_near_lower_bb = self.data.Close[-1] < self.bb_lower[-1] * 1.01
        price_near_upper_bb = self.data.Close[-1] > self.bb_upper[-1] * 0.99

        # Confidence check (basado en magnitud de predicción)
        confidence = abs(current_prediction) > self.confidence_threshold

        # Holding period limit
        self.holding_counter = self.holding_counter + 1 if self.position else 0
        force_exit = self.holding_counter >= self.max_holding_period

        # Exit positions
        if force_exit and self.position:
            self.position.close()
            self.holding_counter = 0
            return

        # Entry long (reversión bajista esperada)
        if (bullish_reversion and confidence and rsi_oversold and price_near_lower_bb and
                not self.position.is_long and not self.position.is_short):

            # Calcular SL/TP
            entry_price = self.data.Close[-1]
            atr_value = self.atr[-1]
            sl_price = entry_price - (self.stop_loss_mult * atr_value)
            tp_price = entry_price + (self.take_profit_mult * (entry_price - sl_price))

            self.buy(sl=sl_price, tp=tp_price)
            self.holding_counter = 0

        # Entry short (reversión alcista esperada)
        elif (bearish_reversion and confidence and rsi_overbought and price_near_upper_bb and
              not self.position.is_long and not self.position.is_short):

            # Calcular SL/TP
            entry_price = self.data.Close[-1]
            atr_value = self.atr[-1]
            sl_price = entry_price + (self.stop_loss_mult * atr_value)
            tp_price = entry_price - (self.take_profit_mult * (sl_price - entry_price))

            self.sell(sl=sl_price, tp=tp_price)
            self.holding_counter = 0


def calculate_metrics(bt):
    """Calcular métricas avanzadas incluyendo ML metrics"""
    trades = bt.trades

    if len(trades) == 0:
        return {
            'sharpe_ratio': 0,
            'calmar_ratio': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'total_return': 0,
            'total_trades': 0,
            'avg_trade': 0,
            'sortino_ratio': 0,
            'var_95': 0,
            'ulcer_index': 0,
            'model_accuracy': 0,
            'prediction_confidence': 0
        }

    # Returns diarios
    returns = trades['ReturnPct'].values
    equity_curve = (1 + returns).cumprod()

    # Sharpe Ratio (anualizado)
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    if len(daily_returns) > 0 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        sharpe = 0

    # Calmar Ratio
    max_dd = (equity_curve / equity_curve.expanding().max() - 1).min()
    if max_dd < 0:
        calmar = -np.mean(daily_returns) / abs(max_dd) * \
            np.sqrt(252) if len(daily_returns) > 0 else 0
    else:
        calmar = 0

    # Win Rate
    win_rate = (returns > 0).mean()

    # Profit Factor
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]
    profit_factor = winning_trades.sum() / abs(losing_trades.sum()) if len(losing_trades) > 0 else float('inf')

    # Sortino Ratio
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 0:
        sortino = np.mean(daily_returns) / np.std(downside_returns) * np.sqrt(252)
    else:
        sortino = float('inf')

    # VaR 95%
    var_95 = np.percentile(returns, 5)

    # Ulcer Index
    drawdowns = 1 - equity_curve / equity_curve.expanding().max()
    ulcer = np.sqrt((drawdowns ** 2).mean())

    # ML-specific metrics (simplified)
    strategy_instance = bt._strategy
    if hasattr(strategy_instance, 'lstm_model') and strategy_instance.lstm_model.is_trained:
        # Model accuracy (directional)
        predictions = strategy_instance.predictions
        actual_returns = pd.Series(strategy_instance.data.Close).pct_change().fillna(0).values

        if len(predictions) == len(actual_returns):
            correct_direction = np.sign(predictions) == np.sign(actual_returns)
            model_accuracy = correct_direction.mean()
            prediction_confidence = np.mean(np.abs(predictions))
        else:
            model_accuracy = 0
            prediction_confidence = 0
    else:
        model_accuracy = 0
        prediction_confidence = 0

    return {
        'sharpe_ratio': sharpe,
        'calmar_ratio': calmar,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
        'total_return': equity_curve[-1] - 1,
        'total_trades': len(trades),
        'avg_trade': np.mean(returns),
        'sortino_ratio': sortino,
        'var_95': var_95,
        'ulcer_index': ulcer,
        'model_accuracy': model_accuracy,
        'prediction_confidence': prediction_confidence
    }


def walk_forward_test(df, n_periods=6, train_months=4, test_months=1):
    """
    Walk-forward testing con re-training del modelo LSTM

    Args:
        df: DataFrame con datos OHLCV
        n_periods: Número de periodos walk-forward
        train_months: Meses para training
        test_months: Meses para testing OOS
    """
    results = []
    period_length = pd.Timedelta(days=30 * (train_months + test_months))

    for i in range(n_periods):
        # Definir periodo
        end_date = df.index[-1] - pd.Timedelta(days=i * 30 * test_months)
        start_date = end_date - period_length

        if start_date < df.index[0]:
            break

        period_data = df.loc[start_date:end_date]

        # Split train/test
        split_date = end_date - pd.Timedelta(days=30 * test_months)
        train_data = period_data.loc[:split_date]
        test_data = period_data.loc[split_date:]

        print(
            f"Period {i+1}: Train {train_data.index[0]} to {train_data.index[-1]}, Test {test_data.index[0]} to {test_data.index[-1]}")

        # Re-entrenar modelo en train data
        lstm_model = LSTMModel()
        try:
            lstm_model.train(train_data)
            print(f"Modelo re-entrenado para periodo {i+1}")
        except Exception as e:
            print(f"Error entrenando modelo en periodo {i+1}: {e}")
            continue

        # Test OOS con modelo entrenado
        bt = Backtest(test_data, LSTMMReversionStrategy, cash=10000, commission=0.0005)

        # Inyectar modelo entrenado en la estrategia
        class LSTMMReversionStrategyTrained(LSTMMReversionStrategy):
            def init(self):
                super().init()
                self.lstm_model = lstm_model
                self.model_trained = True

        result = bt.run()

        # Calcular métricas
        metrics = calculate_metrics(bt)

        results.append({
            'period': i + 1,
            'train_start': train_data.index[0],
            'train_end': train_data.index[-1],
            'test_start': test_data.index[0],
            'test_end': test_data.index[-1],
            'metrics': metrics,
            'backtest_result': result,
            'model_accuracy': metrics['model_accuracy']
        })

    return results


def ab_test_vs_traditional_mr(df, traditional_mr_signals):
    """A/B test vs Traditional Mean Reversion strategy"""
    # Run LSTM strategy
    bt_lstm = Backtest(df, LSTMMReversionStrategy, cash=10000, commission=0.0005)
    result_lstm = bt_lstm.run()

    # Simular Traditional MR strategy
    traditional_returns = traditional_mr_signals * 0.012  # Placeholder

    # Comparación estadística
    lstm_returns = result_lstm['Return [%]'].pct_change().dropna()
    t_stat = (np.mean(lstm_returns) - np.mean(traditional_returns)) / np.sqrt(np.var(lstm_returns) / \
              len(lstm_returns) + np.var(traditional_returns) / len(traditional_returns))
    p_value = 2 * (1 - abs(t_stat) / np.sqrt(2))  # Aproximación simple

    # Superiority percentage
    superiority = (lstm_returns > traditional_returns).mean()

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'lstm_superiority': superiority,
        'lstm_metrics': calculate_metrics(bt_lstm),
        'traditional_mr_returns': traditional_returns
    }


def robustness_analysis(walk_forward_results):
    """Análisis de robustez para ML strategy"""
    sharpes = [r['metrics']['sharpe_ratio'] for r in walk_forward_results]
    accuracies = [r['model_accuracy'] for r in walk_forward_results]
    sortinos = [r['metrics']['sortino_ratio'] for r in walk_forward_results]
    ulcers = [r['metrics']['ulcer_index'] for r in walk_forward_results]

    return {
        'sharpe_stability': np.std(sharpes),
        'model_accuracy_avg': np.mean(accuracies),
        'avg_sortino': np.mean(sortinos),
        'avg_ulcer': np.mean(ulcers),
        'sortino_robust': np.mean(sortinos) > 1.5,
        'ulcer_robust': np.mean(ulcers) < 10,
        'model_robust': np.mean(accuracies) > 0.55,  # >55% directional accuracy
        'overall_robust': (np.std(sharpes) < 0.2 and np.mean(sortinos) > 1.5 and
                           np.mean(ulcers) < 10 and np.mean(accuracies) > 0.55)
    }


def sensitivity_analysis(df, base_params=None, n_tests=5):
    """Análisis de sensibilidad para ML strategy"""
    if base_params is None:
        base_params = {
            'prediction_threshold': 0.02,
            'confidence_threshold': 0.7,
            'max_holding_period': 10
        }

    sensitivities = {}

    for param_name, base_value in base_params.items():
        param_sensitivities = []

        # Test ±10% variation
        for factor in [0.9, 1.0, 1.1]:
            test_params = base_params.copy()
            test_params[param_name] = base_value * factor

            try:
                bt = Backtest(df, LSTMMReversionStrategy, cash=10000, commission=0.0005)
                bt.run(**test_params)
                metrics = calculate_metrics(bt)
                param_sensitivities.append(metrics['sharpe_ratio'])
            except Exception:
                param_sensitivities.append(0)

        sensitivities[param_name] = {
            'values': param_sensitivities,
            'std': np.std(param_sensitivities),
            'stable': np.std(param_sensitivities) < 0.2
        }

    return sensitivities


def anti_overfit_analysis(walk_forward_results, ab_results):
    """Análisis anti-overfit para ML strategy"""
    # Bonferroni correction
    n_tests = 5
    bonferroni_alpha = 0.05 / n_tests

    # Test significance with correction
    p_corrected = ab_results['p_value'] * n_tests
    significant_corrected = p_corrected < 0.05

    # Degradation analysis
    oos_sharpes = [r['metrics']['sharpe_ratio'] for r in walk_forward_results]
    degradation = np.std(oos_sharpes) < 0.2

    # Model stability
    accuracies = [r['model_accuracy'] for r in walk_forward_results]
    model_stable = np.std(accuracies) < 0.1

    return {
        'bonferroni_alpha': bonferroni_alpha,
        'p_corrected': p_corrected,
        'significant_corrected': significant_corrected,
        'degradation_low': degradation,
        'model_stable': model_stable,
        'overfit_risk': 'low' if significant_corrected and degradation and model_stable else 'high'
    }


def generate_report(walk_forward_results, ab_results, robustness, sensitivity, anti_overfit):
    """Generar reporte completo para ML strategy"""
    # Crear directorio de resultados
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Métricas resumen
    oos_sharpes = [r['metrics']['sharpe_ratio'] for r in walk_forward_results]
    oos_win_rates = [r['metrics']['win_rate'] for r in walk_forward_results]
    model_accuracies = [r['model_accuracy'] for r in walk_forward_results]

    summary = {
        'strategy': 'LSTM ML Mean Reversion',
        'total_periods': len(walk_forward_results),
        'avg_oos_sharpe': np.mean(oos_sharpes),
        'std_oos_sharpe': np.std(oos_sharpes),
        'avg_oos_win_rate': np.mean(oos_win_rates),
        'avg_model_accuracy': np.mean(model_accuracies),
        'ab_test_significant': ab_results['significant'],
        'ab_superiority': ab_results['lstm_superiority'],
        'robustness_overall': robustness['overall_robust'],
        'sensitivity_stable': all(s['stable'] for s in sensitivity.values()),
        'anti_overfit_risk': anti_overfit['overfit_risk'],
        'recommendation': 'Deploy' if (np.mean(oos_sharpes) > 1.3 and
                                       ab_results['significant'] and
                                       robustness['overall_robust'] and
                                       anti_overfit['overfit_risk'] == 'low') else 'Further Testing'
    }

    # Guardar resultados
    import json
    with open(RESULTS_DIR / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Guardar trades detallados
    all_trades = []
    for result in walk_forward_results:
        trades_df = result['backtest_result']._trades
        if len(trades_df) > 0:
            trades_df['period'] = result['period']
            all_trades.append(trades_df)

    if all_trades:
        trades_combined = pd.concat(all_trades)
        trades_combined.to_csv(RESULTS_DIR / 'trades.csv')

    # Generar gráficos
    generate_plots(walk_forward_results, summary)

    return summary


def generate_plots(walk_forward_results, summary):
    """Generar gráficos de análisis ML"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Sharpe ratio por periodo
    periods = [r['period'] for r in walk_forward_results]
    sharpes = [r['metrics']['sharpe_ratio'] for r in walk_forward_results]
    axes[0, 0].plot(periods, sharpes, 'o-', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=1.3, color='r', linestyle='--', alpha=0.7, label='Target 1.3')
    axes[0, 0].set_title('OOS Sharpe Ratio por Periodo (LSTM)')
    axes[0, 0].set_xlabel('Periodo Walk-Forward')
    axes[0, 0].set_ylabel('Sharpe Ratio')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Model accuracy por periodo
    accuracies = [r['model_accuracy'] for r in walk_forward_results]
    axes[0, 1].plot(periods, accuracies, 's-', color='green', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=0.55, color='r', linestyle='--', alpha=0.7, label='Target 55%')
    axes[0, 1].set_title('Model Directional Accuracy')
    axes[0, 1].set_xlabel('Periodo Walk-Forward')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Win rate por periodo
    win_rates = [r['metrics']['win_rate'] for r in walk_forward_results]
    axes[1, 0].plot(periods, win_rates, '^-', color='orange', linewidth=2, markersize=8)
    axes[1, 0].axhline(y=0.55, color='r', linestyle='--', alpha=0.7, label='Target 55%')
    axes[1, 0].set_title('OOS Win Rate por Periodo')
    axes[1, 0].set_xlabel('Periodo Walk-Forward')
    axes[1, 0].set_ylabel('Win Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Prediction confidence
    confidences = [r['metrics']['prediction_confidence'] for r in walk_forward_results]
    axes[1, 1].plot(periods, confidences, 'd-', color='purple', linewidth=2, markersize=8)
    axes[1, 1].set_title('Average Prediction Confidence')
    axes[1, 1].set_xlabel('Periodo Walk-Forward')
    axes[1, 1].set_ylabel('Confidence')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'lstm_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_complete_analysis(df_btc, traditional_mr_signals=None):
    """
    Ejecutar análisis completo de la estrategia LSTM ML

    Args:
        df_btc: DataFrame con datos BTC OHLCV
        traditional_mr_signals: Señales traditional mean reversion para comparación A/B
    """
    print("Ejecutando analisis completo LSTM ML Mean Reversion")
    print(f"Datos: {len(df_btc)} velas de {df_btc.index[0]} a {df_btc.index[-1]}")

    # Walk-forward testing con re-training
    print("\nEjecutando Walk-Forward Testing con Re-training...")
    wf_results = walk_forward_test(df_btc)

    # A/B test vs Traditional Mean Reversion
    ab_results = None
    if traditional_mr_signals is not None:
        print("\nEjecutando A/B Test vs Traditional Mean Reversion...")
        ab_results = ab_test_vs_traditional_mr(df_btc, traditional_mr_signals)

    # Robustness analysis
    print("\nEjecutando Robustness Analysis...")
    robustness = robustness_analysis(wf_results)

    # Sensitivity analysis
    print("\nEjecutando Sensitivity Analysis...")
    sensitivity = sensitivity_analysis(df_btc)

    # Anti-overfit analysis
    print("\nEjecutando Anti-Overfit Analysis...")
    anti_overfit = anti_overfit_analysis(wf_results, ab_results or {'p_value': 1.0})

    # Generar reporte
    print("\nGenerando Reporte Final...")
    summary = generate_report(wf_results, ab_results, robustness, sensitivity, anti_overfit)

    print("\nAnalisis LSTM completado!")
    print(f"Sharpe OOS promedio: {summary['avg_oos_sharpe']:.2f}")
    print(f"Win Rate promedio: {summary['avg_oos_win_rate']:.1%}")
    print(f"Model Accuracy promedio: {summary['avg_model_accuracy']:.1%}")
    print(f"Recomendacion: {summary['recommendation']}")

    return {
        'walk_forward_results': wf_results,
        'ab_results': ab_results,
        'robustness': robustness,
        'sensitivity': sensitivity,
        'anti_overfit': anti_overfit,
        'summary': summary
    }


if __name__ == "__main__":
    # Ejemplo de uso
    print("LSTM ML Mean Reversion Strategy")
    print("Para usar: run_complete_analysis(df_btc, traditional_mr_signals)")
    print("Resultados se guardan en results/lstm_ml_reversion/")
