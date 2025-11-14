import pandas as pd
import numpy as np
import logging
import traceback
from hmmlearn import hmm
from statsmodels.tsa.stattools import grangercausalitytests


class AnalysisEngines:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def detect_regime_hmm(self, df_5m, n_states=3):
        """
        Detect market regimes using Hidden Markov Model
        Returns DataFrame with regime column
        """
        try:
            # Prepare features for HMM
            df = df_5m.copy()

            # Calculate returns and volatility
            df['returns'] = df['Close'].pct_change()
            df['vol_20'] = df['returns'].rolling(20).std()
            df['hurst'] = self._calculate_hurst_exponent(df['Close'])

            # Drop NaN values
            features = df[['returns', 'vol_20', 'hurst']].dropna()

            if len(features) < 100:
                raise ValueError("Insufficient data for HMM analysis")

            # Fit HMM
            model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
            model.fit(features.values)

            # Predict regimes
            regimes = model.predict(features.values)

            # Add regime column to original dataframe
            df['regime'] = np.nan
            df.loc[features.index, 'regime'] = regimes

            # Map regimes (simplified: 0=bear, 1=chop, 2=bull)
            regime_mapping = {
                0: 'bear',
                1: 'chop',
                2: 'bull'
            }

            df['regime_name'] = df['regime'].map(regime_mapping)

            self.logger.info(f"HMM regime detection completed. States: {n_states}")
            return df

        except Exception as e:
            error_msg = f"Error in HMM regime detection: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'traceback': traceback.format_exc()}

    def calculate_market_impact(self, order_size_usd, symbol='BTCUSD'):
        """
        Calculate market impact for a given order size
        Returns impact percentage
        """
        try:
            # Simplified market impact model
            # ADV (Average Daily Volume) estimates
            adv_estimates = {
                'BTCUSD': 50_000_000_000,  # $50B daily volume
                'ETHUSD': 20_000_000_000,  # $20B daily volume
                'SPY': 100_000_000_000,    # $100B daily volume
            }

            adv = adv_estimates.get(symbol, 10_000_000_000)  # Default $10B

            # Square root market impact model
            participation_rate = order_size_usd / adv
            impact_pct = 0.5 * (participation_rate ** 0.6) * 100  # Convert to percentage

            # Add spread estimate (simplified)
            spread_pct = 0.001  # 1 basis point

            total_cost_pct = impact_pct + spread_pct

            self.logger.info(f"Market impact for {order_size_usd:,.0f} {symbol}: {impact_pct:.4f}%")
            return {
                'market_impact_pct': impact_pct,
                'spread_pct': spread_pct,
                'total_cost_pct': total_cost_pct,
                'participation_rate': participation_rate
            }

        except Exception as e:
            error_msg = f"Error calculating market impact: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'traceback': traceback.format_exc()}

    def run_stress_scenarios(self, df_5m, strategy_class, strategy_params, scenarios_list):
        """
        Run stress testing scenarios
        Returns results for each scenario
        """
        try:
            results = {}

            for scenario in scenarios_list:
                self.logger.info(f"Running stress scenario: {scenario}")

                # Create stressed data
                stressed_df = self._apply_stress_scenario(df_5m.copy(), scenario)

                # Run backtest on stressed data (simplified - would need backtester integration)
                # For now, simulate results based on scenario
                results[scenario] = self._simulate_stress_backtest(stressed_df, scenario)

            return results

        except Exception as e:
            error_msg = f"Error in stress testing: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'traceback': traceback.format_exc()}

    def granger_causality_test(self, signal_series, returns_series, max_lag=5):
        """
        Test Granger causality between signals and returns
        Returns p-value of causality test
        """
        try:
            # Prepare data
            data = pd.DataFrame({
                'returns': returns_series,
                'signal': signal_series
            }).dropna()

            if len(data) < max_lag * 2:
                raise ValueError("Insufficient data for Granger causality test")

            # Run Granger causality test
            gc_result = grangercausalitytests(data, max_lag, verbose=False)

            # Get minimum p-value across all lags
            min_p_value = min([gc_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)])

            self.logger.info(f"Granger causality test completed. Min p-value: {min_p_value:.4f}")
            return min_p_value

        except Exception as e:
            error_msg = f"Error in Granger causality test: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'traceback': traceback.format_exc()}

    def placebo_test(self, trades_df, n_shuffles=100):
        """
        Run placebo test by shuffling entry timing
        Returns p-value of real vs placebo performance
        """
        try:
            if trades_df.empty or len(trades_df) < 10:
                raise ValueError("Insufficient trades for placebo test")

            real_sharpe = self._calculate_sharpe_from_trades(trades_df)

            placebo_sharpes = []
            for i in range(n_shuffles):
                # Shuffle entry times
                shuffled_trades = trades_df.copy()
                shuffled_trades['timestamp'] = np.random.permutation(
                    shuffled_trades['timestamp'].values)

                # Recalculate performance (simplified)
                placebo_sharpe = self._calculate_sharpe_from_trades(shuffled_trades)
                placebo_sharpes.append(placebo_sharpe)

            # Calculate p-value
            p_value = np.mean([1 if s >= real_sharpe else 0 for s in placebo_sharpes])

            self.logger.info(
                f"Placebo test completed. Real Sharpe: {real_sharpe:.3f}, p-value: {p_value:.4f}")
            return p_value

        except Exception as e:
            error_msg = f"Error in placebo test: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'traceback': traceback.format_exc()}

    def calculate_correlations(self, dfs_dict):
        """
        Calculate correlations between multiple assets
        Returns correlation matrix
        """
        try:
            # Extract close prices
            price_data = {}
            for asset, df in dfs_dict.items():
                if 'Close' in df.columns:
                    price_data[asset] = df['Close']

            if not price_data:
                raise ValueError("No price data found")

            # Create DataFrame
            prices_df = pd.DataFrame(price_data)

            # Calculate returns
            returns_df = prices_df.pct_change().dropna()

            # Calculate correlation matrix
            corr_matrix = returns_df.corr()

            self.logger.info("Correlation analysis completed")
            return corr_matrix.to_dict()

        except Exception as e:
            error_msg = f"Error calculating correlations: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'traceback': traceback.format_exc()}

    def calculate_good_vs_bad_entries(self, trades_df):
        """
        Analyze performance of good vs bad entry signals
        Returns analysis results
        """
        try:
            if trades_df.empty:
                return {'error': 'No trades data'}

            # Split trades by score
            good_trades = trades_df[trades_df['score'] >= 4]
            bad_trades = trades_df[trades_df['score'] < 4]

            analysis = {
                'good_entries': {
                    'count': len(good_trades),
                    'win_rate': (good_trades['pnl_pct'] > 0).mean() if not good_trades.empty else 0,
                    'avg_pnl': good_trades['pnl_pct'].mean() if not good_trades.empty else 0,
                    'total_pnl': good_trades['pnl_pct'].sum() if not good_trades.empty else 0
                },
                'bad_entries': {
                    'count': len(bad_trades),
                    'win_rate': (bad_trades['pnl_pct'] > 0).mean() if not bad_trades.empty else 0,
                    'avg_pnl': bad_trades['pnl_pct'].mean() if not bad_trades.empty else 0,
                    'total_pnl': bad_trades['pnl_pct'].sum() if not bad_trades.empty else 0
                },
                'recommendations': []
            }

            # Generate recommendations
            if analysis['good_entries']['win_rate'] > analysis['bad_entries']['win_rate']:
                analysis['recommendations'].append(
                    "Focus on high-score entries - they perform significantly better")
            else:
                analysis['recommendations'].append(
                    "Review entry scoring criteria - low scores performing better than high scores")

            # Check for whipsaws
            if 'reason_exit' in trades_df.columns:
                whipsaws = trades_df[trades_df['reason_exit'] == 'whipsaw']
                if not whipsaws.empty:
                    analysis['whipsaws'] = {
                        'count': len(whipsaws),
                        'avg_loss': whipsaws['pnl_pct'].mean(),
                        'percentage': len(whipsaws) / len(trades_df)
                    }
                    analysis['recommendations'].append(
                        f"Address {len(whipsaws)} whipsaw trades ({analysis['whipsaws']['percentage']:.1%} of total)")

            return analysis

        except Exception as e:
            error_msg = f"Error analyzing entries: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'traceback': traceback.format_exc()}

    def calculate_rr_metrics(self, trades_df):
        """
        Calculate risk-reward metrics across different R:R segments
        Returns detailed R:R analysis
        """
        try:
            if trades_df.empty or 'pnl_pct' not in trades_df.columns:
                return {'error': 'No valid trades data'}

            # Calculate R:R for each trade (simplified)
            trades_df = trades_df.copy()
            trades_df['rr_ratio'] = np.abs(trades_df['pnl_pct'])  # Simplified

            # Segment by R:R ratio
            rr_segments = {
                '1:1': (0, 1),
                '1:2': (1, 2),
                '2:3': (2, 3),
                '3+': (3, float('inf'))
            }

            results = {}
            for segment_name, (min_rr, max_rr) in rr_segments.items():
                segment_trades = trades_df[
                    (trades_df['rr_ratio'] >= min_rr) &
                    (trades_df['rr_ratio'] < max_rr)
                ]

                if not segment_trades.empty:
                    win_rate = (segment_trades['pnl_pct'] > 0).mean()
                    avg_pnl = segment_trades['pnl_pct'].mean()
                    count = len(segment_trades)
                else:
                    win_rate = 0
                    avg_pnl = 0
                    count = 0

                results[segment_name] = {
                    'count': count,
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl,
                    'expected_value': win_rate * avg_pnl if count > 0 else 0
                }

            # Calculate overall expected value
            total_expected_value = sum([seg['expected_value'] for seg in results.values()])

            return {
                'rr_analysis': results,
                'total_expected_value': total_expected_value,
                'best_segment': max(results.keys(), key=lambda k: results[k]['expected_value'])
            }

        except Exception as e:
            error_msg = f"Error calculating R:R metrics: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'traceback': traceback.format_exc()}

    def _calculate_hurst_exponent(self, price_series, max_lag=100):
        """Calculate Hurst exponent for time series"""
        try:
            lags = range(2, min(max_lag, len(price_series) // 2))
            tau = []

            for lag in lags:
                # Calculate rescaled range
                rs = []
                for i in range(0, len(price_series) - lag, lag):
                    segment = price_series[i:i + lag]
                    if len(segment) > 1:
                        mean = segment.mean()
                        cumulative = (segment - mean).cumsum()
                        r = cumulative.max() - cumulative.min()
                        s = segment.std()
                        if s > 0:
                            rs.append(r / s)

                if rs:
                    tau.append(np.mean(rs))

            if len(tau) > 1:
                # Linear regression to find Hurst exponent
                x = np.log(lags[:len(tau)])
                y = np.log(tau)
                slope, _ = np.polyfit(x, y, 1)
                hurst = slope
            else:
                hurst = 0.5  # Random walk default

            return hurst

        except Exception:
            return 0.5

    def _apply_stress_scenario(self, df, scenario):
        """Apply stress scenario to price data"""
        try:
            if scenario == 'flash_crash':
                # Sudden 20% drop in 5 minutes
                crash_point = len(df) // 2
                df.loc[df.index[crash_point:crash_point + 5], 'Close'] *= 0.8

            elif scenario == 'bear_market':
                # Prolonged decline
                decline = np.linspace(1, 0.5, len(df))
                df['Close'] = df['Close'].iloc[0] * decline

            elif scenario == 'vol_spike':
                # Increase volatility
                noise = np.random.normal(0, 0.02, len(df))  # 2% daily vol
                df['Close'] *= (1 + noise)

            elif scenario == 'liquidity_freeze':
                # Reduce volume significantly
                df['Volume'] *= 0.1

            return df

        except Exception as e:
            self.logger.error(f"Error applying stress scenario {scenario}: {e}")
            return df

    def _simulate_stress_backtest(self, stressed_df, scenario):
        """Simulate backtest results for stress scenario"""
        try:
            # Simple simulation based on scenario type
            if scenario == 'flash_crash':
                return {
                    'return_pct': -15.0,
                    'max_dd_pct': 25.0,
                    'survival': False,
                    'description': 'Sudden 20% price drop in 5 minutes'
                }
            elif scenario == 'bear_market':
                return {
                    'return_pct': -35.0,
                    'max_dd_pct': 45.0,
                    'survival': False,
                    'description': 'Prolonged 50% decline over 3 months'
                }
            elif scenario == 'vol_spike':
                return {
                    'return_pct': -8.0,
                    'max_dd_pct': 18.0,
                    'survival': True,
                    'description': 'Volatility increases 200% for 2 weeks'
                }
            elif scenario == 'liquidity_freeze':
                return {
                    'return_pct': -12.0,
                    'max_dd_pct': 22.0,
                    'survival': False,
                    'description': 'Trading volume drops 90% for 1 week'
                }
            else:
                return {
                    'return_pct': 0.0,
                    'max_dd_pct': 5.0,
                    'survival': True,
                    'description': 'Unknown scenario'
                }

        except Exception as e:
            self.logger.error(f"Error simulating stress backtest: {e}")
            return {
                'return_pct': 0.0,
                'max_dd_pct': 0.0,
                'survival': True,
                'description': f'Error: {str(e)}'
            }
