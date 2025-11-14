#!/usr/bin/env python3
"""
Tests for A/B Testing Base Protocol
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ab_base_protocol import ABTestingBase, ab_protocol

class TestABTestingBase:
    """Test cases for A/B testing base protocol"""

    def setup_method(self):
        """Setup test data"""
        self.protocol = ABTestingBase()

        # Create mock BTC data
        dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
        np.random.seed(42)
        self.df_mock = pd.DataFrame({
            'open': 45000 + np.random.randn(1000).cumsum() * 10,
            'high': 45000 + np.random.randn(1000).cumsum() * 10 + 50,
            'low': 45000 + np.random.randn(1000).cumsum() * 10 - 50,
            'close': 45000 + np.random.randn(1000).cumsum() * 10,
            'volume': np.random.randint(100, 1000, 1000)
        }, index=dates)

        # Create mock signals
        self.signals_a = pd.Series(np.random.choice([0, 1, -1], 1000, p=[0.8, 0.1, 0.1]), index=dates)
        self.signals_b = pd.Series(np.random.choice([0, 1, -1], 1000, p=[0.75, 0.15, 0.1]), index=dates)

    def test_data_split(self):
        """Test random data splitting"""
        df_a, df_b = self.protocol.split_data_random(self.df_mock)

        # Check that splits are non-empty
        assert len(df_a) > 0
        assert len(df_b) > 0
        
        # Check that dates don't overlap
        dates_a = set(df_a.index.date)
        dates_b = set(df_b.index.date)
        assert len(dates_a.intersection(dates_b)) == 0

    # def test_backtest_execution(self):
    #     """Test backtest execution and results structure"""
    #     df_a, df_b = self.protocol.split_data_random(self.df_mock)
    #     results = self.protocol.run_backtest(df_a, self.signals_a)
    #     
    #     # Check results structure
    #     required_keys = ['total_return', 'sharpe_ratio', 'win_rate', 'total_trades', 'max_drawdown', 'returns', 'trades']
    #     for key in required_keys:
    #         assert key in results
    #         
    #     # Check types
    #     assert isinstance(results['total_return'], (int, float))
    #     assert isinstance(results['sharpe_ratio'], (int, float))
    #     assert isinstance(results['win_rate'], (int, float))
    #     assert isinstance(results['total_trades'], int)

    # def test_statistical_comparison(self):
    #     """Test statistical comparison between strategies"""
    #     results_a = {'returns': pd.Series([0.01, -0.005, 0.02] * 100)}
    #     results_b = {'returns': pd.Series([0.015, -0.003, 0.025] * 100)}
    #     
    #     stats = self.protocol.calculate_statistics(results_a, results_b)
    #     
    #     # Check statistical outputs
    #     required_keys = ['t_statistic', 'p_value', 'significant', 'superiority_percentage', 'cohens_d']
    #     for key in required_keys:
    #         assert key in stats
    #     
    #     assert isinstance(stats['p_value'], float)
    #     assert isinstance(stats['significant'], bool)
    #     assert 0 <= stats['superiority_percentage'] <= 1

    def test_decision_making(self):
        """Test decision making logic"""
        # Mock results with clear B superiority
        results_a = {'total_trades': 200, 'sharpe_ratio': 1.2}
        results_b = {'total_trades': 200, 'sharpe_ratio': 1.6}
        stats = {'p_value': 0.02, 'superiority_percentage': 0.7, 'significant': True}
        
        decision = self.protocol.make_decision(results_a, results_b, stats)

        assert 'recommendation' in decision
        assert 'reason' in decision
        assert 'confidence' in decision
        assert decision['recommendation'] == 'ADOPT_B'

        # Test insufficient trades
        results_a_few = {'total_trades': 50, 'sharpe_ratio': 1.2}
        results_b_few = {'total_trades': 45, 'sharpe_ratio': 1.6}
        decision = self.protocol.make_decision(results_a_few, results_b_few, stats, min_trades=100)

        assert decision['recommendation'] == 'INSUFFICIENT_DATA'
        assert 'Insufficient trades' in decision['reason']

    def test_aa_test(self):
        """Test A/A bias detection"""
        aa_results = self.protocol.run_aa_test(self.df_mock, self.signals_a, n_runs=3)

        required_keys = ['n_runs', 'false_positive_rate', 'average_p_value', 'bias_detected']
        for key in required_keys:
            assert key in aa_results

        assert aa_results['n_runs'] == 3
        assert isinstance(aa_results['bias_detected'], bool)

def test_ab_protocol_structure():
    """Test main ab_protocol function structure"""
    # This is a smoke test for the main function
    
    # Check that the function exists and is callable
    assert callable(ab_protocol)

if __name__ == "__main__":
    pytest.main([__file__])

if __name__ == "__main__":
    pytest.main([__file__])