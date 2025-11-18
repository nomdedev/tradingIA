#!/usr/bin/env python3
"""
Tests for Advanced A/B Testing Protocol
=======================================

Tests robustness metrics, anti-snooping detection, and comprehensive analysis.
"""

import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ab_advanced import AdvancedABTesting

class TestAdvancedABTesting(unittest.TestCase):
    """Test cases for AdvancedABTesting class"""

    def setUp(self):
        """Set up test fixtures"""
        self.protocol = AdvancedABTesting()

        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='5min')
        self.sample_data = pd.DataFrame({
            'close': np.random.normal(50000, 1000, 1000),
            'high': np.random.normal(50100, 1000, 1000),
            'low': np.random.normal(49900, 1000, 1000),
            'volume': np.random.normal(100, 20, 1000)
        }, index=dates)

        # Sample returns
        self.sample_returns = pd.Series(np.random.normal(0.001, 0.02, 1000))

    def test_calculate_robustness_metrics(self):
        """Test robustness metrics calculation"""
        metrics = self.protocol.calculate_robustness_metrics(self.sample_returns)

        # Check that all expected metrics are present
        expected_keys = ['sharpe_ratio', 'sortino_ratio', 'ulcer_index', 'probabilistic_sharpe', 'var_95']
        for key in expected_keys:
            self.assertIn(key, metrics)

        # Check that metrics are reasonable numbers
        for key, value in metrics.items():
            self.assertIsInstance(value, (int, float))
            self.assertFalse(np.isnan(value))

    def test_anti_snooping_analysis(self):
        """Test anti-snooping analysis"""
        returns_a = pd.Series(np.random.normal(0.001, 0.02, 500))
        returns_b = pd.Series(np.random.normal(0.002, 0.02, 500))

        results_a = {'returns': returns_a}
        results_b = {'returns': returns_b}

        analysis = self.protocol.anti_snooping_analysis(results_a, results_b)

        # Check structure
        expected_keys = ['aic_snooping', 'aic_values', 'whites_reality_check',
                        'bonferroni_correction', 'bootstrap_ci', 'overall_snooping_detected']
        for key in expected_keys:
            self.assertIn(key, analysis)

    def test_comprehensive_ab_analysis(self):
        """Test comprehensive A/B analysis"""
        # Mock results
        results_a = {
            'returns': pd.Series(np.random.normal(0.001, 0.02, 500)),
            'total_return': 1.05,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.1,
            'win_rate': 0.52
        }
        results_b = {
            'returns': pd.Series(np.random.normal(0.002, 0.02, 500)),
            'total_return': 1.08,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.08,
            'win_rate': 0.55
        }

        analysis = self.protocol.comprehensive_ab_analysis(results_a, results_b)

        # Check that analysis contains expected sections
        expected_sections = ['base_statistics', 'robustness_metrics',
                           'anti_snooping', 'decision']
        for section in expected_sections:
            self.assertIn(section, analysis)

    def test_advanced_decision_making(self):
        """Test advanced decision making logic"""
        base_stats = {
            'p_value': 0.03,
            'superiority_percentage': 0.15
        }

        robust_a = {
            'sortino_ratio': 1.2,
            'ulcer_index': 0.05,
            'probabilistic_sharpe': 0.6
        }

        robust_b = {
            'sortino_ratio': 1.5,
            'ulcer_index': 0.03,
            'probabilistic_sharpe': 0.7
        }

        snooping = {'overall_snooping_detected': False}

        decision = self.protocol._advanced_decision_making(base_stats, robust_a, robust_b, snooping)

        # Check decision structure
        expected_keys = ['recommendation', 'reason', 'confidence']
        for key in expected_keys:
            self.assertIn(key, decision)

        # Check that recommendation is valid
        valid_recommendations = ['ADOPT_B_STRONG', 'ADOPT_B_WEAK', 'ADOPT_B_LOW_RISK', 'KEEP_A']
        self.assertIn(decision['recommendation'], valid_recommendations)

    def test_multi_armed_bandit(self):
        """Test multi-armed bandit variant"""
        # Create simple signals
        variant_a_signals = pd.Series([1, -1, 1, -1] * 25)  # Simple alternating signals
        variant_b_signals = pd.Series([1, 1, -1, -1] * 25)  # Different pattern

        result = self.protocol.multi_armed_bandit_test(variant_a_signals, variant_b_signals, self.sample_data)

        # Check result structure
        expected_keys = ['final_arm_stats', 'win_rates', 'best_arm']
        for key in expected_keys:
            self.assertIn(key, result)

        # Check that best_arm is either 'A' or 'B'
        self.assertIn(result['best_arm'], ['A', 'B'])

    @patch('src.ab_advanced.bootstrap')
    def test_bootstrap_error_handling(self, mock_bootstrap):
        """Test error handling in bootstrap operations"""
        # Make bootstrap raise an exception
        mock_bootstrap.side_effect = Exception("Bootstrap failed")

        metrics = self.protocol.calculate_robustness_metrics(self.sample_returns)

        # Should still return valid metrics (with defaults)
        self.assertIn('probabilistic_sharpe', metrics)
        self.assertEqual(metrics['probabilistic_sharpe'], 0.5)  # Default value

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Empty returns
        empty_returns = pd.Series([])
        metrics = self.protocol.calculate_robustness_metrics(empty_returns)
        self.assertIn('sharpe_ratio', metrics)

        # Single value returns
        single_returns = pd.Series([0.01])
        metrics = self.protocol.calculate_robustness_metrics(single_returns)
        self.assertIn('sharpe_ratio', metrics)

        # All zero returns
        zero_returns = pd.Series([0.0] * 100)
        metrics = self.protocol.calculate_robustness_metrics(zero_returns)
        self.assertIn('sharpe_ratio', metrics)

if __name__ == '__main__':
    unittest.main()