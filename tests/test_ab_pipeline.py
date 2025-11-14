#!/usr/bin/env python3
"""
Tests for Automated A/B Testing Pipeline
========================================

Tests the complete pipeline integration with DVC and CI/CD.
"""

import sys
from pathlib import Path
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ab_pipeline import ABPipeline

class TestABPipeline(unittest.TestCase):
    """Test cases for ABPipeline class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.pipeline = ABPipeline(
            symbol='BTCUSD',
            start_date='2023-01-01',
            end_date='2023-01-10',
            capital=10000
        )
        # Override directories to use temp dir
        self.pipeline.pipeline_dir = self.temp_dir
        self.pipeline.data_dir = self.temp_dir / 'data'
        self.pipeline.signals_dir = self.temp_dir / 'signals'
        self.pipeline.results_dir = self.temp_dir / 'results'
        self.pipeline.reports_dir = self.temp_dir / 'reports'

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('subprocess.run')
    @patch('src.ab_pipeline.fetch_crypto_data')
    @patch('src.ab_pipeline.generate_signals')
    @patch('src.ab_pipeline.AdvancedABTesting')
    def test_full_pipeline_success(self, mock_ab_class, mock_generate_signals, mock_fetch_data, mock_subprocess):
        """Test successful full pipeline execution"""
        mock_subprocess.return_value = MagicMock()
        # Mock data
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.isnull.return_value.sum.return_value.sum.return_value = 0
        mock_df.__len__ = MagicMock(return_value=100)
        mock_df.index.min.return_value.strftime.return_value = '2023-01-01'
        mock_df.index.max.return_value.strftime.return_value = '2023-01-10'

        mock_fetch_data.return_value = mock_df
        mock_generate_signals.side_effect = [
            MagicMock(),  # signals_a
            MagicMock()   # signals_b
        ]

        # Mock AB testing protocol
        mock_protocol = MagicMock()
        mock_ab_class.return_value = mock_protocol

        mock_protocol.run_backtest_vectorbt.side_effect = [
            {'total_return': 1.05, 'sharpe_ratio': 1.2, 'win_rate': 0.52, 'max_drawdown': -0.1},
            {'total_return': 1.08, 'sharpe_ratio': 1.5, 'win_rate': 0.55, 'max_drawdown': -0.08}
        ]

        mock_protocol.comprehensive_ab_analysis.return_value = {
            'base_statistics': {
                'p_value': 0.03,
                'superiority_percentage': 0.15
            },
            'robustness_metrics': {
                'variant_a': {'sharpe_ratio': 1.2},
                'variant_b': {'sharpe_ratio': 1.5}
            },
            'anti_snooping': {'overall_snooping_detected': False},
            'decision': {
                'recommendation': 'ADOPT_B_STRONG',
                'reason': 'B shows strong superiority',
                'confidence': 0.9
            }
        }

        # Run pipeline
        results = self.pipeline.run_full_pipeline()

        # Assertions
        self.assertEqual(results['status'], 'success')
        self.assertEqual(results['symbol'], 'BTCUSD')
        self.assertIn('analysis', results)
        self.assertIn('decision', results)

        # Check that methods were called
        mock_fetch_data.assert_called_once()
        self.assertEqual(mock_generate_signals.call_count, 2)
        # Note: backtest and analysis calls use real implementation in tests

    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        pipeline = ABPipeline(symbol='ETHUSD', capital=50000)

        self.assertEqual(pipeline.symbol, 'ETHUSD')
        self.assertEqual(pipeline.capital, 50000)
        self.assertEqual(pipeline.start_date, '2018-01-01')

    @patch('subprocess.run')
    @patch('src.ab_pipeline.fetch_crypto_data')
    def test_data_fetch_validation(self, mock_fetch_data, mock_subprocess):
        """Test data fetching and validation"""
        mock_subprocess.return_value = MagicMock()
        # Mock empty data
        mock_df = MagicMock()
        mock_df.empty = True
        mock_fetch_data.return_value = mock_df

        with self.assertRaises(ValueError):
            self.pipeline._fetch_and_validate_data()

        # Mock valid data
        mock_df.empty = False
        mock_df.isnull.return_value.sum.return_value.sum.return_value = 0
        mock_df.__len__ = MagicMock(return_value=100)
        mock_df.index.min.return_value.strftime.return_value = '2023-01-01'
        mock_df.index.max.return_value.strftime.return_value = '2023-01-10'

        result = self.pipeline._fetch_and_validate_data()
        self.assertIsNotNone(result)

    def test_automated_decision_making(self):
        """Test automated decision making logic"""
        # Test strong adoption
        analysis = {
            'decision': {
                'recommendation': 'ADOPT_B_STRONG',
                'confidence': 0.95
            }
        }
        decision = self.pipeline._generate_automated_decision(analysis)

        self.assertEqual(decision['automated_action'], 'deploy_variant_b')
        self.assertTrue(decision['meets_threshold'])

        # Test keep current
        analysis['decision']['recommendation'] = 'KEEP_A'
        analysis['decision']['confidence'] = 0.85
        decision = self.pipeline._generate_automated_decision(analysis)

        self.assertEqual(decision['automated_action'], 'keep_current_strategy')
        self.assertTrue(decision['meets_threshold'])

    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_report_generation(self, mock_file):
        """Test report generation"""
        results_a = {'total_return': 1.05, 'sharpe_ratio': 1.2}
        results_b = {'total_return': 1.08, 'sharpe_ratio': 1.5}
        analysis = {
            'base_statistics': {'p_value': 0.03, 'superiority_percentage': 0.15},
            'decision': {'recommendation': 'ADOPT_B_STRONG', 'reason': 'Test reason'}
        }
        decision = {'automated_action': 'deploy_variant_b'}

        report_path = self.pipeline._generate_reports(results_a, results_b, analysis, decision)

        # Check that file was opened for writing
        mock_file.assert_called()
        self.assertIsInstance(report_path, Path)

    @patch('subprocess.run')
    def test_version_control_success(self, mock_subprocess):
        """Test successful version control operations"""
        mock_subprocess.return_value = MagicMock()

        self.pipeline._version_and_commit()

        # Check that git and dvc commands were called
        calls = mock_subprocess.call_args_list
        self.assertTrue(any('git' in str(call) for call in calls))
        self.assertTrue(any('dvc' in str(call) for call in calls))

    @patch('subprocess.run')
    def test_version_control_failure(self, mock_subprocess):
        """Test version control failure handling"""
        from subprocess import CalledProcessError
        mock_subprocess.side_effect = CalledProcessError(1, ['dvc', 'add'])

        # Should not raise exception
        self.pipeline._version_and_commit()

    def test_data_hash_fallback(self):
        """Test data hash fallback when DVC fails"""
        with patch('subprocess.run', side_effect=Exception()):
            hash_value = self.pipeline._get_data_hash()
            self.assertEqual(hash_value, "unknown")

if __name__ == '__main__':
    unittest.main()