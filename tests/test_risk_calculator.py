import pytest
import numpy as np
import pandas as pd
from core.risk.risk_metrics import RiskMetricsCalculator

class TestRiskMetricsCalculator:
    
    def test_var_historical(self):
        """Test Historical VaR calculation"""
        # Create a predictable distribution
        # 100 returns from -5% to +4%
        returns = np.linspace(-0.05, 0.04, 100) 
        calc = RiskMetricsCalculator(returns)
        
        # 95% confidence -> 5th percentile
        # In our sorted array of 100 items, the 5th item (index 4) is approx -0.048
        # VaR should be positive 0.048 (4.8%)
        var = calc.calculate_var(confidence_level=0.95, method="historical")
        
        # Percentile 5 of linspace(-0.05, 0.04, 100)
        expected_percentile = np.percentile(returns, 5) # Should be around -0.0455
        expected_var = -expected_percentile
        
        assert var == pytest.approx(expected_var, abs=0.001)
        assert var > 0

    def test_cvar_calculation(self):
        """Test CVaR (Expected Shortfall) calculation"""
        # Returns: [-10%, -5%, -1%, 1%, 2%, ...]
        returns = np.array([-0.10, -0.05, -0.01, 0.01, 0.02] * 20) # 100 items
        calc = RiskMetricsCalculator(returns)
        
        # 95% confidence -> 5% worst returns
        # 5% of 100 items = 5 items.
        # The worst returns are -0.10 (20 times in the array? No, array is repeated)
        # Wait, array is [-0.1, -0.05, ...] repeated 20 times.
        # So we have 20 instances of -0.10, 20 of -0.05, etc.
        # Sorted: 20x -0.10, 20x -0.05, ...
        # 5th percentile of 100 items is roughly between the 5th and 6th item.
        # All bottom 20 items are -0.10.
        # So VaR threshold is -0.10.
        # Tail losses are all -0.10.
        # CVaR should be 0.10.
        
        cvar = calc.calculate_cvar(confidence_level=0.95)
        assert cvar == pytest.approx(0.10)

    def test_max_drawdown(self):
        """Test Max Drawdown calculation"""
        calc = RiskMetricsCalculator()
        
        # Equity: 100 -> 110 -> 99 -> 120
        # Peak 1: 110. Drop to 99. DD = (110-99)/110 = 11/110 = 0.10 (10%)
        equity = [100, 110, 99, 120]
        mdd = calc.calculate_max_drawdown(equity)
        assert mdd == pytest.approx(0.10)
        
        # Equity: 100 -> 90 -> 80 -> 70 (Continuous loss)
        # Peak is 100. Low is 70. DD = 30%.
        equity = [100, 90, 80, 70]
        mdd = calc.calculate_max_drawdown(equity)
        assert mdd == pytest.approx(0.30)

    def test_monte_carlo(self):
        """Test Monte Carlo simulation structure"""
        returns = np.random.normal(0.001, 0.02, 100) # Mean 0.1%, Vol 2%
        calc = RiskMetricsCalculator(returns)
        
        results = calc.monte_carlo_simulation(num_simulations=50, horizon=10)
        
        assert "mean_final" in results
        assert "paths" in results
        assert results["paths"].shape == (50, 10)
        
    def test_empty_inputs(self):
        """Test handling of empty inputs"""
        calc = RiskMetricsCalculator([])
        assert calc.calculate_var() == 0.0
        assert calc.calculate_cvar() == 0.0
        assert calc.calculate_max_drawdown([]) == 0.0
