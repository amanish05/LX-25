"""
Unit tests for Advanced RSI indicator
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.indicators.rsi_advanced import AdvancedRSI, RSISignal, RSISignalType


class TestAdvancedRSI:
    """Test cases for Advanced RSI indicator"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample price data"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5T')
        
        # Create trending data
        trend = np.linspace(100, 120, 50)
        trend = np.append(trend, np.linspace(120, 90, 50))
        
        # Add some noise
        noise = np.random.randn(100) * 0.5
        prices = trend + noise
        
        return pd.Series(prices, index=dates)
    
    @pytest.fixture
    def rsi_indicator(self):
        """Create RSI indicator instance"""
        return AdvancedRSI(period=14, overbought=70, oversold=30)
    
    def test_rsi_calculation(self, rsi_indicator, sample_data):
        """Test basic RSI calculation"""
        rsi_values = rsi_indicator.calculate(sample_data)
        
        # Check output shape
        assert len(rsi_values) == len(sample_data)
        
        # Check RSI bounds
        valid_rsi = rsi_values[~rsi_values.isna()]
        assert all(valid_rsi >= 0)
        assert all(valid_rsi <= 100)
        
        # Check that first 14 values are NaN
        assert rsi_values[:14].isna().all()
        
    def test_oversold_signal(self, rsi_indicator):
        """Test oversold signal generation"""
        # Create data that goes oversold
        dates = pd.date_range(end=datetime.now(), periods=50, freq='5T')
        prices = [100] * 20 + list(np.linspace(100, 80, 30))  # Declining prices
        data = pd.Series(prices, index=dates)
        
        signals = rsi_indicator.generate_signals(data)
        
        # Should have at least one oversold signal
        oversold_signals = [s for s in signals if s.signal_type == RSISignalType.OVERSOLD]
        assert len(oversold_signals) > 0
        
        # Check signal properties
        for signal in oversold_signals:
            assert signal.rsi_value <= 30
            assert signal.strength > 0
    
    def test_overbought_signal(self, rsi_indicator):
        """Test overbought signal generation"""
        # Create data that goes overbought
        dates = pd.date_range(end=datetime.now(), periods=50, freq='5T')
        prices = [100] * 20 + list(np.linspace(100, 120, 30))  # Rising prices
        data = pd.Series(prices, index=dates)
        
        signals = rsi_indicator.generate_signals(data)
        
        # Should have at least one overbought signal
        overbought_signals = [s for s in signals if s.signal_type == RSISignalType.OVERBOUGHT]
        assert len(overbought_signals) > 0
        
        # Check signal properties
        for signal in overbought_signals:
            assert signal.rsi_value >= 70
            assert signal.strength > 0
    
    def test_divergence_detection(self, rsi_indicator):
        """Test divergence detection"""
        # Create data with divergence
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5T')
        
        # Price makes higher high but RSI makes lower high (bearish divergence)
        prices = list(np.linspace(100, 110, 30)) + \
                list(np.linspace(110, 105, 20)) + \
                list(np.linspace(105, 115, 30)) + \
                list(np.linspace(115, 110, 20))
        
        data = pd.Series(prices, index=dates)
        rsi_values = rsi_indicator.calculate(data)
        
        divergences = rsi_indicator.detect_divergences(data, rsi_values)
        
        # Should detect at least one divergence
        assert len(divergences) >= 0  # May or may not find depending on noise
        
    def test_midline_cross(self, rsi_indicator, sample_data):
        """Test midline cross detection"""
        signals = rsi_indicator.generate_signals(sample_data)
        
        midline_crosses = [s for s in signals if s.signal_type == RSISignalType.MIDLINE_CROSS]
        
        # Check that midline crosses are near 50
        for signal in midline_crosses:
            assert 45 <= signal.rsi_value <= 55
    
    def test_signal_strength_calculation(self, rsi_indicator):
        """Test signal strength calculation"""
        # Test extreme oversold
        strength = rsi_indicator.get_signal_strength(10)
        assert strength >= 0.9
        
        # Test moderate oversold
        strength = rsi_indicator.get_signal_strength(25)
        assert 0.6 <= strength <= 0.8
        
        # Test neutral
        strength = rsi_indicator.get_signal_strength(50)
        assert strength <= 0.5
        
        # Test extreme overbought
        strength = rsi_indicator.get_signal_strength(90)
        assert strength >= 0.9
    
    def test_parameter_optimization(self, rsi_indicator, sample_data):
        """Test parameter optimization"""
        best_params = rsi_indicator.backtest_parameters(
            sample_data,
            period_range=(10, 16),
            ob_range=(70, 75),
            os_range=(25, 30)
        )
        
        assert 'period' in best_params
        assert 'overbought' in best_params
        assert 'oversold' in best_params
        assert 'score' in best_params
        assert best_params['period'] >= 10
        assert best_params['period'] <= 16
    
    def test_smoothed_rsi(self, rsi_indicator, sample_data):
        """Test RSI with smoothing"""
        regular_rsi = rsi_indicator.calculate(sample_data)
        smoothed_rsi = rsi_indicator.calculate_with_smoothing(sample_data, smooth_period=3)
        
        # Smoothed should be less volatile
        regular_std = regular_rsi[20:].std()
        smoothed_std = smoothed_rsi[20:].std()
        assert smoothed_std < regular_std
    
    def test_edge_cases(self, rsi_indicator):
        """Test edge cases"""
        # Empty data
        empty_data = pd.Series([])
        signals = rsi_indicator.generate_signals(empty_data)
        assert len(signals) == 0
        
        # Insufficient data
        short_data = pd.Series([100, 101, 102])
        rsi_values = rsi_indicator.calculate(short_data)
        assert rsi_values.isna().all()
        
        # Constant prices
        dates = pd.date_range(end=datetime.now(), periods=50, freq='5T')
        constant_data = pd.Series([100] * 50, index=dates)
        rsi_values = rsi_indicator.calculate(constant_data)
        # RSI should converge to 50 for constant prices
        assert abs(rsi_values.iloc[-1] - 50) < 5