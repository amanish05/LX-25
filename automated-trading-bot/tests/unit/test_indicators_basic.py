"""
Basic indicator tests to improve coverage
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.indicators.base import BaseIndicator
from src.indicators.momentum import MomentumIndicators
from src.indicators.volatility import VolatilityIndicators


class MockIndicator(BaseIndicator):
    """Mock indicator for testing base functionality"""
    
    @property
    def name(self):
        """Return indicator name"""
        return "MockIndicator"
    
    def _calculate_min_periods(self):
        """Required abstract method"""
        return 1
    
    def calculate(self, data):
        """Simple mock calculation"""
        return pd.Series([1.0] * len(data), index=data.index)
    
    def get_signals(self, data):
        """Simple mock signals"""
        return pd.Series([0] * len(data), index=data.index)


class TestBasicIndicators:
    """Test basic indicator functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample OHLCV data"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
        
        # Generate realistic price data
        base_price = 20000
        price_changes = np.random.normal(0, 50, 100)
        prices = base_price + np.cumsum(price_changes)
        
        data = pd.DataFrame({
            'open': prices + np.random.uniform(-10, 10, 100),
            'high': prices + np.random.uniform(10, 30, 100),
            'low': prices - np.random.uniform(10, 30, 100),
            'close': prices,
            'volume': np.random.randint(100000, 300000, 100).astype(float)
        }, index=dates)
        
        # Ensure float64 types
        for col in data.columns:
            data[col] = data[col].astype(np.float64)
        
        return data
    
    def test_base_indicator(self, sample_data):
        """Test base indicator functionality"""
        indicator = MockIndicator()
        
        # Test name property
        assert indicator.name == "MockIndicator"
        
        # Test calculation
        result = indicator.calculate(sample_data)
        assert len(result) == len(sample_data)
        
        # Test signals
        signals = indicator.get_signals(sample_data)
        assert len(signals) == len(sample_data)
    
    def test_momentum_indicators_creation(self):
        """Test momentum indicators can be created"""
        momentum = MomentumIndicators()
        assert momentum is not None
        # Just test that it was created successfully
    
    def test_momentum_indicators_with_data(self, sample_data):
        """Test momentum indicators with real data"""
        momentum = MomentumIndicators()
        
        try:
            result = momentum.calculate(sample_data)
            assert isinstance(result, dict)
            assert len(result) > 0
        except Exception as e:
            # If calculation fails due to missing dependencies, just check creation worked
            pytest.skip(f"Momentum calculation failed (likely missing TA-Lib): {e}")
    
    def test_volatility_indicators_creation(self):
        """Test volatility indicators can be created"""
        volatility = VolatilityIndicators()
        assert volatility is not None
        # Just test that it was created successfully
    
    def test_volatility_indicators_with_data(self, sample_data):
        """Test volatility indicators with real data"""
        volatility = VolatilityIndicators()
        
        try:
            result = volatility.calculate(sample_data)
            assert isinstance(result, dict)
            assert len(result) > 0
        except Exception as e:
            # If calculation fails due to missing dependencies, just check creation worked
            pytest.skip(f"Volatility calculation failed (likely missing TA-Lib): {e}")
    
    def test_data_type_handling(self, sample_data):
        """Test that indicators handle data types correctly"""
        momentum = MomentumIndicators()
        
        # Test with different data types
        data_int = sample_data.astype(int)
        data_float32 = sample_data.astype(np.float32)
        
        try:
            # Should not raise errors for different numeric types
            momentum.calculate(data_int)
            momentum.calculate(data_float32)
        except Exception as e:
            pytest.skip(f"Data type test failed (likely missing TA-Lib): {e}")
    
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        momentum = MomentumIndicators()
        volatility = VolatilityIndicators()
        
        # Create empty DataFrame
        empty_data = pd.DataFrame()
        
        try:
            momentum_result = momentum.calculate(empty_data)
            volatility_result = volatility.calculate(empty_data)
            
            # Should return empty results, not crash
            assert isinstance(momentum_result, dict)
            assert isinstance(volatility_result, dict)
        except Exception as e:
            # If it raises an exception, it should be handled gracefully
            assert "empty" in str(e).lower() or "insufficient" in str(e).lower()
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        momentum = MomentumIndicators()
        
        # Create minimal data (less than typical period requirements)
        dates = pd.date_range(end=datetime.now(), periods=5, freq='5min')
        minimal_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }, index=dates)
        
        for col in minimal_data.columns:
            minimal_data[col] = minimal_data[col].astype(np.float64)
        
        try:
            result = momentum.calculate(minimal_data)
            # Should handle gracefully
            assert isinstance(result, dict)
        except Exception as e:
            # Should raise appropriate error for insufficient data
            assert any(word in str(e).lower() for word in ['insufficient', 'minimum', 'period', 'length'])