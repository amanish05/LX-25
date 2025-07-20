"""
Standalone RSI tests that don't depend on conftest imports
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.indicators.rsi_advanced import AdvancedRSI, RSISignal, RSISignalType


class TestAdvancedRSIStandalone:
    """Standalone test cases for Advanced RSI indicator"""
    
    def test_rsi_calculation_basic(self):
        """Test basic RSI calculation functionality"""
        rsi = AdvancedRSI(period=14, overbought=70, oversold=30)
        
        # Create simple test data
        dates = pd.date_range(end=datetime.now(), periods=50, freq='5min')
        prices = list(range(100, 150))  # Simple ascending prices
        data = pd.Series(prices, index=dates)
        
        rsi_values = rsi.calculate(data)
        
        # Check output shape
        assert len(rsi_values) == len(data)
        
        # Check RSI bounds for valid values
        valid_rsi = rsi_values[~rsi_values.isna()]
        if len(valid_rsi) > 0:
            assert all(valid_rsi >= 0)
            assert all(valid_rsi <= 100)
        
        print("RSI calculation test: PASSED")
    
    def test_signal_generation_basic(self):
        """Test signal generation without complex market data"""
        rsi = AdvancedRSI(period=14, overbought=70, oversold=30)
        
        # Create simple test data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='5min')
        prices = [100] * 30  # Flat prices to avoid complex RSI behavior
        data = pd.Series(prices, index=dates)
        
        signals = rsi.generate_signals(data)
        
        # Should return a list (may be empty for flat prices)
        assert isinstance(signals, list)
        
        print("Signal generation test: PASSED")


if __name__ == "__main__":
    test = TestAdvancedRSIStandalone()
    test.test_rsi_calculation_basic()
    test.test_signal_generation_basic()
    print("All RSI standalone tests passed!")