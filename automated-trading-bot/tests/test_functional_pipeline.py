#!/usr/bin/env python3
"""Functional tests for core system components"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

# Test indicators with realistic data
def test_indicators():
    from src.indicators.momentum import MomentumIndicators
    from src.indicators.volatility import VolatilityIndicators
    
    # Create realistic market data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
    prices = 20000 + np.cumsum(np.random.randn(100) * 50)
    
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
    
    # Test momentum
    momentum = MomentumIndicators()
    m_results = momentum.calculate(data)
    print(f"✓ Momentum indicators: {len(m_results)} calculated")
    
    # Test volatility
    volatility = VolatilityIndicators()
    v_results = volatility.calculate(data)
    print(f"✓ Volatility indicators: {len(v_results)} calculated")
    
    return True

# Test bot initialization
def test_bots():
    from src.bots.momentum_rider_bot import MomentumRiderBot
    from src.bots.short_straddle_bot import ShortStraddleBot
    
    print("✓ Momentum Rider Bot imported successfully")
    print("✓ Short Straddle Bot imported successfully")
    
    return True

if __name__ == "__main__":
    try:
        test_indicators()
        test_bots()
        print("\n✅ All functional tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Functional test failed: {e}")
        sys.exit(1)
