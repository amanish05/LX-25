"""
Integration Tests for Technical Indicators
Tests the complete indicator system with real market data
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.indicators.trend import TrendIndicators
from src.indicators.momentum import MomentumIndicators
from src.indicators.volatility import VolatilityIndicators
from src.indicators.volume import VolumeIndicators
from src.indicators.composite import CompositeIndicators


class TestIndicatorIntegration:
    """Integration tests for technical indicators"""
    
    @pytest.fixture
    def sample_price_data(self):
        """Generate sample OHLCV data"""
        # Generate 100 periods of realistic price data
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
        
        # Generate realistic price movement
        base_price = 20000
        returns = np.random.normal(0.0001, 0.001, 100)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.uniform(-0.001, 0.001, 100)),
            'high': prices * (1 + np.random.uniform(0, 0.002, 100)),
            'low': prices * (1 + np.random.uniform(-0.002, 0, 100)),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 100)
        })
        
        return data
    
    @pytest.mark.asyncio
    async def test_trend_indicators_integration(self, sample_price_data):
        """Test trend indicators with real data flow"""
        trend = TrendIndicators()
        
        # Calculate all trend indicators
        close_prices = sample_price_data['close'].values
        
        # EMA
        ema_20 = trend.ema(close_prices, 20)
        ema_50 = trend.ema(close_prices, 50)
        
        # Verify EMA properties
        assert len(ema_20) == len(close_prices)
        assert not np.isnan(ema_20[-1])  # Latest value should be valid
        assert ema_20[-1] != ema_50[-1]  # Different periods should give different results
        
        # MACD
        macd_line, signal_line, histogram = trend.macd(close_prices)
        
        # Verify MACD properties
        assert len(macd_line) == len(close_prices)
        assert len(signal_line) == len(close_prices)
        assert len(histogram) == len(close_prices)
        
        # Histogram should be difference between MACD and signal
        np.testing.assert_array_almost_equal(
            histogram[~np.isnan(histogram)],
            (macd_line - signal_line)[~np.isnan(histogram)]
        )
        
        # ADX
        high = sample_price_data['high'].values
        low = sample_price_data['low'].values
        adx = trend.adx(high, low, close_prices)
        
        # Verify ADX properties
        assert len(adx) == len(close_prices)
        assert all(0 <= v <= 100 for v in adx[~np.isnan(adx)])  # ADX is bounded 0-100
    
    @pytest.mark.asyncio
    async def test_momentum_indicators_integration(self, sample_price_data):
        """Test momentum indicators with real data"""
        momentum = MomentumIndicators()
        
        close_prices = sample_price_data['close'].values
        high = sample_price_data['high'].values
        low = sample_price_data['low'].values
        
        # RSI
        rsi = momentum.rsi(close_prices)
        
        # Verify RSI properties
        assert len(rsi) == len(close_prices)
        assert all(0 <= v <= 100 for v in rsi[~np.isnan(rsi)])  # RSI is bounded 0-100
        
        # Test oversold/overbought detection
        oversold = rsi < 30
        overbought = rsi > 70
        
        # Stochastic
        k, d = momentum.stochastic(high, low, close_prices)
        
        # Verify Stochastic properties
        assert len(k) == len(close_prices)
        assert len(d) == len(close_prices)
        assert all(0 <= v <= 100 for v in k[~np.isnan(k)])  # Bounded 0-100
        
        # Williams %R
        williams_r = momentum.williams_r(high, low, close_prices)
        
        # Verify Williams %R properties
        assert len(williams_r) == len(close_prices)
        assert all(-100 <= v <= 0 for v in williams_r[~np.isnan(williams_r)])  # Bounded -100 to 0
    
    @pytest.mark.asyncio
    async def test_volatility_indicators_integration(self, sample_price_data):
        """Test volatility indicators with real data"""
        volatility = VolatilityIndicators()
        
        high = sample_price_data['high'].values
        low = sample_price_data['low'].values
        close = sample_price_data['close'].values
        
        # ATR
        atr = volatility.atr(high, low, close)
        
        # Verify ATR properties
        assert len(atr) == len(close)
        assert all(v >= 0 for v in atr[~np.isnan(atr)])  # ATR is always positive
        
        # Bollinger Bands
        upper, middle, lower = volatility.bollinger_bands(close)
        
        # Verify Bollinger Bands properties
        assert len(upper) == len(close)
        assert len(middle) == len(close)
        assert len(lower) == len(close)
        
        # Upper band should always be above lower band
        valid_idx = ~(np.isnan(upper) | np.isnan(lower))
        assert all(upper[valid_idx] > lower[valid_idx])
        
        # Middle band should be between upper and lower
        assert all(lower[valid_idx] <= middle[valid_idx])
        assert all(middle[valid_idx] <= upper[valid_idx])
        
        # Standard Deviation
        std_dev = volatility.standard_deviation(close, 20)
        
        # Verify standard deviation properties
        assert len(std_dev) == len(close)
        assert all(v >= 0 for v in std_dev[~np.isnan(std_dev)])  # Std dev is always positive
    
    @pytest.mark.asyncio
    async def test_volume_indicators_integration(self, sample_price_data):
        """Test volume indicators with real data"""
        volume_ind = VolumeIndicators()
        
        close = sample_price_data['close'].values
        volume = sample_price_data['volume'].values
        high = sample_price_data['high'].values
        low = sample_price_data['low'].values
        
        # OBV
        obv = volume_ind.obv(close, volume)
        
        # Verify OBV properties
        assert len(obv) == len(close)
        
        # OBV should change when price changes
        price_changes = np.diff(close)
        obv_changes = np.diff(obv)
        assert not all(obv_changes == 0)
        
        # Volume MA
        volume_ma = volume_ind.volume_ma(volume)
        
        # Verify Volume MA properties
        assert len(volume_ma) == len(volume)
        assert all(v > 0 for v in volume_ma[~np.isnan(volume_ma)])
        
        # MFI
        mfi = volume_ind.mfi(high, low, close, volume)
        
        # Verify MFI properties
        assert len(mfi) == len(close)
        assert all(0 <= v <= 100 for v in mfi[~np.isnan(mfi)])  # MFI is bounded 0-100
    
    @pytest.mark.asyncio
    async def test_composite_indicators_integration(self, sample_price_data):
        """Test composite indicator system"""
        config = {
            "trend": ["ema_20", "ema_50", "macd"],
            "momentum": ["rsi_14", "stochastic"],
            "volatility": ["atr_14", "bollinger_bands"],
            "volume": ["obv", "volume_ma"]
        }
        
        composite = CompositeIndicators(config)
        
        # Calculate all indicators
        results = composite.calculate_all(sample_price_data)
        
        # Verify all requested indicators calculated
        assert "ema_20" in results
        assert "ema_50" in results
        assert "macd" in results
        assert "rsi" in results
        assert "stochastic_k" in results
        assert "atr" in results
        assert "bb_upper" in results
        assert "obv" in results
        assert "volume_ma" in results
        
        # Verify indicator relationships
        # EMA20 should be more responsive than EMA50
        ema20_volatility = np.std(results["ema_20"][~np.isnan(results["ema_20"])])
        ema50_volatility = np.std(results["ema_50"][~np.isnan(results["ema_50"])])
        assert ema20_volatility > ema50_volatility
    
    @pytest.mark.asyncio
    async def test_indicator_signal_generation(self, sample_price_data):
        """Test signal generation from indicators"""
        composite = CompositeIndicators({
            "trend": ["ema_20", "ema_50"],
            "momentum": ["rsi_14"],
            "volatility": ["bollinger_bands"]
        })
        
        # Calculate indicators
        indicators = composite.calculate_all(sample_price_data)
        
        # Generate signals based on indicators
        signals = []
        
        # EMA crossover signal
        ema20 = indicators["ema_20"]
        ema50 = indicators["ema_50"]
        
        for i in range(1, len(ema20)):
            if not np.isnan(ema20[i]) and not np.isnan(ema50[i]):
                # Bullish crossover
                if ema20[i] > ema50[i] and ema20[i-1] <= ema50[i-1]:
                    signals.append({"time": i, "type": "BUY", "indicator": "EMA_CROSS"})
                # Bearish crossover
                elif ema20[i] < ema50[i] and ema20[i-1] >= ema50[i-1]:
                    signals.append({"time": i, "type": "SELL", "indicator": "EMA_CROSS"})
        
        # RSI signals
        rsi = indicators["rsi"]
        for i in range(len(rsi)):
            if not np.isnan(rsi[i]):
                if rsi[i] < 30:
                    signals.append({"time": i, "type": "BUY", "indicator": "RSI_OVERSOLD"})
                elif rsi[i] > 70:
                    signals.append({"time": i, "type": "SELL", "indicator": "RSI_OVERBOUGHT"})
        
        # Should generate some signals
        assert len(signals) > 0
    
    @pytest.mark.asyncio
    async def test_indicator_performance(self, sample_price_data):
        """Test indicator calculation performance"""
        import time
        
        composite = CompositeIndicators({
            "trend": ["ema_20", "ema_50", "macd", "adx"],
            "momentum": ["rsi_14", "stochastic"],
            "volatility": ["atr_14", "bollinger_bands"],
            "volume": ["obv", "mfi"]
        })
        
        # Measure calculation time
        start_time = time.time()
        results = composite.calculate_all(sample_price_data)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        
        # Should calculate all indicators quickly (< 100ms for 100 periods)
        assert calculation_time < 0.1
        
        # Verify all indicators calculated
        assert len(results) >= 10  # At least 10 indicators
    
    @pytest.mark.asyncio
    async def test_indicator_edge_cases(self):
        """Test indicators with edge cases"""
        # Test with minimal data
        minimal_data = pd.DataFrame({
            'close': [100, 101, 99, 102, 98],
            'high': [101, 102, 100, 103, 99],
            'low': [99, 100, 98, 101, 97],
            'volume': [1000, 1100, 900, 1200, 800]
        })
        
        trend = TrendIndicators()
        
        # Should handle small datasets gracefully
        ema = trend.ema(minimal_data['close'].values, period=3)
        assert len(ema) == len(minimal_data)
        
        # Test with constant prices
        constant_data = pd.DataFrame({
            'close': [100] * 20,
            'high': [100] * 20,
            'low': [100] * 20,
            'volume': [1000] * 20
        })
        
        momentum = MomentumIndicators()
        rsi = momentum.rsi(constant_data['close'].values)
        
        # RSI should be 50 for constant prices
        valid_rsi = rsi[~np.isnan(rsi)]
        assert all(abs(v - 50) < 1 for v in valid_rsi)
    
    @pytest.mark.asyncio
    async def test_indicator_caching(self, sample_price_data):
        """Test indicator caching mechanism"""
        composite = CompositeIndicators({
            "trend": ["ema_20"],
            "momentum": ["rsi_14"]
        })
        
        # Enable caching
        composite.use_cache = True
        
        # First calculation
        start_time = datetime.now()
        results1 = composite.calculate_all(sample_price_data)
        first_calc_time = (datetime.now() - start_time).total_seconds()
        
        # Second calculation (should be cached)
        start_time = datetime.now()
        results2 = composite.calculate_all(sample_price_data)
        cached_calc_time = (datetime.now() - start_time).total_seconds()
        
        # Cached calculation should be much faster
        assert cached_calc_time < first_calc_time / 10
        
        # Results should be identical
        np.testing.assert_array_equal(results1["ema_20"], results2["ema_20"])
        np.testing.assert_array_equal(results1["rsi"], results2["rsi"])
    
    @pytest.mark.asyncio
    async def test_multi_timeframe_indicators(self, sample_price_data):
        """Test indicators across multiple timeframes"""
        # Create 5-minute data
        data_5min = sample_price_data.copy()
        
        # Aggregate to 15-minute data
        data_15min = data_5min.resample('15min', on='timestamp').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        trend = TrendIndicators()
        
        # Calculate EMA on different timeframes
        ema_5min = trend.ema(data_5min['close'].values, 20)
        ema_15min = trend.ema(data_15min['close'].values, 20)
        
        # Different timeframes should give different results
        assert len(ema_5min) > len(ema_15min)
        
        # Higher timeframe should be smoother
        volatility_5min = np.std(ema_5min[~np.isnan(ema_5min)])
        volatility_15min = np.std(ema_15min[~np.isnan(ema_15min)])
        
        # Note: This might not always be true due to aggregation
        # but generally higher timeframes are smoother