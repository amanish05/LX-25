"""
Unit tests for Market Structure indicator
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.indicators.market_structure import MarketStructure, MarketStructurePoint, StructureBreak


class TestMarketStructure:
    """Test cases for Market Structure indicator"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLC data with clear structure"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1h')
        
        # Create trending data with swing points
        prices = []
        base_price = 100
        
        # Uptrend with HH and HL
        for i in range(len(dates)):
            if i % 10 < 5:  # Swing up
                price = base_price + i * 0.1 + (i % 10) * 0.2
            else:  # Swing down
                price = base_price + i * 0.1 - ((i % 10) - 5) * 0.15
            prices.append(price)
        
        data = pd.DataFrame({
            'open': prices,
            'high': [p + np.random.uniform(0, 0.5) for p in prices],
            'low': [p - np.random.uniform(0, 0.5) for p in prices],
            'close': [p + np.random.uniform(-0.2, 0.2) for p in prices],
            'volume': np.random.randint(1000, 5000, len(dates))
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def indicator(self):
        """Create Market Structure indicator instance"""
        return MarketStructure(swing_lookback=5, internal_lookback=3)
    
    def test_initialization(self, indicator):
        """Test indicator initialization"""
        assert indicator.swing_lookback == 5
        assert indicator.internal_lookback == 3
        assert indicator.equal_threshold == 0.0001
        assert indicator.min_structure_points == 4
        assert indicator.current_trend is None
    
    def test_find_pivots(self, indicator, sample_data):
        """Test pivot point detection"""
        highs, lows = indicator.find_pivots(sample_data['high'], 5)
        
        # Should find some pivot points
        assert len(highs) > 0
        assert len(lows) > 0
        
        # Verify pivot highs are local maxima
        for h in highs:
            if 5 <= h < len(sample_data) - 5:
                assert all(sample_data['high'].iloc[h] >= sample_data['high'].iloc[h-i] 
                          for i in range(1, 6))
                assert all(sample_data['high'].iloc[h] >= sample_data['high'].iloc[h+i] 
                          for i in range(1, 6))
    
    def test_classify_structure_point(self, indicator):
        """Test structure point classification"""
        # Test first high
        assert indicator.classify_structure_point(100, 0, None, None, True) == 'HH'
        
        # Test higher high
        assert indicator.classify_structure_point(110, 1, 100, 95, True) == 'HH'
        
        # Test lower high
        assert indicator.classify_structure_point(95, 2, 100, 90, True) == 'LH'
        
        # Test higher low
        assert indicator.classify_structure_point(92, 3, 100, 90, False) == 'HL'
        
        # Test lower low
        assert indicator.classify_structure_point(88, 4, 100, 90, False) == 'LL'
    
    def test_detect_bos(self, indicator):
        """Test Break of Structure detection"""
        # Create bullish structure points
        points = [
            MarketStructurePoint(0, 100, 'LL', pd.Timestamp('2024-01-01')),
            MarketStructurePoint(1, 95, 'HL', pd.Timestamp('2024-01-02')),
            MarketStructurePoint(2, 105, 'HH', pd.Timestamp('2024-01-03')),
            MarketStructurePoint(3, 98, 'HL', pd.Timestamp('2024-01-04')),
            MarketStructurePoint(4, 110, 'HH', pd.Timestamp('2024-01-05'))
        ]
        
        indicator.current_trend = 'bullish'
        bos = indicator.detect_bos(points)
        
        assert bos is not None
        assert bos.type == 'BOS'
        assert bos.direction == 'bullish'
        assert bos.price == 110
    
    def test_detect_choch(self, indicator):
        """Test Change of Character detection"""
        # Create bearish to bullish reversal
        points = [
            MarketStructurePoint(0, 100, 'HH', pd.Timestamp('2024-01-01')),
            MarketStructurePoint(1, 95, 'LL', pd.Timestamp('2024-01-02')),
            MarketStructurePoint(2, 98, 'LH', pd.Timestamp('2024-01-03')),
            MarketStructurePoint(3, 92, 'LL', pd.Timestamp('2024-01-04')),
            MarketStructurePoint(4, 96, 'HL', pd.Timestamp('2024-01-05'))  # CHoCH
        ]
        
        indicator.current_trend = 'bearish'
        choch = indicator.detect_choch(points)
        
        assert choch is not None
        assert choch.type == 'CHoCH'
        assert choch.direction == 'bullish'
    
    def test_detect_equal_highs_lows(self, indicator):
        """Test equal highs/lows detection"""
        # Create data with equal levels
        data = pd.DataFrame({
            'high': [100, 98, 100.01, 97, 100, 99, 100.005],  # Equal highs around 100
            'low': [95, 90, 94, 90.01, 93, 90, 91]  # Equal lows around 90
        })
        
        indicator.swing_highs = [0, 2, 4, 6]
        indicator.swing_lows = [1, 3, 5]
        
        equal_levels = indicator.detect_equal_highs_lows(data)
        
        assert len(equal_levels['EQH']) > 0  # Should find equal highs
        assert len(equal_levels['EQL']) > 0  # Should find equal lows
    
    def test_update_trend(self, indicator):
        """Test trend update logic"""
        # Bullish structure points
        bullish_points = [
            MarketStructurePoint(0, 100, 'HL', pd.Timestamp('2024-01-01')),
            MarketStructurePoint(1, 105, 'HH', pd.Timestamp('2024-01-02')),
            MarketStructurePoint(2, 102, 'HL', pd.Timestamp('2024-01-03')),
            MarketStructurePoint(3, 108, 'HH', pd.Timestamp('2024-01-04'))
        ]
        
        indicator.update_trend(bullish_points)
        assert indicator.current_trend == 'bullish'
        
        # Bearish structure points
        bearish_points = [
            MarketStructurePoint(0, 100, 'LH', pd.Timestamp('2024-01-01')),
            MarketStructurePoint(1, 95, 'LL', pd.Timestamp('2024-01-02')),
            MarketStructurePoint(2, 98, 'LH', pd.Timestamp('2024-01-03')),
            MarketStructurePoint(3, 92, 'LL', pd.Timestamp('2024-01-04'))
        ]
        
        indicator.update_trend(bearish_points)
        assert indicator.current_trend == 'bearish'
    
    def test_calculate(self, indicator, sample_data):
        """Test full calculation"""
        result = indicator.calculate(sample_data)
        
        assert not result.empty
        assert 'trend' in result.columns
        assert 'bos_bullish' in result.columns
        assert 'bos_bearish' in result.columns
        assert 'choch_bullish' in result.columns
        assert 'choch_bearish' in result.columns
        assert 'swing_high' in result.columns
        assert 'swing_low' in result.columns
        
        # Should identify some structure
        assert result['swing_high'].sum() > 0
        assert result['swing_low'].sum() > 0
    
    def test_get_current_structure(self, indicator, sample_data):
        """Test getting current structure state"""
        indicator.calculate(sample_data)
        structure = indicator.get_current_structure()
        
        assert 'trend' in structure
        assert 'swing_high_count' in structure
        assert 'swing_low_count' in structure
        assert 'structure_breaks' in structure
        assert structure['swing_high_count'] > 0
        assert structure['swing_low_count'] > 0
    
    def test_get_key_levels(self, indicator, sample_data):
        """Test key level extraction"""
        indicator.calculate(sample_data)
        levels = indicator.get_key_levels(sample_data, lookback=20)
        
        assert 'support' in levels
        assert 'resistance' in levels
        assert len(levels['support']) > 0
        assert len(levels['resistance']) > 0
        
        # Support should be below resistance
        if levels['support'] and levels['resistance']:
            assert max(levels['support']) < min(levels['resistance'])
    
    def test_empty_data(self, indicator):
        """Test with empty data"""
        empty_data = pd.DataFrame()
        result = indicator.calculate(empty_data)
        assert result.empty
    
    def test_insufficient_data(self, indicator):
        """Test with insufficient data"""
        small_data = pd.DataFrame({
            'high': [100, 101, 102],
            'low': [99, 100, 101],
            'close': [100, 101, 102]
        })
        result = indicator.calculate(small_data)
        assert result.empty