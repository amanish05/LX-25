"""
Unit tests for Order Blocks indicator
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.indicators.order_blocks import OrderBlocks, OrderBlock


class TestOrderBlocks:
    """Test cases for Order Blocks indicator"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLC data with order block patterns"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1h')
        
        # Create data with clear order blocks
        data = []
        for i in range(len(dates)):
            if i == 50:  # Bearish order block before bullish move
                data.append({
                    'open': 100,
                    'high': 101,
                    'low': 98,
                    'close': 98.5,  # Bearish candle
                    'volume': 5000  # High volume
                })
            elif i == 51:  # Strong bullish move
                data.append({
                    'open': 98.5,
                    'high': 104,
                    'low': 98,
                    'close': 103,
                    'volume': 3000
                })
            elif i == 100:  # Bullish order block before bearish move
                data.append({
                    'open': 105,
                    'high': 107,
                    'low': 104.5,
                    'close': 106.5,  # Bullish candle
                    'volume': 6000  # High volume
                })
            elif i == 101:  # Strong bearish move
                data.append({
                    'open': 106.5,
                    'high': 107,
                    'low': 102,
                    'close': 102.5,
                    'volume': 4000
                })
            else:
                # Normal price action
                base = 100 + i * 0.01
                data.append({
                    'open': base,
                    'high': base + np.random.uniform(0, 1),
                    'low': base - np.random.uniform(0, 1),
                    'close': base + np.random.uniform(-0.5, 0.5),
                    'volume': np.random.randint(1000, 2000)
                })
        
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def indicator(self):
        """Create Order Blocks indicator instance"""
        return OrderBlocks(
            lookback=10,
            volume_threshold=1.5,
            imbalance_threshold=0.7,
            min_block_size=0.001
        )
    
    def test_initialization(self, indicator):
        """Test indicator initialization"""
        assert indicator.lookback == 10
        assert indicator.volume_threshold == 1.5
        assert indicator.imbalance_threshold == 0.7
        assert indicator.min_block_size == 0.001
        assert indicator.max_blocks == 20
        assert len(indicator.order_blocks) == 0
    
    def test_calculate_imbalance(self, indicator, sample_data):
        """Test order flow imbalance calculation"""
        # Test bullish imbalance
        bullish_idx = 51
        imbalance = indicator.calculate_imbalance(sample_data, bullish_idx)
        assert imbalance > 0.5  # Should show strong imbalance
        
        # Test normal candle
        normal_idx = 10
        normal_imbalance = indicator.calculate_imbalance(sample_data, normal_idx)
        assert normal_imbalance < imbalance  # Should be lower
    
    def test_identify_order_block(self, indicator, sample_data):
        """Test order block identification"""
        # Test bullish order block
        block = indicator.identify_order_block(sample_data, 50)
        assert block is not None
        assert block.type == 'bullish'
        assert block.volume > 4000
        assert block.strength > 50
        
        # Test bearish order block
        block = indicator.identify_order_block(sample_data, 100)
        assert block is not None
        assert block.type == 'bearish'
        assert block.volume > 5000
    
    def test_check_mitigation(self, indicator):
        """Test order block mitigation"""
        # Create bullish order block
        block = OrderBlock(
            start_index=10,
            end_index=10,
            top=101,
            bottom=99,
            type='bullish',
            volume=5000,
            strength=80
        )
        
        # Create data that mitigates the block
        data = pd.DataFrame({
            'high': [100] * 20 + [102],
            'low': [98] * 20 + [97]  # Breaks below block
        })
        
        # Before mitigation
        assert not indicator.check_mitigation(block, data, 15)
        assert not block.mitigated
        
        # At mitigation
        assert indicator.check_mitigation(block, data, 20)
        assert block.mitigated
        assert block.mitigation_index == 20
    
    def test_check_breaker_block(self, indicator):
        """Test breaker block formation"""
        # Create mitigated bullish order block
        block = OrderBlock(
            start_index=10,
            end_index=10,
            top=101,
            bottom=99,
            type='bullish',
            volume=5000,
            strength=80,
            mitigated=True,
            mitigation_index=20
        )
        
        # Create data showing continuation after mitigation
        data = pd.DataFrame({
            'close': [100] * 25 + [95]  # Continues lower
        })
        
        assert indicator.check_breaker_block(block, data)
        assert block.breaker_block
    
    def test_rank_order_blocks(self, indicator):
        """Test order block ranking"""
        # Create multiple blocks
        blocks = [
            OrderBlock(0, 0, 101, 99, 'bullish', 5000, 60),
            OrderBlock(5, 5, 105, 103, 'bearish', 6000, 80),
            OrderBlock(10, 10, 98, 96, 'bullish', 4000, 70, mitigated=True)
        ]
        
        indicator.order_blocks = blocks
        ranked = indicator.rank_order_blocks()
        
        # Should only include non-mitigated blocks
        assert len(ranked) == 2
        # Should be sorted by strength
        assert ranked[0].strength == 80
        assert ranked[1].strength == 60
    
    def test_calculate(self, indicator, sample_data):
        """Test full calculation"""
        result = indicator.calculate(sample_data)
        
        assert not result.empty
        assert 'bullish_ob' in result.columns
        assert 'bearish_ob' in result.columns
        assert 'ob_strength' in result.columns
        assert 'breaker_bullish' in result.columns
        assert 'breaker_bearish' in result.columns
        assert 'ob_volume_ratio' in result.columns
        assert 'order_flow_imbalance' in result.columns
        
        # Should identify some order blocks
        assert (result['bullish_ob'] > 0).any()
        assert (result['bearish_ob'] > 0).any()
        assert (result['ob_strength'] > 0).any()
    
    def test_get_active_blocks(self, indicator):
        """Test getting active blocks relative to price"""
        # Create blocks
        indicator.order_blocks = [
            OrderBlock(0, 0, 101, 99, 'bullish', 5000, 60),  # Below current
            OrderBlock(5, 5, 105, 103, 'bearish', 6000, 80),  # Above current
            OrderBlock(10, 10, 98, 96, 'bullish', 4000, 70, mitigated=True)  # Mitigated
        ]
        
        current_price = 102
        active = indicator.get_active_blocks(current_price)
        
        assert len(active['above']) == 1
        assert len(active['below']) == 1
        assert active['above'][0].type == 'bearish'
        assert active['below'][0].type == 'bullish'
    
    def test_get_nearest_blocks(self, indicator):
        """Test getting nearest blocks"""
        # Create blocks
        indicator.order_blocks = [
            OrderBlock(0, 0, 101, 99, 'bullish', 5000, 60),
            OrderBlock(5, 5, 105, 103, 'bearish', 6000, 80),
            OrderBlock(10, 10, 98, 96, 'bullish', 4000, 70),
            OrderBlock(15, 15, 108, 106, 'bearish', 5500, 75)
        ]
        
        current_price = 102
        nearest = indicator.get_nearest_blocks(current_price, n=2)
        
        assert len(nearest['support']) <= 2
        assert len(nearest['resistance']) <= 2
        
        # Check format
        if nearest['resistance']:
            assert len(nearest['resistance'][0]) == 3  # (price, strength, volume)
    
    def test_volume_validation(self, indicator, sample_data):
        """Test volume threshold validation"""
        # Modify indicator with high volume threshold
        indicator.volume_threshold = 5.0  # Very high threshold
        
        result = indicator.calculate(sample_data)
        
        # Should find fewer or no blocks due to high threshold
        blocks_found = (result['bullish_ob'] > 0).sum() + (result['bearish_ob'] > 0).sum()
        
        # Reset and test with low threshold
        indicator.volume_threshold = 1.1
        indicator.order_blocks = []
        result2 = indicator.calculate(sample_data)
        
        blocks_found2 = (result2['bullish_ob'] > 0).sum() + (result2['bearish_ob'] > 0).sum()
        assert blocks_found2 >= blocks_found
    
    def test_empty_data(self, indicator):
        """Test with empty data"""
        empty_data = pd.DataFrame()
        result = indicator.calculate(empty_data)
        assert result.empty
    
    def test_insufficient_data(self, indicator):
        """Test with insufficient data"""
        small_data = pd.DataFrame({
            'open': [100, 101],
            'high': [101, 102],
            'low': [99, 100],
            'close': [100, 101],
            'volume': [1000, 1000]
        })
        result = indicator.calculate(small_data)
        assert result.empty