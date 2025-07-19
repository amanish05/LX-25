"""
Unit tests for Fair Value Gaps indicator
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.indicators.fair_value_gaps import FairValueGaps, FairValueGap


class TestFairValueGaps:
    """Test cases for Fair Value Gaps indicator"""
    
    @pytest.fixture
    def sample_data_with_gaps(self):
        """Create sample OHLC data with fair value gaps"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='15min')
        
        data = []
        for i in range(len(dates)):
            if i == 50:  # Create bullish FVG
                # Candle 1
                data.append({
                    'open': 100,
                    'high': 101,
                    'low': 99.5,
                    'close': 100.5,
                    'volume': 2000
                })
            elif i == 51:  # Middle candle with momentum
                data.append({
                    'open': 100.5,
                    'high': 103,
                    'low': 100.5,
                    'close': 102.8,
                    'volume': 53000  # High volume
                })
            elif i == 52:  # Gap up - creates FVG
                data.append({
                    'open': 102,
                    'high': 104,
                    'low': 102,  # Low is above candle 1 high (101) - FVG
                    'close': 103.5,
                    'volume': 2500
                })
            elif i == 100:  # Create bearish FVG
                data.append({
                    'open': 105,
                    'high': 105.5,
                    'low': 104,
                    'close': 104.5,
                    'volume': 2000
                })
            elif i == 101:  # Middle candle with momentum
                data.append({
                    'open': 104.5,
                    'high': 104.5,
                    'low': 102,
                    'close': 102.2,
                    'volume': 4000
                })
            elif i == 102:  # Gap down - creates FVG
                data.append({
                    'open': 103,
                    'high': 103,  # High is below candle 1 low (104) - FVG
                    'low': 101,
                    'close': 101.5,
                    'volume': 2500
                })
            else:
                # Normal price action
                base = 100 + i * 0.01
                data.append({
                    'open': base,
                    'high': base + np.random.uniform(0, 0.5),
                    'low': base - np.random.uniform(0, 0.5),
                    'close': base + np.random.uniform(-0.25, 0.25),
                    'volume': np.random.randint(1000, 2000)
                })
        
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def indicator(self):
        """Create Fair Value Gaps indicator instance"""
        return FairValueGaps(
            min_gap_size=0.001,
            volume_imbalance_threshold=1.5,
            max_gaps=50,
            gap_validity_bars=100
        )
    
    def test_initialization(self, indicator):
        """Test indicator initialization"""
        assert indicator.min_gap_size == 0.001
        assert indicator.volume_imbalance_threshold == 1.5
        assert indicator.max_gaps == 50
        assert indicator.gap_validity_bars == 100
        assert len(indicator.gaps) == 0
        assert len(indicator.filled_gaps) == 0
    
    def test_identify_bullish_fvg(self, indicator, sample_data_with_gaps):
        """Test bullish FVG identification"""
        # Test at index 51 (middle of bullish FVG pattern)
        gap = indicator.identify_fvg(sample_data_with_gaps, 51)
        
        assert gap is not None
        assert gap.type == 'bullish'
        assert gap.low == 101  # Previous high
        assert gap.high == 102  # Next low
        assert gap.size == 1  # Gap size
        assert gap.volume_imbalance > 1  # High volume on middle candle
    
    def test_identify_bearish_fvg(self, indicator, sample_data_with_gaps):
        """Test bearish FVG identification"""
        # Test at index 101 (middle of bearish FVG pattern)
        gap = indicator.identify_fvg(sample_data_with_gaps, 101)
        
        assert gap is not None
        assert gap.type == 'bearish'
        assert gap.high == 104  # Previous low
        assert gap.low == 103  # Next high
        assert gap.size == 1  # Gap size
    
    def test_check_gap_fill(self, indicator):
        """Test gap fill detection"""
        # Create bullish gap
        gap = FairValueGap(
            start_index=10,
            high=102,
            low=101,
            type='bullish',
            size=1,
            volume_imbalance=2.0
        )
        
        # Create data that fills the gap
        data = pd.DataFrame({
            'high': [103] * 20 + [102.5],
            'low': [102] * 20 + [100.5]  # Trades back through gap
        })
        
        # Before fill
        assert not indicator.check_gap_fill(gap, data, 15)
        assert not gap.filled
        assert gap.fill_percentage < 100
        
        # At fill
        assert indicator.check_gap_fill(gap, data, 20)
        assert gap.filled
        assert gap.fill_index == 20
        assert gap.fill_percentage == 100.0
    
    def test_partial_gap_fill(self, indicator):
        """Test partial gap fill tracking"""
        # Create bullish gap
        gap = FairValueGap(
            start_index=10,
            high=102,
            low=100,
            type='bullish',
            size=2,
            volume_imbalance=2.0
        )
        
        # Data that partially fills gap
        data = pd.DataFrame({
            'high': [103] * 30,
            'low': [102] * 20 + [101]  # Fills 50% of gap
        })
        
        indicator.check_gap_fill(gap, data, 20)
        
        assert not gap.filled
        assert gap.fill_percentage == 50.0
    
    def test_classify_gap(self, indicator):
        """Test gap classification"""
        # Create sample data
        data = pd.DataFrame({
            'open': [100] * 30 + [100, 102, 104],  # Breakaway pattern
            'high': [100.5] * 30 + [101, 103, 105],
            'low': [99.5] * 30 + [99, 101, 103],
            'close': [100] * 30 + [100.5, 102.5, 104.5],
            'volume': [1000] * 30 + [1500, 3000, 2000]
        })
        
        # Breakaway gap (after consolidation)
        gap = FairValueGap(
            start_index=31,
            high=103,
            low=101,
            type='bullish',
            size=2,
            volume_imbalance=3.0
        )
        
        classification = indicator.classify_gap(gap, data)
        assert classification == 'breakaway'
    
    def test_calculate_gap_strength(self, indicator):
        """Test gap strength calculation"""
        gap = FairValueGap(
            start_index=10,
            high=102,
            low=100,
            type='bullish',
            size=2,  # Large gap
            volume_imbalance=2.5  # High volume
        )
        
        strength = indicator.calculate_gap_strength(gap, 'breakaway')
        
        assert strength > 50  # Should be strong
        assert strength <= 100  # Should be capped at 100
    
    def test_calculate(self, indicator, sample_data_with_gaps):
        """Test full calculation"""
        result = indicator.calculate(sample_data_with_gaps)
        
        assert not result.empty
        assert 'bullish_fvg' in result.columns
        assert 'bearish_fvg' in result.columns
        assert 'fvg_high' in result.columns
        assert 'fvg_low' in result.columns
        assert 'fvg_strength' in result.columns
        assert 'fvg_classification' in result.columns
        assert 'fvg_fill_percentage' in result.columns
        assert 'gap_density' in result.columns
        
        # Should identify gaps
        assert (result['bullish_fvg'] > 0).any()
        assert (result['bearish_fvg'] > 0).any()
    
    def test_get_unfilled_gaps(self, indicator):
        """Test getting unfilled gaps"""
        # Create gaps
        indicator.gaps = [
            FairValueGap(0, 102, 101, 'bullish', 1, False),  # Below current
            FairValueGap(5, 106, 105, 'bearish', 1, False),  # Above current
            FairValueGap(10, 98, 97, 'bullish', 1, True)   # Filled
        ]
        
        current_price = 103.5
        unfilled = indicator.get_unfilled_gaps(current_price)
        
        assert len(unfilled['above']) == 1
        assert len(unfilled['below']) == 1
        assert unfilled['above'][0].type == 'bearish'
        assert unfilled['below'][0].type == 'bullish'
    
    def test_get_gap_targets(self, indicator):
        """Test gap target calculation"""
        # Create gaps
        indicator.gaps = [
            FairValueGap(0, 102, 100, 'bullish', 2, False),
            FairValueGap(5, 108, 106, 'bearish', 2, False)
        ]
        
        current_price = 104
        targets = indicator.get_gap_targets(current_price, n=1)
        
        assert len(targets['upside']) == 1
        assert len(targets['downside']) == 1
        
        # Targets should be middle of gaps
        assert targets['upside'][0] == 107  # (108+106)/2
        assert targets['downside'][0] == 101  # (102+100)/2
    
    def test_get_gap_statistics(self, indicator):
        """Test gap statistics calculation"""
        # Create gaps
        indicator.gaps = [
            FairValueGap(0, 102, 100, 'bullish', 2, False),
            FairValueGap(5, 106, 104, 'bearish', 2, False)
        ]
        
        indicator.filled_gaps = [
            FairValueGap(10, 98, 96, 'bullish', 2, True, fill_index=15)
        ]
        
        stats = indicator.get_gap_statistics()
        
        assert stats['total_gaps'] == 3
        assert stats['filled_gaps'] == 1
        assert stats['fill_rate'] == 1/3
        assert stats['avg_gap_size'] == 2.0
        assert stats['bullish_gaps'] == 2
        assert stats['bearish_gaps'] == 1
    
    def test_gap_validity(self, indicator, sample_data_with_gaps):
        """Test gap validity period"""
        # Create old gap
        old_gap = FairValueGap(
            start_index=0,
            high=102,
            low=100,
            type='bullish',
            size=2,
            volume_imbalance=2.0
        )
        
        indicator.gaps = [old_gap]
        indicator.gap_validity_bars = 50  # Set short validity
        
        # Calculate with data longer than validity period
        result = indicator.calculate(sample_data_with_gaps)
        
        # Old gap should be removed if beyond validity
        if len(sample_data_with_gaps) > 50:
            assert len([g for g in indicator.gaps if g.start_index == 0]) == 0
    
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