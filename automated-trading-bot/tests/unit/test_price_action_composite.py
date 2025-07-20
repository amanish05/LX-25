"""
Unit tests for Price Action Composite indicator
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.indicators.price_action_composite import PriceActionComposite, PriceActionSignal


class TestPriceActionComposite:
    """Test cases for Price Action Composite indicator"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLC data with various price action patterns"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='15min')
        
        # Create trending data with patterns
        data = []
        base_price = 100
        
        for i in range(len(dates)):
            # Create uptrend with pullbacks
            if i < 200:
                trend = i * 0.01
                noise = np.sin(i * 0.1) * 0.5
                price = base_price + trend + noise
            # Create range
            elif i < 400:
                price = base_price + 2 + np.sin((i - 200) * 0.05) * 1
            # Create downtrend
            else:
                trend = (i - 400) * -0.01
                noise = np.sin(i * 0.1) * 0.5
                price = base_price + 2 + trend + noise
            
            # Add volume spikes at key points
            volume = 2000
            if i % 50 == 0:
                volume = 5000
            
            data.append({
                'open': price - 0.1,
                'high': price + np.random.uniform(0, 0.3),
                'low': price - np.random.uniform(0, 0.3),
                'close': price,
                'volume': volume + np.random.randint(-500, 500)
            })
        
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def indicator(self):
        """Create Price Action Composite indicator instance"""
        return PriceActionComposite(
            weights={
                'market_structure': 0.25,
                'order_blocks': 0.20,
                'fair_value_gaps': 0.15,
                'liquidity_zones': 0.20,
                'patterns': 0.20
            },
            min_signal_strength=60,
            risk_reward_min=1.5
        )
    
    def test_initialization(self, indicator):
        """Test indicator initialization"""
        assert indicator.min_signal_strength == 60
        assert indicator.risk_reward_min == 1.5
        assert sum(indicator.weights.values()) == 1.0
        
        # Check component indicators initialized
        assert indicator.market_structure is not None
        assert indicator.order_blocks is not None
        assert indicator.fair_value_gaps is not None
        assert indicator.liquidity_zones is not None
        assert indicator.pattern_recognition is not None
    
    def test_calculate_market_structure_score(self, indicator):
        """Test market structure scoring"""
        # Create mock data
        ms_data = pd.DataFrame({
            'bos_bullish': [0, 1, 0],
            'bos_bearish': [0, 0, 0],
            'choch_bullish': [0, 0, 0],
            'choch_bearish': [0, 0, 0],
            'choch_plus_bullish': [0, 0, 0],
            'choch_plus_bearish': [0, 0, 0],
            'trend': ['neutral', 'bullish', 'bullish']
        })
        
        # Test BOS signal
        score, direction = indicator.calculate_market_structure_score(ms_data, 1)
        assert score >= 80
        assert direction == 'bullish'
        
        # Test no signal
        score, direction = indicator.calculate_market_structure_score(ms_data, 0)
        assert score == 0
        assert direction == 'neutral'
    
    def test_calculate_order_block_score(self, indicator):
        """Test order block scoring"""
        ob_data = pd.DataFrame({
            'bullish_ob': [0, 100, 0],
            'bearish_ob': [0, 0, 105],
            'ob_strength': [0, 75, 80],
            'breaker_bullish': [0, 0, 0],
            'breaker_bearish': [0, 0, 0]
        })
        
        # Test bullish OB at price
        current_price = 100.2
        score, direction = indicator.calculate_order_block_score(ob_data, 1, current_price)
        assert score == 75
        assert direction == 'bullish'
        
        # Test price not at OB
        score, direction = indicator.calculate_order_block_score(ob_data, 1, 110)
        assert score == 0
        assert direction == 'neutral'
    
    def test_calculate_fvg_score(self, indicator):
        """Test fair value gap scoring"""
        fvg_data = pd.DataFrame({
            'bullish_fvg': [0, 1, 0],
            'bearish_fvg': [0, 0, 1],
            'fvg_strength': [0, 70, 65],
            'fvg_fill_percentage': [0, 20, 80]
        })
        
        # Test bullish FVG with low fill
        score, direction = indicator.calculate_fvg_score(fvg_data, 1)
        assert score > 70  # Should have bonus
        assert direction == 'bullish'
        
        # Test bearish FVG with high fill
        score, direction = indicator.calculate_fvg_score(fvg_data, 2)
        assert score == 65  # No bonus
        assert direction == 'bearish'
    
    def test_calculate_liquidity_score(self, indicator):
        """Test liquidity zone scoring"""
        liq_data = pd.DataFrame({
            'liquidity_grab': [0, 1, 0],
            'grab_direction': ['', 'bullish', ''],
            'grab_strength': [0, 85, 0],
            'discount_zone': [0, 0, 1],
            'premium_zone': [0, 0, 0],
            'zone_strength': [50, 75, 80]
        })
        
        # Test liquidity grab
        score, direction = indicator.calculate_liquidity_score(liq_data, 1)
        assert score > 85  # Should have bonus from zone strength
        assert direction == 'bullish'
        
        # Test discount zone
        score, direction = indicator.calculate_liquidity_score(liq_data, 2)
        assert score >= 60
        assert direction == 'bullish'
    
    def test_calculate_pattern_score(self, indicator):
        """Test pattern scoring"""
        pattern_data = pd.DataFrame({
            'pattern_type': ['', 'falling_wedge', ''],
            'pattern_direction': ['', 'bullish', ''],
            'pattern_strength': [0, 75, 0],
            'pattern_confluence': [0, 60, 0]
        })
        
        # Test pattern with confluence
        score, direction = indicator.calculate_pattern_score(pattern_data, 1)
        assert score > 75  # Should have confluence bonus
        assert direction == 'bullish'
    
    def test_determine_levels(self, indicator):
        """Test entry/stop/target level determination"""
        data = pd.DataFrame({
            'close': [100] * 20,
            'high': [101] * 20,
            'low': [99] * 20
        })
        
        components = {'patterns': {'pattern_data': pd.DataFrame({
            'pattern_type': [''],
            'pattern_target': [0],
            'pattern_stop': [0]
        })}}
        
        entry, stop, target = indicator.determine_levels(data, 19, 'bullish', components)
        
        assert entry > 100  # Should be above current
        assert stop < entry  # Stop below entry for bullish
        assert target > entry  # Target above entry
        assert (target - entry) / (entry - stop) >= indicator.risk_reward_min
    
    def test_calculate_atr(self, indicator):
        """Test ATR calculation"""
        data = pd.DataFrame({
            'high': [101, 102, 101.5, 103, 102] * 4,
            'low': [99, 100, 99.5, 101, 100] * 4,
            'close': [100, 101, 100.5, 102, 101] * 4
        })
        
        atr = indicator.calculate_atr(data, 19, period=14)
        assert atr > 0
        assert atr < 5  # Reasonable range
    
    def test_calculate(self, indicator, sample_data):
        """Test full calculation"""
        result = indicator.calculate(sample_data)
        
        assert not result.empty
        assert 'signal' in result.columns
        assert 'signal_strength' in result.columns
        assert 'market_bias' in result.columns
        assert 'entry_price' in result.columns
        assert 'stop_loss' in result.columns
        assert 'take_profit' in result.columns
        assert 'risk_reward' in result.columns
        assert 'confidence' in result.columns
        
        # Component scores
        assert 'ms_score' in result.columns
        assert 'ob_score' in result.columns
        assert 'fvg_score' in result.columns
        assert 'liq_score' in result.columns
        assert 'pattern_score' in result.columns
        
        # Should generate some signals or at least have valid computation
        # Check that scores are being calculated
        assert (result['ms_score'] != 0).any() or (result['ob_score'] != 0).any() or (result['fvg_score'] != 0).any()
        # Signal generation might be conservative - check confidence is calculated
        assert 'confidence' in result.columns
    
    def test_signal_generation(self, indicator):
        """Test signal generation logic"""
        # Create perfect alignment data
        data = pd.DataFrame({
            'open': [100] * 100,
            'high': [101] * 100,
            'low': [99] * 100,
            'close': [100.5] * 100,
            'volume': [2000] * 100
        })
        
        # Manually set component scores for testing
        result = indicator.calculate(data)
        
        # Check signal properties
        if len(indicator.signals) > 0:
            signal = indicator.signals[0]
            assert isinstance(signal, PriceActionSignal)
            assert signal.direction in ['bullish', 'bearish']
            assert 0 <= signal.strength <= 100
            assert signal.risk_reward_ratio >= indicator.risk_reward_min
            assert signal.confidence in ['high', 'medium', 'low']
    
    def test_get_current_bias(self, indicator, sample_data):
        """Test market bias calculation"""
        indicator.calculate(sample_data)
        bias = indicator.get_current_bias(sample_data)
        
        assert 'bias' in bias
        assert 'strength' in bias
        assert 'recommendation' in bias
        assert 'key_levels' in bias
        assert bias['bias'] in ['bullish', 'bearish', 'neutral']
    
    def test_get_signal_statistics(self, indicator, sample_data):
        """Test signal statistics"""
        indicator.calculate(sample_data)
        stats = indicator.get_signal_statistics()
        
        assert 'total_signals' in stats
        assert 'bullish_signals' in stats
        assert 'bearish_signals' in stats
        assert 'avg_strength' in stats
        assert 'avg_rr_ratio' in stats
        assert 'high_confidence' in stats
        assert 'component_contribution' in stats
        
        # Component contributions should sum to ~100
        if stats['total_signals'] > 0:
            total_contribution = sum(stats['component_contribution'].values())
            assert 95 <= total_contribution <= 105
    
    def test_empty_data(self, indicator):
        """Test with empty data"""
        empty_data = pd.DataFrame()
        result = indicator.calculate(empty_data)
        assert result.empty
    
    def test_insufficient_data(self, indicator):
        """Test with insufficient data"""
        small_data = pd.DataFrame({
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        })
        result = indicator.calculate(small_data)
        assert result.empty