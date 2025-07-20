"""
ML-Enhanced Price Action Indicator Wrappers
Integrates ML validation with traditional price action indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .price_action_ml_validator import PriceActionMLValidator, StructureBreakValidation
try:
    # Try absolute import first (for when running as module)
    from src.indicators.market_structure import MarketStructure
    from src.indicators.order_blocks import OrderBlocks
    from src.indicators.fair_value_gaps import FairValueGaps
    from src.indicators.liquidity_zones import LiquidityZones
except ImportError:
    # Fall back to relative import
    from ...indicators.market_structure import MarketStructure
    from ...indicators.order_blocks import OrderBlocks
    from ...indicators.fair_value_gaps import FairValueGaps
    from ...indicators.liquidity_zones import LiquidityZones


class MLEnhancedMarketStructure:
    """
    ML-enhanced Market Structure indicator
    Reduces false BOS/CHoCH signals using ML validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ML-enhanced market structure"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Base indicator
        self.market_structure = MarketStructure()
        
        # ML validator
        self.ml_validator = PriceActionMLValidator(
            config_path=self.config.get('ml_validator_config')
        )
        
        # Performance tracking
        self.validation_results = []
        self.false_positive_count = 0
        self.true_positive_count = 0
    
    def detect_structure_breaks(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect structure breaks with ML validation
        
        Returns:
            Dictionary with validated structure breaks and their confidence
        """
        # Get traditional structure breaks
        structure_data = self.market_structure.calculate(data)
        
        if structure_data.empty:
            return {'validated_breaks': [], 'ml_enhanced': True}
        
        # Extract recent structure breaks
        recent_breaks = []
        
        # Check for BOS signals
        if 'bos_signal' in structure_data.columns:
            bos_signals = structure_data[structure_data['bos_signal'] != 0].tail(5)
            for idx, row in bos_signals.iterrows():
                recent_breaks.append({
                    'type': 'BOS',
                    'index': idx,
                    'direction': 'bullish' if row['bos_signal'] > 0 else 'bearish',
                    'price': data.loc[idx, 'close'],
                    'confidence': 0.7,  # Default confidence
                    'swing_strength': abs(row['bos_signal'])
                })
        
        # Check for CHoCH signals
        if 'choch_signal' in structure_data.columns:
            choch_signals = structure_data[structure_data['choch_signal'] != 0].tail(5)
            for idx, row in choch_signals.iterrows():
                recent_breaks.append({
                    'type': 'CHoCH',
                    'index': idx,
                    'direction': 'bullish' if row['choch_signal'] > 0 else 'bearish',
                    'price': data.loc[idx, 'close'],
                    'confidence': 0.6,  # Default confidence
                    'pattern_clarity': abs(row['choch_signal']) / 100
                })
        
        # Validate each break with ML
        validated_breaks = []
        
        for break_event in recent_breaks:
            # Prepare break event for validation
            validation_event = {
                'type': break_event['type'],
                'confidence': break_event['confidence'],
                'price_change': (data['close'].iloc[-1] - break_event['price']) / break_event['price'],
                'levels_broken': 1,  # Simplified
                'swing_strength': break_event.get('swing_strength', 0.5),
                'pattern_clarity': break_event.get('pattern_clarity', 0.5),
                'confluence_score': 0.5,  # Could be enhanced with other indicators
                'sr_distance': 0.01  # Distance to nearest support/resistance
            }
            
            # Get ML validation
            validation = self.ml_validator.validate_structure_break(
                validation_event,
                data.tail(50)  # Recent context
            )
            
            # Store validation result
            self.validation_results.append({
                'timestamp': datetime.now(),
                'break_type': break_event['type'],
                'original_confidence': break_event['confidence'],
                'ml_confidence': validation.ml_confidence,
                'is_valid': validation.is_valid,
                'rejection_reason': validation.rejection_reason
            })
            
            # Add validated break if approved
            if validation.is_valid:
                validated_break = break_event.copy()
                validated_break['ml_confidence'] = validation.ml_confidence
                validated_break['ml_enhanced'] = True
                validated_break['validation_features'] = validation.features
                validated_breaks.append(validated_break)
                
                self.logger.info(
                    f"Validated {break_event['type']} break with ML confidence: {validation.ml_confidence:.2f}"
                )
            else:
                self.logger.debug(
                    f"Rejected {break_event['type']} break: {validation.rejection_reason}"
                )
        
        return {
            'validated_breaks': validated_breaks,
            'ml_enhanced': True,
            'total_detected': len(recent_breaks),
            'total_validated': len(validated_breaks),
            'rejection_rate': 1 - (len(validated_breaks) / len(recent_breaks)) if recent_breaks else 0
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get ML validation performance metrics"""
        if not self.validation_results:
            return {
                'total_validations': 0,
                'approval_rate': 0.5,
                'avg_ml_confidence': 0.5
            }
        
        total = len(self.validation_results)
        approved = sum(1 for v in self.validation_results if v['is_valid'])
        avg_confidence = np.mean([v['ml_confidence'] for v in self.validation_results])
        
        return {
            'total_validations': total,
            'approval_rate': approved / total if total > 0 else 0.5,
            'avg_ml_confidence': avg_confidence,
            'false_positive_rate': self.false_positive_count / total if total > 0 else 0
        }
    
    def update_break_outcome(self, break_id: str, was_successful: bool):
        """Update ML validator with actual break outcome"""
        self.ml_validator.update_performance(break_id, was_successful)
        
        if was_successful:
            self.true_positive_count += 1
        else:
            self.false_positive_count += 1


class MLEnhancedOrderBlocks:
    """
    ML-enhanced Order Blocks indicator
    Validates order block quality using ML
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ML-enhanced order blocks"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Base indicator
        self.order_blocks = OrderBlocks()
        
        # ML validator
        self.ml_validator = PriceActionMLValidator(
            config_path=self.config.get('ml_validator_config')
        )
        
        # Cache for validated blocks
        self.validated_blocks = []
    
    def detect_order_blocks(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect and validate order blocks with ML
        
        Returns:
            Dictionary with validated order blocks
        """
        # Get traditional order blocks
        ob_data = self.order_blocks.calculate(data)
        
        if ob_data.empty:
            return {'validated_blocks': [], 'ml_enhanced': True}
        
        # Extract active order blocks
        active_blocks = []
        
        # Check for bullish order blocks
        if 'bullish_ob' in ob_data.columns:
            bullish_obs = ob_data[ob_data['bullish_ob'] > 0].tail(10)
            for idx, row in bullish_obs.iterrows():
                block_index = data.index.get_loc(idx)
                active_blocks.append({
                    'type': 'bullish',
                    'index': block_index,
                    'timestamp': idx,
                    'price_low': row['bullish_ob_low'],
                    'price_high': row['bullish_ob_high'],
                    'volume': data.loc[idx, 'volume'],
                    'strength': row.get('ob_strength', 0.5),
                    'confluence_score': 0.5
                })
        
        # Check for bearish order blocks
        if 'bearish_ob' in ob_data.columns:
            bearish_obs = ob_data[ob_data['bearish_ob'] > 0].tail(10)
            for idx, row in bearish_obs.iterrows():
                block_index = data.index.get_loc(idx)
                active_blocks.append({
                    'type': 'bearish',
                    'index': block_index,
                    'timestamp': idx,
                    'price_low': row['bearish_ob_low'],
                    'price_high': row['bearish_ob_high'],
                    'volume': data.loc[idx, 'volume'],
                    'strength': row.get('ob_strength', 0.5),
                    'confluence_score': 0.5
                })
        
        # Validate each order block with ML
        validated_blocks = []
        
        for block in active_blocks:
            # Calculate additional features
            current_idx = len(data) - 1
            block['age_bars'] = current_idx - block['index']
            block['times_tested'] = self._count_block_tests(data, block)
            block['at_significant_level'] = self._is_at_significant_level(data, block)
            
            # Get ML validation
            validation = self.ml_validator.validate_order_block(block, data)
            
            # Add validated block if score is high enough
            if validation['is_valid']:
                validated_block = block.copy()
                validated_block['ml_validation'] = validation
                validated_block['ml_score'] = validation['overall_score']
                validated_block['ml_enhanced'] = True
                validated_blocks.append(validated_block)
                
                self.logger.info(
                    f"Validated {block['type']} order block with ML score: {validation['overall_score']:.2f}"
                )
        
        # Update cache
        self.validated_blocks = validated_blocks
        
        return {
            'validated_blocks': validated_blocks,
            'ml_enhanced': True,
            'total_detected': len(active_blocks),
            'total_validated': len(validated_blocks),
            'avg_ml_score': np.mean([b['ml_score'] for b in validated_blocks]) if validated_blocks else 0
        }
    
    def _count_block_tests(self, data: pd.DataFrame, block: Dict) -> int:
        """Count how many times an order block has been tested"""
        test_count = 0
        block_range = (block['price_low'], block['price_high'])
        
        # Check price interactions after block formation
        for i in range(block['index'] + 1, len(data)):
            low = data.iloc[i]['low']
            high = data.iloc[i]['high']
            
            # Check if price touched the block
            if (low <= block_range[1] and high >= block_range[0]):
                test_count += 1
        
        return test_count
    
    def _is_at_significant_level(self, data: pd.DataFrame, block: Dict) -> bool:
        """Check if order block is at a significant price level"""
        # Simple check: is it near round numbers or previous highs/lows
        block_mid = (block['price_low'] + block['price_high']) / 2
        
        # Check for round numbers
        if block_mid % 100 < 5 or block_mid % 100 > 95:
            return True
        
        # Check for historical significance
        lookback = data.iloc[max(0, block['index']-50):block['index']]
        if len(lookback) > 0:
            hist_high = lookback['high'].max()
            hist_low = lookback['low'].min()
            
            # Near historical high/low
            if abs(block_mid - hist_high) / hist_high < 0.01:
                return True
            if abs(block_mid - hist_low) / hist_low < 0.01:
                return True
        
        return False


class MLEnhancedFairValueGaps:
    """
    ML-enhanced Fair Value Gaps indicator
    Validates FVG reliability using ML
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ML-enhanced FVGs"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Base indicator
        self.fvg_indicator = FairValueGaps()
        
        # Simple ML-based scoring (could be enhanced with actual ML)
        self.validation_history = []
    
    def detect_fvgs(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect and score fair value gaps
        
        Returns:
            Dictionary with scored FVGs
        """
        # Get traditional FVGs
        fvg_data = self.fvg_indicator.calculate(data)
        
        if fvg_data.empty:
            return {'validated_gaps': [], 'ml_enhanced': True}
        
        # Extract recent FVGs
        recent_fvgs = []
        
        if 'fvg_signal' in fvg_data.columns:
            fvg_signals = fvg_data[fvg_data['fvg_signal'] != 0].tail(10)
            
            for idx, row in fvg_signals.iterrows():
                gap_info = {
                    'type': 'bullish' if row['fvg_signal'] > 0 else 'bearish',
                    'timestamp': idx,
                    'gap_start': row.get('fvg_low', 0),
                    'gap_end': row.get('fvg_high', 0),
                    'gap_size': abs(row.get('fvg_high', 0) - row.get('fvg_low', 0)),
                    'filled': row.get('fvg_filled', False)
                }
                
                # ML-based scoring (simplified)
                gap_info['ml_score'] = self._score_fvg(data, gap_info)
                gap_info['is_valid'] = gap_info['ml_score'] > 0.6
                
                if gap_info['is_valid']:
                    recent_fvgs.append(gap_info)
        
        return {
            'validated_gaps': recent_fvgs,
            'ml_enhanced': True,
            'total_gaps': len(recent_fvgs),
            'avg_ml_score': np.mean([g['ml_score'] for g in recent_fvgs]) if recent_fvgs else 0
        }
    
    def _score_fvg(self, data: pd.DataFrame, gap: Dict) -> float:
        """Score FVG quality using ML-like heuristics"""
        score = 0.5  # Base score
        
        # Size relative to ATR
        atr = self._calculate_atr(data, 14)
        relative_size = gap['gap_size'] / atr if atr > 0 else 0
        
        if 0.5 < relative_size < 3.0:  # Optimal size range
            score += 0.2
        elif relative_size > 3.0:  # Too large
            score -= 0.1
        
        # Volume at gap creation
        gap_idx = data.index.get_loc(gap['timestamp'])
        if gap_idx > 0:
            vol_ratio = data.iloc[gap_idx]['volume'] / data['volume'].rolling(20).mean().iloc[gap_idx]
            if vol_ratio > 1.5:  # High volume
                score += 0.2
        
        # Market context (trending vs ranging)
        if self._is_trending_market(data):
            score += 0.1
        
        return max(0, min(1, score))
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(period).mean().iloc[-1]
    
    def _is_trending_market(self, data: pd.DataFrame) -> bool:
        """Simple trend detection"""
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        
        if len(data) < 50:
            return False
        
        # Check if price is consistently above/below moving averages
        recent_closes = data['close'].tail(10)
        recent_sma_20 = sma_20.tail(10)
        
        above_ma = (recent_closes > recent_sma_20).sum() > 7
        below_ma = (recent_closes < recent_sma_20).sum() > 7
        
        return above_ma or below_ma


class MLEnhancedLiquidityZones:
    """
    ML-enhanced Liquidity Zones indicator
    Identifies high-probability liquidity zones
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ML-enhanced liquidity zones"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Base indicator
        self.liquidity_zones = LiquidityZones()
        
        # Zone validation tracking
        self.zone_performance = {}
    
    def detect_liquidity_zones(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect and rank liquidity zones
        
        Returns:
            Dictionary with ranked liquidity zones
        """
        # Get traditional liquidity zones
        lz_data = self.liquidity_zones.calculate(data)
        
        if lz_data.empty:
            return {'ranked_zones': [], 'ml_enhanced': True}
        
        # Extract active zones
        active_zones = []
        
        # Check for liquidity highs
        if 'liquidity_high' in lz_data.columns:
            liq_highs = lz_data[lz_data['liquidity_high'] > 0].tail(5)
            for idx, row in liq_highs.iterrows():
                zone = {
                    'type': 'resistance',
                    'timestamp': idx,
                    'price': row['liquidity_high'],
                    'strength': row.get('liquidity_strength', 0.5),
                    'touches': self._count_zone_touches(data, row['liquidity_high'], is_resistance=True)
                }
                zone['ml_rank'] = self._rank_liquidity_zone(data, zone)
                active_zones.append(zone)
        
        # Check for liquidity lows
        if 'liquidity_low' in lz_data.columns:
            liq_lows = lz_data[lz_data['liquidity_low'] > 0].tail(5)
            for idx, row in liq_lows.iterrows():
                zone = {
                    'type': 'support',
                    'timestamp': idx,
                    'price': row['liquidity_low'],
                    'strength': row.get('liquidity_strength', 0.5),
                    'touches': self._count_zone_touches(data, row['liquidity_low'], is_resistance=False)
                }
                zone['ml_rank'] = self._rank_liquidity_zone(data, zone)
                active_zones.append(zone)
        
        # Sort by ML rank
        ranked_zones = sorted(active_zones, key=lambda x: x['ml_rank'], reverse=True)
        
        return {
            'ranked_zones': ranked_zones,
            'ml_enhanced': True,
            'total_zones': len(ranked_zones),
            'high_probability_zones': [z for z in ranked_zones if z['ml_rank'] > 0.7]
        }
    
    def _count_zone_touches(self, data: pd.DataFrame, price_level: float, 
                           is_resistance: bool, tolerance: float = 0.002) -> int:
        """Count how many times a zone has been touched"""
        touches = 0
        price_range = price_level * tolerance
        
        for _, row in data.iterrows():
            if is_resistance:
                # Check if high touched resistance
                if abs(row['high'] - price_level) <= price_range:
                    touches += 1
            else:
                # Check if low touched support
                if abs(row['low'] - price_level) <= price_range:
                    touches += 1
        
        return touches
    
    def _rank_liquidity_zone(self, data: pd.DataFrame, zone: Dict) -> float:
        """Rank liquidity zone importance using ML-like scoring"""
        rank = 0.5  # Base rank
        
        # More touches = higher rank
        if zone['touches'] >= 3:
            rank += 0.2
        elif zone['touches'] >= 2:
            rank += 0.1
        
        # Recent zones get higher rank
        zone_age = len(data) - data.index.get_loc(zone['timestamp'])
        if zone_age < 20:
            rank += 0.1
        elif zone_age > 100:
            rank -= 0.1
        
        # Zone strength
        rank += zone['strength'] * 0.2
        
        # Historical performance (if available)
        zone_key = f"{zone['type']}_{zone['price']:.2f}"
        if zone_key in self.zone_performance:
            performance = self.zone_performance[zone_key]
            rank += performance * 0.2
        
        return max(0, min(1, rank))
    
    def update_zone_performance(self, zone_price: float, zone_type: str, 
                               was_respected: bool):
        """Update zone performance tracking"""
        zone_key = f"{zone_type}_{zone_price:.2f}"
        
        if zone_key not in self.zone_performance:
            self.zone_performance[zone_key] = 0.5
        
        # Update with exponential moving average
        alpha = 0.3
        success = 1.0 if was_respected else 0.0
        self.zone_performance[zone_key] = (
            alpha * success + (1 - alpha) * self.zone_performance[zone_key]
        )


# Consolidated ML Price Action System
class MLEnhancedPriceActionSystem:
    """
    Complete ML-enhanced price action system
    Combines all ML-enhanced price action indicators
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize complete ML price action system"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize all ML-enhanced indicators
        self.market_structure = MLEnhancedMarketStructure(config)
        self.order_blocks = MLEnhancedOrderBlocks(config)
        self.fair_value_gaps = MLEnhancedFairValueGaps(config)
        self.liquidity_zones = MLEnhancedLiquidityZones(config)
        
        # System-wide ML validator
        self.ml_validator = PriceActionMLValidator(
            config_path=config.get('ml_validator_config') if config else None
        )
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Complete ML-enhanced price action analysis
        
        Returns:
            Comprehensive price action analysis with ML validation
        """
        analysis = {
            'timestamp': datetime.now(),
            'ml_enhanced': True,
            'components': {}
        }
        
        # Get ML-validated structure breaks
        structure_analysis = self.market_structure.detect_structure_breaks(data)
        analysis['components']['structure_breaks'] = structure_analysis
        
        # Get ML-validated order blocks
        ob_analysis = self.order_blocks.detect_order_blocks(data)
        analysis['components']['order_blocks'] = ob_analysis
        
        # Get ML-scored FVGs
        fvg_analysis = self.fair_value_gaps.detect_fvgs(data)
        analysis['components']['fair_value_gaps'] = fvg_analysis
        
        # Get ML-ranked liquidity zones
        lz_analysis = self.liquidity_zones.detect_liquidity_zones(data)
        analysis['components']['liquidity_zones'] = lz_analysis
        
        # Generate composite signal
        analysis['composite_signal'] = self._generate_composite_signal(analysis['components'])
        
        # Add performance metrics
        analysis['ml_metrics'] = {
            'structure_performance': self.market_structure.get_performance_metrics(),
            'total_ml_validations': len(self.market_structure.validation_results),
            'avg_confidence': self._calculate_avg_confidence(analysis['components'])
        }
        
        return analysis
    
    def _generate_composite_signal(self, components: Dict) -> Dict[str, Any]:
        """Generate composite signal from all ML-enhanced components"""
        signal_strength = 0
        signal_count = 0
        bullish_factors = 0
        bearish_factors = 0
        
        # Structure breaks
        for break_event in components['structure_breaks']['validated_breaks']:
            if break_event['direction'] == 'bullish':
                bullish_factors += break_event['ml_confidence']
            else:
                bearish_factors += break_event['ml_confidence']
            signal_count += 1
        
        # Order blocks
        for block in components['order_blocks']['validated_blocks']:
            if block['type'] == 'bullish':
                bullish_factors += block['ml_score'] * 0.8
            else:
                bearish_factors += block['ml_score'] * 0.8
            signal_count += 1
        
        # Fair value gaps
        for gap in components['fair_value_gaps']['validated_gaps']:
            if gap['type'] == 'bullish':
                bullish_factors += gap['ml_score'] * 0.6
            else:
                bearish_factors += gap['ml_score'] * 0.6
            signal_count += 1
        
        # Liquidity zones (support/resistance)
        high_prob_zones = components['liquidity_zones']['high_probability_zones']
        for zone in high_prob_zones[:2]:  # Top 2 zones
            if zone['type'] == 'support':
                bullish_factors += zone['ml_rank'] * 0.5
            else:
                bearish_factors += zone['ml_rank'] * 0.5
        
        # Calculate final signal
        if signal_count == 0:
            return {
                'direction': 'neutral',
                'strength': 0,
                'confidence': 0,
                'factors': {'bullish': 0, 'bearish': 0}
            }
        
        net_signal = bullish_factors - bearish_factors
        signal_strength = abs(net_signal) / signal_count
        
        return {
            'direction': 'bullish' if net_signal > 0.5 else 'bearish' if net_signal < -0.5 else 'neutral',
            'strength': min(signal_strength, 1.0),
            'confidence': min(signal_count / 10, 1.0),  # More signals = higher confidence
            'factors': {
                'bullish': bullish_factors,
                'bearish': bearish_factors,
                'signal_count': signal_count
            }
        }
    
    def _calculate_avg_confidence(self, components: Dict) -> float:
        """Calculate average ML confidence across all components"""
        confidences = []
        
        # Structure breaks
        for break_event in components['structure_breaks']['validated_breaks']:
            confidences.append(break_event['ml_confidence'])
        
        # Order blocks
        for block in components['order_blocks']['validated_blocks']:
            confidences.append(block['ml_score'])
        
        # FVGs
        for gap in components['fair_value_gaps']['validated_gaps']:
            confidences.append(gap['ml_score'])
        
        # Liquidity zones
        for zone in components['liquidity_zones']['ranked_zones']:
            confidences.append(zone['ml_rank'])
        
        return np.mean(confidences) if confidences else 0.5
    
    def train_validators(self, historical_data: List[Dict[str, Any]]):
        """Train all ML validators with historical data"""
        if historical_data:
            self.ml_validator.train_validator(historical_data)
            self.logger.info("ML validators trained with historical data")
    
    def save_models(self, filepath: str):
        """Save all ML models"""
        self.ml_validator.save_model(filepath)
        self.logger.info(f"ML models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load all ML models"""
        self.ml_validator.load_model(filepath)
        self.logger.info(f"ML models loaded from {filepath}")