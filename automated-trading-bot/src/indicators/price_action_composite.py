"""
Price Action Composite Indicator - LuxAlgo Price Action Concepts
Combines all price action signals into a unified scoring system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .base import BaseIndicator
from .market_structure import MarketStructure
from .order_blocks import OrderBlocks
from .fair_value_gaps import FairValueGaps
from .liquidity_zones import LiquidityZones
from .pattern_recognition import PatternRecognition


@dataclass
class PriceActionSignal:
    """Represents a composite price action signal"""
    timestamp: pd.Timestamp
    direction: str  # 'bullish' or 'bearish'
    strength: float  # 0-100
    components: Dict[str, float]  # Individual component scores
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    confidence: str  # 'high', 'medium', 'low'


class PriceActionComposite(BaseIndicator):
    """
    Comprehensive Price Action Analysis combining all LuxAlgo concepts
    
    Features:
    - Combines market structure, order blocks, FVGs, liquidity, patterns
    - Generates composite buy/sell signals
    - Calculates entry, stop loss, and take profit levels
    - Provides signal confidence scoring
    - Market bias determination
    """
    
    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 min_signal_strength: float = 60,
                 risk_reward_min: float = 1.5):
        """
        Initialize Price Action Composite indicator
        
        Args:
            weights: Component weights for scoring
            min_signal_strength: Minimum strength for signal generation
            risk_reward_min: Minimum risk/reward ratio
        """
        # Default weights if not provided
        self.weights = weights or {
            'market_structure': 0.25,
            'order_blocks': 0.20,
            'fair_value_gaps': 0.15,
            'liquidity_zones': 0.20,
            'patterns': 0.20
        }
        
        self.min_signal_strength = min_signal_strength
        self.risk_reward_min = risk_reward_min
        
        super().__init__()
        
        # Initialize component indicators
        self.market_structure = MarketStructure()
        self.order_blocks = OrderBlocks()
        self.fair_value_gaps = FairValueGaps()
        self.liquidity_zones = LiquidityZones()
        self.pattern_recognition = PatternRecognition()
        
        self.signals = []
    
    def _calculate_min_periods(self) -> int:
        """Calculate minimum periods required for this indicator"""
        return 50  # Need sufficient data for all components
        
    def calculate_market_structure_score(self, ms_data: pd.DataFrame, idx: int) -> Tuple[float, str]:
        """
        Calculate market structure component score
        
        Args:
            ms_data: Market structure data
            idx: Current index
            
        Returns:
            Tuple of (score, direction)
        """
        if idx >= len(ms_data):
            return 0, 'neutral'
        
        score = 0
        direction = 'neutral'
        
        # Check for BOS/CHoCH signals
        if ms_data['bos_bullish'].iloc[idx] > 0:
            score += 80
            direction = 'bullish'
        elif ms_data['bos_bearish'].iloc[idx] > 0:
            score += 80
            direction = 'bearish'
        elif ms_data['choch_bullish'].iloc[idx] > 0:
            score += 60
            direction = 'bullish'
        elif ms_data['choch_bearish'].iloc[idx] > 0:
            score += 60
            direction = 'bearish'
        elif ms_data['choch_plus_bullish'].iloc[idx] > 0:
            score += 100
            direction = 'bullish'
        elif ms_data['choch_plus_bearish'].iloc[idx] > 0:
            score += 100
            direction = 'bearish'
        
        # Trend alignment bonus
        if 'trend' in ms_data.columns:
            trend = ms_data['trend'].iloc[idx]
            if trend == direction:
                score = min(100, score * 1.2)
        
        return score, direction
    
    def calculate_order_block_score(self, ob_data: pd.DataFrame, idx: int, 
                                  current_price: float) -> Tuple[float, str]:
        """
        Calculate order block component score
        
        Args:
            ob_data: Order block data
            idx: Current index
            current_price: Current price
            
        Returns:
            Tuple of (score, direction)
        """
        if idx >= len(ob_data):
            return 0, 'neutral'
        
        score = 0
        direction = 'neutral'
        
        # Check if price is at order block
        if ob_data['bullish_ob'].iloc[idx] > 0:
            ob_price = ob_data['bullish_ob'].iloc[idx]
            if abs(current_price - ob_price) / ob_price < 0.005:  # Within 0.5%
                score = ob_data['ob_strength'].iloc[idx]
                direction = 'bullish'
        elif ob_data['bearish_ob'].iloc[idx] > 0:
            ob_price = ob_data['bearish_ob'].iloc[idx]
            if abs(current_price - ob_price) / ob_price < 0.005:
                score = ob_data['ob_strength'].iloc[idx]
                direction = 'bearish'
        
        # Breaker block signals
        if ob_data['breaker_bullish'].iloc[idx] > 0:
            score = max(score, 70)
            direction = 'bullish'
        elif ob_data['breaker_bearish'].iloc[idx] > 0:
            score = max(score, 70)
            direction = 'bearish'
        
        return score, direction
    
    def calculate_fvg_score(self, fvg_data: pd.DataFrame, idx: int) -> Tuple[float, str]:
        """
        Calculate fair value gap component score
        
        Args:
            fvg_data: Fair value gap data
            idx: Current index
            
        Returns:
            Tuple of (score, direction)
        """
        if idx >= len(fvg_data):
            return 0, 'neutral'
        
        score = 0
        direction = 'neutral'
        
        # Check for FVG signals
        if fvg_data['bullish_fvg'].iloc[idx] > 0:
            score = fvg_data['fvg_strength'].iloc[idx]
            direction = 'bullish'
            
            # Bonus for unfilled gaps
            if fvg_data['fvg_fill_percentage'].iloc[idx] < 50:
                score = min(100, score * 1.2)
                
        elif fvg_data['bearish_fvg'].iloc[idx] > 0:
            score = fvg_data['fvg_strength'].iloc[idx]
            direction = 'bearish'
            
            # Bonus for unfilled gaps
            if fvg_data['fvg_fill_percentage'].iloc[idx] < 50:
                score = min(100, score * 1.2)
        
        return score, direction
    
    def calculate_liquidity_score(self, liq_data: pd.DataFrame, idx: int) -> Tuple[float, str]:
        """
        Calculate liquidity zone component score
        
        Args:
            liq_data: Liquidity zone data
            idx: Current index
            
        Returns:
            Tuple of (score, direction)
        """
        if idx >= len(liq_data):
            return 0, 'neutral'
        
        score = 0
        direction = 'neutral'
        
        # Check for liquidity grab signals
        if liq_data['liquidity_grab'].iloc[idx] > 0:
            grab_direction = liq_data['grab_direction'].iloc[idx]
            score = liq_data['grab_strength'].iloc[idx]
            direction = grab_direction
        
        # Premium/discount zone signals
        elif liq_data['discount_zone'].iloc[idx] > 0:
            score = 60
            direction = 'bullish'
        elif liq_data['premium_zone'].iloc[idx] > 0:
            score = 60
            direction = 'bearish'
        
        # Zone strength bonus
        if liq_data['zone_strength'].iloc[idx] > 70:
            score = min(100, score * 1.1)
        
        return score, direction
    
    def calculate_pattern_score(self, pattern_data: pd.DataFrame, idx: int) -> Tuple[float, str]:
        """
        Calculate pattern component score
        
        Args:
            pattern_data: Pattern data
            idx: Current index
            
        Returns:
            Tuple of (score, direction)
        """
        if idx >= len(pattern_data):
            return 0, 'neutral'
        
        score = 0
        direction = 'neutral'
        
        if pattern_data['pattern_type'].iloc[idx]:
            score = pattern_data['pattern_strength'].iloc[idx]
            direction = pattern_data['pattern_direction'].iloc[idx]
            
            # Confluence bonus
            confluence = pattern_data['pattern_confluence'].iloc[idx]
            if confluence > 50:
                score = min(100, score * (1 + confluence / 200))
        
        return score, direction
    
    def determine_levels(self, data: pd.DataFrame, idx: int, direction: str,
                        components: Dict) -> Tuple[float, float, float]:
        """
        Determine entry, stop loss, and take profit levels
        
        Args:
            data: OHLC DataFrame
            idx: Current index
            direction: Signal direction
            components: Component data dictionaries
            
        Returns:
            Tuple of (entry, stop_loss, take_profit)
        """
        current_price = data['close'].iloc[idx]
        atr = self.calculate_atr(data, idx, 14)
        
        # Default levels
        if direction == 'bullish':
            entry = current_price * 1.001  # Small buffer above current
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 4)
        else:
            entry = current_price * 0.999  # Small buffer below current
            stop_loss = current_price + (atr * 2)
            take_profit = current_price - (atr * 4)
        
        # Adjust based on patterns
        if 'patterns' in components and idx < len(components['patterns']['pattern_data']['pattern_type']):
            if components['patterns']['pattern_data']['pattern_type'].iloc[idx]:
                pattern_target = components['patterns']['pattern_data']['pattern_target'].iloc[idx]
                pattern_stop = components['patterns']['pattern_data']['pattern_stop'].iloc[idx]
                
                if pattern_target > 0:
                    take_profit = pattern_target
                if pattern_stop > 0:
                    stop_loss = pattern_stop
        
        # Adjust based on liquidity zones
        if 'liquidity' in components:
            zones = self.liquidity_zones.get_nearest_zones(current_price)
            
            if direction == 'bullish' and zones['below']:
                # Use nearest support as stop
                stop_loss = zones['below'][0][0] * 0.995
            elif direction == 'bearish' and zones['above']:
                # Use nearest resistance as stop
                stop_loss = zones['above'][0][0] * 1.005
        
        # Ensure minimum risk/reward
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        
        if reward / risk < self.risk_reward_min:
            if direction == 'bullish':
                take_profit = entry + (risk * self.risk_reward_min)
            else:
                take_profit = entry - (risk * self.risk_reward_min)
        
        return entry, stop_loss, take_profit
    
    def calculate_atr(self, data: pd.DataFrame, idx: int, period: int = 14) -> float:
        """
        Calculate Average True Range
        
        Args:
            data: OHLC DataFrame
            idx: Current index
            period: ATR period
            
        Returns:
            ATR value
        """
        if idx < period:
            return 0
        
        tr_values = []
        for i in range(idx - period + 1, idx + 1):
            if i > 0:
                tr = max(
                    data['high'].iloc[i] - data['low'].iloc[i],
                    abs(data['high'].iloc[i] - data['close'].iloc[i-1]),
                    abs(data['low'].iloc[i] - data['close'].iloc[i-1])
                )
            else:
                tr = data['high'].iloc[i] - data['low'].iloc[i]
            tr_values.append(tr)
        
        return np.mean(tr_values)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite price action signals
        
        Args:
            data: OHLC DataFrame with volume
            
        Returns:
            DataFrame with composite analysis
        """
        if len(data) < 50:
            return pd.DataFrame()
        
        # Calculate all component indicators
        ms_data = self.market_structure.calculate(data)
        ob_data = self.order_blocks.calculate(data)
        fvg_data = self.fair_value_gaps.calculate(data)
        liq_data = self.liquidity_zones.calculate(data)
        pattern_data = self.pattern_recognition.calculate(data)
        
        # Create output DataFrame
        result = pd.DataFrame(index=data.index)
        result['signal'] = 0  # -1 sell, 0 neutral, 1 buy
        result['signal_strength'] = 0.0
        result['market_bias'] = 'neutral'
        result['entry_price'] = 0.0
        result['stop_loss'] = 0.0
        result['take_profit'] = 0.0
        result['risk_reward'] = 0.0
        result['confidence'] = 'low'
        
        # Component scores
        result['ms_score'] = 0.0
        result['ob_score'] = 0.0
        result['fvg_score'] = 0.0
        result['liq_score'] = 0.0
        result['pattern_score'] = 0.0
        
        # Reset signals
        self.signals = []
        
        # Analyze each bar
        for i in range(50, len(data)):
            current_price = data['close'].iloc[i]
            
            # Get component scores
            ms_score, ms_dir = self.calculate_market_structure_score(ms_data, i)
            ob_score, ob_dir = self.calculate_order_block_score(ob_data, i, current_price)
            fvg_score, fvg_dir = self.calculate_fvg_score(fvg_data, i)
            liq_score, liq_dir = self.calculate_liquidity_score(liq_data, i)
            pattern_score, pattern_dir = self.calculate_pattern_score(pattern_data, i)
            
            # Store component scores
            result.iloc[i, result.columns.get_loc('ms_score')] = ms_score
            result.iloc[i, result.columns.get_loc('ob_score')] = ob_score
            result.iloc[i, result.columns.get_loc('fvg_score')] = fvg_score
            result.iloc[i, result.columns.get_loc('liq_score')] = liq_score
            result.iloc[i, result.columns.get_loc('pattern_score')] = pattern_score
            
            # Calculate direction consensus
            directions = [d for d in [ms_dir, ob_dir, fvg_dir, liq_dir, pattern_dir] 
                         if d != 'neutral']
            
            if len(directions) >= 2:
                bullish_count = sum(1 for d in directions if d == 'bullish')
                bearish_count = sum(1 for d in directions if d == 'bearish')
                
                if bullish_count > bearish_count * 1.5:
                    consensus_direction = 'bullish'
                elif bearish_count > bullish_count * 1.5:
                    consensus_direction = 'bearish'
                else:
                    consensus_direction = 'neutral'
            else:
                consensus_direction = 'neutral'
            
            # Calculate composite score
            components = {
                'market_structure': (ms_score, ms_dir),
                'order_blocks': (ob_score, ob_dir),
                'fair_value_gaps': (fvg_score, fvg_dir),
                'liquidity_zones': (liq_score, liq_dir),
                'patterns': (pattern_score, pattern_dir)
            }
            
            total_score = 0
            aligned_score = 0
            
            for comp_name, (score, direction) in components.items():
                weighted_score = score * self.weights[comp_name]
                total_score += weighted_score
                
                # Bonus for alignment with consensus
                if direction == consensus_direction and consensus_direction != 'neutral':
                    aligned_score += weighted_score
            
            # Alignment bonus
            if aligned_score / (total_score + 0.001) > 0.7:
                total_score = min(100, total_score * 1.2)
            
            # Determine market bias
            if total_score > 40:
                result.iloc[i, result.columns.get_loc('market_bias')] = consensus_direction
            
            # Generate signal if strong enough
            if total_score >= self.min_signal_strength and consensus_direction != 'neutral':
                # Determine levels
                component_data = {
                    'patterns': {'pattern_data': pattern_data},
                    'liquidity': {'liq_data': liq_data}
                }
                
                entry, stop, target = self.determine_levels(
                    data, i, consensus_direction, component_data
                )
                
                risk = abs(entry - stop)
                reward = abs(target - entry)
                rr_ratio = reward / risk if risk > 0 else 0
                
                if rr_ratio >= self.risk_reward_min:
                    # Determine confidence
                    if total_score >= 80 and aligned_score / total_score > 0.8:
                        confidence = 'high'
                    elif total_score >= 65:
                        confidence = 'medium'
                    else:
                        confidence = 'low'
                    
                    # Create signal
                    signal = PriceActionSignal(
                        timestamp=data.index[i],
                        direction=consensus_direction,
                        strength=total_score,
                        components={name: score for name, (score, _) in components.items()},
                        entry_price=entry,
                        stop_loss=stop,
                        take_profit=target,
                        risk_reward_ratio=rr_ratio,
                        confidence=confidence
                    )
                    
                    self.signals.append(signal)
                    
                    # Update result
                    result.iloc[i, result.columns.get_loc('signal')] = 1 if consensus_direction == 'bullish' else -1
                    result.iloc[i, result.columns.get_loc('signal_strength')] = total_score
                    result.iloc[i, result.columns.get_loc('entry_price')] = entry
                    result.iloc[i, result.columns.get_loc('stop_loss')] = stop
                    result.iloc[i, result.columns.get_loc('take_profit')] = target
                    result.iloc[i, result.columns.get_loc('risk_reward')] = rr_ratio
                    result.iloc[i, result.columns.get_loc('confidence')] = confidence
        
        return result
    
    def get_current_bias(self, data: pd.DataFrame) -> Dict:
        """
        Get current market bias and recommendations
        
        Args:
            data: OHLC DataFrame
            
        Returns:
            Dictionary with bias and recommendations
        """
        if not self.signals:
            self.calculate(data)
        
        recent_signals = [s for s in self.signals[-10:]]
        
        if not recent_signals:
            return {
                'bias': 'neutral',
                'strength': 0,
                'recommendation': 'Wait for clearer signals',
                'key_levels': {}
            }
        
        # Calculate recent bias
        bullish_signals = sum(1 for s in recent_signals if s.direction == 'bullish')
        bearish_signals = sum(1 for s in recent_signals if s.direction == 'bearish')
        avg_strength = np.mean([s.strength for s in recent_signals])
        
        if bullish_signals > bearish_signals * 1.5:
            bias = 'bullish'
            recommendation = 'Look for long entries at support levels'
        elif bearish_signals > bullish_signals * 1.5:
            bias = 'bearish'
            recommendation = 'Look for short entries at resistance levels'
        else:
            bias = 'neutral'
            recommendation = 'Market is ranging, trade the boundaries'
        
        # Get key levels
        current_price = data['close'].iloc[-1]
        key_levels = self.liquidity_zones.get_nearest_zones(current_price)
        
        return {
            'bias': bias,
            'strength': avg_strength,
            'recommendation': recommendation,
            'key_levels': key_levels,
            'recent_signals': len(recent_signals),
            'signal_quality': 'high' if avg_strength > 75 else 'medium' if avg_strength > 60 else 'low'
        }
    
    def get_signal_statistics(self) -> Dict:
        """
        Get statistics about generated signals
        
        Returns:
            Dictionary with signal statistics
        """
        if not self.signals:
            return {
                'total_signals': 0,
                'bullish_signals': 0,
                'bearish_signals': 0,
                'avg_strength': 0,
                'avg_rr_ratio': 0,
                'high_confidence': 0,
                'component_contribution': {}
            }
        
        bullish = sum(1 for s in self.signals if s.direction == 'bullish')
        bearish = sum(1 for s in self.signals if s.direction == 'bearish')
        high_conf = sum(1 for s in self.signals if s.confidence == 'high')
        
        # Calculate component contributions
        component_totals = {comp: 0 for comp in self.weights.keys()}
        for signal in self.signals:
            for comp, score in signal.components.items():
                component_totals[comp] += score
        
        # Normalize contributions
        total_contribution = sum(component_totals.values())
        if total_contribution > 0:
            component_contribution = {
                comp: (total / total_contribution) * 100 
                for comp, total in component_totals.items()
            }
        else:
            component_contribution = component_totals
        
        return {
            'total_signals': len(self.signals),
            'bullish_signals': bullish,
            'bearish_signals': bearish,
            'avg_strength': np.mean([s.strength for s in self.signals]),
            'avg_rr_ratio': np.mean([s.risk_reward_ratio for s in self.signals]),
            'high_confidence': high_conf,
            'high_confidence_rate': high_conf / len(self.signals) if self.signals else 0,
            'component_contribution': component_contribution
        }