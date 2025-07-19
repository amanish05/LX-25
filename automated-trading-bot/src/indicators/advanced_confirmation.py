"""
Advanced Confirmation System
Integrates multiple indicators to reduce false positives
Based on TradingView indicators: Trendlines, Predictive Ranges, and IFVG
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import talib


class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class ConfirmationSignal:
    """Container for confirmed signals"""
    timestamp: pd.Timestamp
    symbol: str
    signal_type: str  # 'BUY' or 'SELL'
    primary_indicator: str
    confirmations: List[str]
    confluence_score: float
    strength: SignalStrength
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward: float
    false_positive_probability: float


class AdvancedConfirmationSystem:
    """
    Multi-layer confirmation system to validate trading signals
    Reduces false positives by requiring multiple confirmations
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize advanced confirmation system"""
        self.config = config or {}
        
        # Confirmation weights
        self.weights = {
            'trendline_break': 0.25,
            'predictive_range': 0.20,
            'fair_value_gap': 0.15,
            'reversal_signal': 0.20,
            'volume_confirmation': 0.10,
            'momentum_alignment': 0.10
        }
        
        # Minimum requirements
        self.min_confirmations = self.config.get('min_confirmations', 3)
        self.min_confluence_score = self.config.get('min_confluence_score', 0.65)
        self.max_false_positive_rate = self.config.get('max_false_positive_rate', 0.30)
        
        # Initialize sub-systems
        self.trendline_analyzer = TrendlineAnalyzer()
        self.range_predictor = PredictiveRangeCalculator()
        self.fvg_detector = FairValueGapDetector()
        
        # Performance tracking
        self.signal_history = []
        self.false_positive_tracker = {}
        
    def validate_signal(self, 
                       primary_signal: Dict,
                       market_data: pd.DataFrame,
                       option_chain: Optional[Dict] = None) -> Optional[ConfirmationSignal]:
        """
        Validate a primary signal with multiple confirmations
        
        Args:
            primary_signal: Initial signal from primary indicator
            market_data: Historical price data
            option_chain: Current option chain data
            
        Returns:
            ConfirmationSignal if validated, None otherwise
        """
        confirmations = []
        confirmation_scores = {}
        
        symbol = primary_signal['symbol']
        signal_type = primary_signal['type']  # 'BUY' or 'SELL'
        
        # 1. Check trendline break confirmation
        trendline_result = self.trendline_analyzer.check_break(
            market_data, signal_type
        )
        if trendline_result['confirmed']:
            confirmations.append('trendline_break')
            confirmation_scores['trendline_break'] = trendline_result['strength']
        
        # 2. Check predictive range position
        range_result = self.range_predictor.check_position(
            market_data, signal_type
        )
        if range_result['favorable']:
            confirmations.append('predictive_range')
            confirmation_scores['predictive_range'] = range_result['score']
        
        # 3. Check fair value gaps
        fvg_result = self.fvg_detector.check_gaps(
            market_data, signal_type
        )
        if fvg_result['supportive']:
            confirmations.append('fair_value_gap')
            confirmation_scores['fair_value_gap'] = fvg_result['strength']
        
        # 4. Check reversal signals (from existing indicator)
        if 'reversal_confirmed' in primary_signal and primary_signal['reversal_confirmed']:
            confirmations.append('reversal_signal')
            confirmation_scores['reversal_signal'] = primary_signal.get('reversal_strength', 0.7)
        
        # 5. Volume confirmation
        volume_result = self._check_volume_confirmation(market_data)
        if volume_result['confirmed']:
            confirmations.append('volume_confirmation')
            confirmation_scores['volume_confirmation'] = volume_result['strength']
        
        # 6. Momentum alignment
        momentum_result = self._check_momentum_alignment(market_data, signal_type)
        if momentum_result['aligned']:
            confirmations.append('momentum_alignment')
            confirmation_scores['momentum_alignment'] = momentum_result['score']
        
        # Calculate confluence score
        confluence_score = self._calculate_confluence_score(
            confirmations, confirmation_scores
        )
        
        # Check minimum requirements
        if len(confirmations) < self.min_confirmations:
            return None
        
        if confluence_score < self.min_confluence_score:
            return None
        
        # Calculate false positive probability
        false_positive_prob = self._estimate_false_positive_probability(
            confirmations, market_data
        )
        
        if false_positive_prob > self.max_false_positive_rate:
            return None
        
        # Calculate entry, stop, and target
        entry_details = self._calculate_entry_details(
            market_data, signal_type, confirmations
        )
        
        # Determine signal strength
        strength = self._determine_signal_strength(
            len(confirmations), confluence_score
        )
        
        # Create confirmed signal
        confirmed_signal = ConfirmationSignal(
            timestamp=market_data.index[-1],
            symbol=symbol,
            signal_type=signal_type,
            primary_indicator=primary_signal.get('indicator', 'unknown'),
            confirmations=confirmations,
            confluence_score=confluence_score,
            strength=strength,
            entry_price=entry_details['entry'],
            stop_loss=entry_details['stop_loss'],
            target_price=entry_details['target'],
            risk_reward=entry_details['risk_reward'],
            false_positive_probability=false_positive_prob
        )
        
        # Track signal for performance analysis
        self.signal_history.append(confirmed_signal)
        
        return confirmed_signal
    
    def _check_volume_confirmation(self, data: pd.DataFrame) -> Dict:
        """Check if volume confirms the signal"""
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].iloc[-20:].mean()
        
        # Volume spike detection
        volume_spike = current_volume > avg_volume * 1.5
        
        # Volume trend
        volume_trend = data['volume'].iloc[-5:].mean() > data['volume'].iloc[-10:-5].mean()
        
        confirmed = volume_spike or volume_trend
        strength = min(current_volume / avg_volume, 2.0) / 2.0 if confirmed else 0
        
        return {
            'confirmed': confirmed,
            'strength': strength,
            'volume_ratio': current_volume / avg_volume
        }
    
    def _check_momentum_alignment(self, data: pd.DataFrame, signal_type: str) -> Dict:
        """Check if momentum indicators align with signal"""
        close = data['close'].values
        
        # Calculate momentum indicators
        rsi = talib.RSI(close, timeperiod=14)
        macd, signal, _ = talib.MACD(close)
        
        # Get current values
        current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
        current_macd = macd[-1] if not np.isnan(macd[-1]) else 0
        
        # Check alignment
        if signal_type == 'BUY':
            rsi_aligned = 30 < current_rsi < 70  # Not overbought
            macd_aligned = current_macd > signal[-1] if not np.isnan(signal[-1]) else False
        else:
            rsi_aligned = 30 < current_rsi < 70  # Not oversold
            macd_aligned = current_macd < signal[-1] if not np.isnan(signal[-1]) else False
        
        aligned = rsi_aligned and macd_aligned
        score = 0.0
        
        if aligned:
            # Calculate alignment strength
            if signal_type == 'BUY':
                score = (current_rsi - 30) / 40 * 0.5 + (0.5 if macd_aligned else 0)
            else:
                score = (70 - current_rsi) / 40 * 0.5 + (0.5 if macd_aligned else 0)
        
        return {
            'aligned': aligned,
            'score': min(score, 1.0),
            'rsi': current_rsi,
            'macd_signal': 'bullish' if current_macd > signal[-1] else 'bearish'
        }
    
    def _calculate_confluence_score(self, confirmations: List[str], 
                                  scores: Dict[str, float]) -> float:
        """Calculate weighted confluence score"""
        total_score = 0.0
        total_weight = 0.0
        
        for confirmation in confirmations:
            weight = self.weights.get(confirmation, 0.1)
            score = scores.get(confirmation, 0.5)
            total_score += weight * score
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            return total_score / total_weight
        
        return 0.0
    
    def _estimate_false_positive_probability(self, confirmations: List[str],
                                           data: pd.DataFrame) -> float:
        """Estimate probability of false positive based on historical performance"""
        base_probability = 0.4  # Base false positive rate
        
        # Reduce probability based on number of confirmations
        reduction_per_confirmation = 0.08
        probability = base_probability - (len(confirmations) * reduction_per_confirmation)
        
        # Adjust based on market conditions
        volatility = data['close'].pct_change().std() * np.sqrt(252)
        if volatility > 0.30:  # High volatility
            probability += 0.1
        elif volatility < 0.15:  # Low volatility
            probability -= 0.05
        
        # Adjust based on historical performance
        if len(self.signal_history) > 20:
            recent_signals = self.signal_history[-20:]
            # This would be calculated based on actual outcomes in production
            historical_accuracy = 0.65  # Placeholder
            probability = probability * (1 - historical_accuracy)
        
        return max(0.0, min(1.0, probability))
    
    def _calculate_entry_details(self, data: pd.DataFrame, signal_type: str,
                               confirmations: List[str]) -> Dict[str, float]:
        """Calculate entry, stop loss, and target prices"""
        current_price = data['close'].iloc[-1]
        atr = talib.ATR(data['high'].values, data['low'].values, 
                       data['close'].values, timeperiod=14)[-1]
        
        # Adjust risk based on confirmations
        risk_multiplier = 2.5 - (len(confirmations) * 0.1)  # More confirmations = tighter stop
        reward_multiplier = 1.5 + (len(confirmations) * 0.2)  # More confirmations = higher target
        
        if signal_type == 'BUY':
            stop_loss = current_price - (atr * risk_multiplier)
            target = current_price + (atr * risk_multiplier * reward_multiplier)
        else:
            stop_loss = current_price + (atr * risk_multiplier)
            target = current_price - (atr * risk_multiplier * reward_multiplier)
        
        risk = abs(current_price - stop_loss)
        reward = abs(target - current_price)
        risk_reward = reward / risk if risk > 0 else 0
        
        return {
            'entry': current_price,
            'stop_loss': stop_loss,
            'target': target,
            'risk_reward': risk_reward
        }
    
    def _determine_signal_strength(self, num_confirmations: int, 
                                 confluence_score: float) -> SignalStrength:
        """Determine overall signal strength"""
        if num_confirmations >= 5 and confluence_score >= 0.80:
            return SignalStrength.VERY_STRONG
        elif num_confirmations >= 4 and confluence_score >= 0.70:
            return SignalStrength.STRONG
        elif num_confirmations >= 3 and confluence_score >= 0.60:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics of the confirmation system"""
        if not self.signal_history:
            return {
                'total_signals': 0,
                'avg_confirmations': 0,
                'avg_confluence_score': 0,
                'signal_distribution': {}
            }
        
        total_signals = len(self.signal_history)
        avg_confirmations = np.mean([len(s.confirmations) for s in self.signal_history])
        avg_confluence = np.mean([s.confluence_score for s in self.signal_history])
        
        # Signal strength distribution
        strength_dist = {}
        for strength in SignalStrength:
            count = sum(1 for s in self.signal_history if s.strength == strength)
            strength_dist[strength.name] = count / total_signals * 100
        
        return {
            'total_signals': total_signals,
            'avg_confirmations': avg_confirmations,
            'avg_confluence_score': avg_confluence,
            'signal_distribution': strength_dist,
            'avg_risk_reward': np.mean([s.risk_reward for s in self.signal_history])
        }


class TrendlineAnalyzer:
    """Analyzes trendlines and detects breaks"""
    
    def check_break(self, data: pd.DataFrame, signal_type: str) -> Dict:
        """Check if a trendline break confirms the signal"""
        # Implementation based on TradingView logic
        lookback = 20
        
        # Find pivot points
        highs = data['high'].values
        lows = data['low'].values
        close = data['close'].values
        
        # Simple trendline calculation using linear regression
        x = np.arange(lookback)
        
        if signal_type == 'BUY':
            # Check for bullish trendline break (upward break of resistance)
            recent_highs = highs[-lookback:]
            slope, intercept = np.polyfit(x, recent_highs, 1)
            trendline_value = slope * (lookback - 1) + intercept
            
            # Check if current close breaks above trendline
            break_confirmed = close[-1] > trendline_value and close[-2] <= trendline_value
            strength = min((close[-1] - trendline_value) / trendline_value * 100, 1.0)
        else:
            # Check for bearish trendline break (downward break of support)
            recent_lows = lows[-lookback:]
            slope, intercept = np.polyfit(x, recent_lows, 1)
            trendline_value = slope * (lookback - 1) + intercept
            
            # Check if current close breaks below trendline
            break_confirmed = close[-1] < trendline_value and close[-2] >= trendline_value
            strength = min((trendline_value - close[-1]) / trendline_value * 100, 1.0)
        
        return {
            'confirmed': break_confirmed,
            'strength': strength if break_confirmed else 0,
            'trendline_value': trendline_value
        }


class PredictiveRangeCalculator:
    """Calculates predictive ranges based on ATR"""
    
    def check_position(self, data: pd.DataFrame, signal_type: str) -> Dict:
        """Check if price position relative to predictive ranges is favorable"""
        # Calculate ATR-based ranges
        atr = talib.ATR(data['high'].values, data['low'].values, 
                       data['close'].values, timeperiod=14)
        
        current_price = data['close'].iloc[-1]
        current_atr = atr[-1]
        
        # Calculate range levels
        central_level = data['close'].iloc[-14:].mean()
        upper_range = central_level + (current_atr * 2)
        lower_range = central_level - (current_atr * 2)
        
        # Check position
        if signal_type == 'BUY':
            # Favorable if price is near lower range (good risk-reward for longs)
            distance_from_lower = abs(current_price - lower_range)
            favorable = distance_from_lower < current_atr * 0.5
            score = 1 - (distance_from_lower / (upper_range - lower_range))
        else:
            # Favorable if price is near upper range (good risk-reward for shorts)
            distance_from_upper = abs(upper_range - current_price)
            favorable = distance_from_upper < current_atr * 0.5
            score = 1 - (distance_from_upper / (upper_range - lower_range))
        
        return {
            'favorable': favorable,
            'score': max(0, min(1, score)),
            'upper_range': upper_range,
            'lower_range': lower_range,
            'central_level': central_level
        }


class FairValueGapDetector:
    """Detects and analyzes Fair Value Gaps"""
    
    def check_gaps(self, data: pd.DataFrame, signal_type: str) -> Dict:
        """Check if fair value gaps support the signal"""
        # Detect FVGs in recent price action
        gaps = self._detect_fvgs(data)
        
        if not gaps:
            return {'supportive': False, 'strength': 0}
        
        current_price = data['close'].iloc[-1]
        
        # Check if gaps support the signal
        supportive = False
        strength = 0
        
        for gap in gaps[-3:]:  # Check last 3 gaps
            if signal_type == 'BUY':
                # Bullish signal supported by inverted bearish FVG below
                if gap['type'] == 'bearish' and gap['inverted'] and gap['lower'] < current_price:
                    supportive = True
                    strength = max(strength, gap['strength'])
            else:
                # Bearish signal supported by inverted bullish FVG above
                if gap['type'] == 'bullish' and gap['inverted'] and gap['upper'] > current_price:
                    supportive = True
                    strength = max(strength, gap['strength'])
        
        return {
            'supportive': supportive,
            'strength': strength,
            'gaps_detected': len(gaps)
        }
    
    def _detect_fvgs(self, data: pd.DataFrame) -> List[Dict]:
        """Detect Fair Value Gaps in price data"""
        gaps = []
        
        for i in range(2, len(data) - 1):
            # Bullish FVG: Gap up
            if data['low'].iloc[i] > data['high'].iloc[i-2]:
                gap_size = data['low'].iloc[i] - data['high'].iloc[i-2]
                gap_percent = gap_size / data['close'].iloc[i-1] * 100
                
                # Check if inverted (price came back to test)
                inverted = any(data['high'].iloc[i+1:].values < data['low'].iloc[i])
                
                gaps.append({
                    'index': i,
                    'type': 'bullish',
                    'upper': data['low'].iloc[i],
                    'lower': data['high'].iloc[i-2],
                    'size': gap_size,
                    'strength': min(gap_percent / 2, 1.0),
                    'inverted': inverted
                })
            
            # Bearish FVG: Gap down
            elif data['high'].iloc[i] < data['low'].iloc[i-2]:
                gap_size = data['low'].iloc[i-2] - data['high'].iloc[i]
                gap_percent = gap_size / data['close'].iloc[i-1] * 100
                
                # Check if inverted (price came back to test)
                inverted = any(data['low'].iloc[i+1:].values > data['high'].iloc[i])
                
                gaps.append({
                    'index': i,
                    'type': 'bearish',
                    'upper': data['low'].iloc[i-2],
                    'lower': data['high'].iloc[i],
                    'size': gap_size,
                    'strength': min(gap_percent / 2, 1.0),
                    'inverted': inverted
                })
        
        return gaps