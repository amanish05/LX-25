"""
Pattern Recognition Indicator - LuxAlgo Price Action Concepts
Identifies classic chart patterns with price action confirmation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from .base import BaseIndicator


@dataclass
class ChartPattern:
    """Represents a chart pattern"""
    pattern_type: str  # 'wedge', 'triangle', 'double_top', etc.
    start_index: int
    end_index: int
    key_points: List[Tuple[int, float]]  # Important price points
    direction: str  # 'bullish' or 'bearish'
    strength: float  # 0-100
    target_price: float
    stop_loss: float
    confluence_score: float  # Based on other indicators


class PatternRecognition(BaseIndicator):
    """
    Advanced Pattern Recognition based on LuxAlgo Price Action Concepts
    
    Features:
    - Wedge patterns (rising/falling)
    - Triangle patterns (ascending/descending/symmetrical)
    - Double tops/bottoms
    - Head and shoulders
    - Flag and pennant patterns
    - Pattern confluence scoring
    - Target and stop calculation
    """
    
    def __init__(self,
                 min_pattern_bars: int = 10,
                 max_pattern_bars: int = 50,
                 min_touches: int = 2,
                 tolerance: float = 0.02,
                 min_pattern_strength: float = 60):
        """
        Initialize Pattern Recognition indicator
        
        Args:
            min_pattern_bars: Minimum bars for pattern formation
            max_pattern_bars: Maximum bars for pattern formation
            min_touches: Minimum touches for trendline validation
            tolerance: Price tolerance for pattern recognition
            min_pattern_strength: Minimum strength to report pattern
        """
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        self.min_touches = min_touches
        self.tolerance = tolerance
        self.min_pattern_strength = min_pattern_strength
        super().__init__()
        
        self.patterns = []
    
    def _calculate_min_periods(self) -> int:
        """Calculate minimum periods required for this indicator"""
        return self.max_pattern_bars
        
    def find_peaks_troughs(self, data: pd.Series, lookback: int = 3) -> Tuple[List[int], List[int]]:
        """
        Find peaks and troughs in price data
        
        Args:
            data: Price series
            lookback: Bars to confirm peak/trough
            
        Returns:
            Tuple of (peak_indices, trough_indices)
        """
        peaks = []
        troughs = []
        
        for i in range(lookback, len(data) - lookback):
            # Peak
            if all(data.iloc[i] >= data.iloc[i-j] for j in range(1, lookback+1)) and \
               all(data.iloc[i] >= data.iloc[i+j] for j in range(1, lookback+1)):
                peaks.append(i)
            
            # Trough
            if all(data.iloc[i] <= data.iloc[i-j] for j in range(1, lookback+1)) and \
               all(data.iloc[i] <= data.iloc[i+j] for j in range(1, lookback+1)):
                troughs.append(i)
        
        return peaks, troughs
    
    def fit_trendline(self, indices: List[int], prices: List[float]) -> Tuple[float, float, float]:
        """
        Fit a trendline to given points
        
        Args:
            indices: X coordinates (bar indices)
            prices: Y coordinates (prices)
            
        Returns:
            Tuple of (slope, intercept, r_squared)
        """
        if len(indices) < 2:
            return 0, 0, 0
        
        slope, intercept, r_value, _, _ = stats.linregress(indices, prices)
        r_squared = r_value ** 2
        
        return slope, intercept, r_squared
    
    def detect_wedge(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> Optional[ChartPattern]:
        """
        Detect wedge pattern
        
        Args:
            data: OHLC DataFrame
            start_idx: Start index for pattern search
            end_idx: End index for pattern search
            
        Returns:
            ChartPattern if wedge found, None otherwise
        """
        if end_idx - start_idx < self.min_pattern_bars:
            return None
        
        # Get highs and lows
        highs = data['high'].iloc[start_idx:end_idx+1]
        lows = data['low'].iloc[start_idx:end_idx+1]
        
        # Find peaks and troughs
        peaks, _ = self.find_peaks_troughs(highs, 2)
        _, troughs = self.find_peaks_troughs(lows, 2)
        
        if len(peaks) < self.min_touches or len(troughs) < self.min_touches:
            return None
        
        # Adjust indices to data frame
        peaks = [p + start_idx for p in peaks]
        troughs = [t + start_idx for t in troughs]
        
        # Get peak and trough prices
        peak_prices = [data['high'].iloc[p] for p in peaks]
        trough_prices = [data['low'].iloc[t] for t in troughs]
        
        # Fit trendlines
        upper_slope, upper_intercept, upper_r2 = self.fit_trendline(peaks, peak_prices)
        lower_slope, lower_intercept, lower_r2 = self.fit_trendline(troughs, trough_prices)
        
        # Check if lines are converging (wedge requirement)
        if upper_r2 < 0.8 or lower_r2 < 0.8:
            return None
        
        # Determine wedge type
        if upper_slope < 0 and lower_slope < 0:
            # Both lines declining - falling wedge (bullish)
            if abs(upper_slope) > abs(lower_slope):
                pattern_type = 'falling_wedge'
                direction = 'bullish'
                # Target is breakout above upper line
                target = upper_intercept + upper_slope * (end_idx + 10)
                target *= 1.02  # Add 2% for breakout
            else:
                return None
        elif upper_slope > 0 and lower_slope > 0:
            # Both lines rising - rising wedge (bearish)
            if upper_slope < lower_slope:
                pattern_type = 'rising_wedge'
                direction = 'bearish'
                # Target is breakdown below lower line
                target = lower_intercept + lower_slope * (end_idx + 10)
                target *= 0.98  # Subtract 2% for breakdown
            else:
                return None
        else:
            return None
        
        # Calculate pattern strength
        touches = len(peaks) + len(troughs)
        convergence = abs(upper_slope - lower_slope)
        strength = min(100, (touches * 10) + (50 / (1 + convergence * 100)))
        
        if strength < self.min_pattern_strength:
            return None
        
        # Set stop loss
        if direction == 'bullish':
            stop_loss = min(trough_prices) * 0.99
        else:
            stop_loss = max(peak_prices) * 1.01
        
        return ChartPattern(
            pattern_type=pattern_type,
            start_index=start_idx,
            end_index=end_idx,
            key_points=[(p, data['high'].iloc[p]) for p in peaks] + 
                      [(t, data['low'].iloc[t]) for t in troughs],
            direction=direction,
            strength=strength,
            target_price=target,
            stop_loss=stop_loss,
            confluence_score=0  # Set later
        )
    
    def detect_triangle(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> Optional[ChartPattern]:
        """
        Detect triangle pattern
        
        Args:
            data: OHLC DataFrame
            start_idx: Start index for pattern search
            end_idx: End index for pattern search
            
        Returns:
            ChartPattern if triangle found, None otherwise
        """
        if end_idx - start_idx < self.min_pattern_bars:
            return None
        
        # Similar to wedge but with different slope requirements
        highs = data['high'].iloc[start_idx:end_idx+1]
        lows = data['low'].iloc[start_idx:end_idx+1]
        
        peaks, _ = self.find_peaks_troughs(highs, 2)
        _, troughs = self.find_peaks_troughs(lows, 2)
        
        if len(peaks) < self.min_touches or len(troughs) < self.min_touches:
            return None
        
        peaks = [p + start_idx for p in peaks]
        troughs = [t + start_idx for t in troughs]
        
        peak_prices = [data['high'].iloc[p] for p in peaks]
        trough_prices = [data['low'].iloc[t] for t in troughs]
        
        upper_slope, upper_intercept, upper_r2 = self.fit_trendline(peaks, peak_prices)
        lower_slope, lower_intercept, lower_r2 = self.fit_trendline(troughs, trough_prices)
        
        if upper_r2 < 0.7 or lower_r2 < 0.7:
            return None
        
        # Classify triangle type
        slope_threshold = 0.0001
        
        if abs(upper_slope) < slope_threshold and lower_slope > slope_threshold:
            pattern_type = 'ascending_triangle'
            direction = 'bullish'
            target = upper_intercept * 1.02
        elif upper_slope < -slope_threshold and abs(lower_slope) < slope_threshold:
            pattern_type = 'descending_triangle'
            direction = 'bearish'
            target = lower_intercept * 0.98
        elif upper_slope < -slope_threshold and lower_slope > slope_threshold:
            pattern_type = 'symmetrical_triangle'
            # Direction depends on breakout
            current_price = data['close'].iloc[end_idx]
            mid_point = (upper_intercept + upper_slope * end_idx + 
                        lower_intercept + lower_slope * end_idx) / 2
            direction = 'bullish' if current_price > mid_point else 'bearish'
            height = abs(peak_prices[0] - trough_prices[0])
            target = current_price + (height if direction == 'bullish' else -height)
        else:
            return None
        
        # Calculate strength
        touches = len(peaks) + len(troughs)
        convergence = abs(upper_slope - lower_slope)
        strength = min(100, (touches * 12) + (40 * upper_r2 * lower_r2))
        
        if strength < self.min_pattern_strength:
            return None
        
        # Stop loss
        if direction == 'bullish':
            stop_loss = min(trough_prices) * 0.99
        else:
            stop_loss = max(peak_prices) * 1.01
        
        return ChartPattern(
            pattern_type=pattern_type,
            start_index=start_idx,
            end_index=end_idx,
            key_points=[(p, data['high'].iloc[p]) for p in peaks] + 
                      [(t, data['low'].iloc[t]) for t in troughs],
            direction=direction,
            strength=strength,
            target_price=target,
            stop_loss=stop_loss,
            confluence_score=0
        )
    
    def detect_double_pattern(self, data: pd.DataFrame, lookback: int) -> Optional[ChartPattern]:
        """
        Detect double top/bottom pattern
        
        Args:
            data: OHLC DataFrame
            lookback: Bars to look back
            
        Returns:
            ChartPattern if double pattern found, None otherwise
        """
        if len(data) < lookback:
            return None
        
        # Find recent peaks and troughs
        highs = data['high'].iloc[-lookback:]
        lows = data['low'].iloc[-lookback:]
        
        peaks, _ = self.find_peaks_troughs(highs, 3)
        _, troughs = self.find_peaks_troughs(lows, 3)
        
        # Adjust indices
        peaks = [p + len(data) - lookback for p in peaks]
        troughs = [t + len(data) - lookback for t in troughs]
        
        # Check for double top
        if len(peaks) >= 2:
            # Get two most recent peaks
            p1_idx, p2_idx = peaks[-2], peaks[-1]
            p1_price = data['high'].iloc[p1_idx]
            p2_price = data['high'].iloc[p2_idx]
            
            # Check if peaks are similar height
            if abs(p1_price - p2_price) / p1_price <= self.tolerance:
                # Find trough between peaks
                trough_between = None
                trough_price = float('inf')
                for t in troughs:
                    if p1_idx < t < p2_idx:
                        if data['low'].iloc[t] < trough_price:
                            trough_between = t
                            trough_price = data['low'].iloc[t]
                
                if trough_between:
                    # Valid double top
                    neckline = trough_price
                    pattern_height = ((p1_price + p2_price) / 2) - neckline
                    target = neckline - pattern_height
                    
                    strength = 70 + (30 * (1 - abs(p1_price - p2_price) / p1_price / self.tolerance))
                    
                    return ChartPattern(
                        pattern_type='double_top',
                        start_index=p1_idx,
                        end_index=p2_idx,
                        key_points=[(p1_idx, p1_price), (trough_between, trough_price), (p2_idx, p2_price)],
                        direction='bearish',
                        strength=strength,
                        target_price=target,
                        stop_loss=max(p1_price, p2_price) * 1.01,
                        confluence_score=0
                    )
        
        # Check for double bottom
        if len(troughs) >= 2:
            # Get two most recent troughs
            t1_idx, t2_idx = troughs[-2], troughs[-1]
            t1_price = data['low'].iloc[t1_idx]
            t2_price = data['low'].iloc[t2_idx]
            
            # Check if troughs are similar depth
            if abs(t1_price - t2_price) / t1_price <= self.tolerance:
                # Find peak between troughs
                peak_between = None
                peak_price = 0
                for p in peaks:
                    if t1_idx < p < t2_idx:
                        if data['high'].iloc[p] > peak_price:
                            peak_between = p
                            peak_price = data['high'].iloc[p]
                
                if peak_between:
                    # Valid double bottom
                    neckline = peak_price
                    pattern_height = neckline - ((t1_price + t2_price) / 2)
                    target = neckline + pattern_height
                    
                    strength = 70 + (30 * (1 - abs(t1_price - t2_price) / t1_price / self.tolerance))
                    
                    return ChartPattern(
                        pattern_type='double_bottom',
                        start_index=t1_idx,
                        end_index=t2_idx,
                        key_points=[(t1_idx, t1_price), (peak_between, peak_price), (t2_idx, t2_price)],
                        direction='bullish',
                        strength=strength,
                        target_price=target,
                        stop_loss=min(t1_price, t2_price) * 0.99,
                        confluence_score=0
                    )
        
        return None
    
    def detect_head_shoulders(self, data: pd.DataFrame, lookback: int) -> Optional[ChartPattern]:
        """
        Detect head and shoulders pattern
        
        Args:
            data: OHLC DataFrame
            lookback: Bars to look back
            
        Returns:
            ChartPattern if H&S found, None otherwise
        """
        if len(data) < lookback:
            return None
        
        highs = data['high'].iloc[-lookback:]
        lows = data['low'].iloc[-lookback:]
        
        peaks, _ = self.find_peaks_troughs(highs, 3)
        _, troughs = self.find_peaks_troughs(lows, 3)
        
        # Need at least 3 peaks and 2 troughs
        if len(peaks) < 3 or len(troughs) < 2:
            return None
        
        # Adjust indices
        peaks = [p + len(data) - lookback for p in peaks]
        troughs = [t + len(data) - lookback for t in troughs]
        
        # Check regular H&S (bearish)
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]
            
            ls_price = data['high'].iloc[left_shoulder]
            h_price = data['high'].iloc[head]
            rs_price = data['high'].iloc[right_shoulder]
            
            # Head should be highest
            if h_price > ls_price and h_price > rs_price:
                # Shoulders should be similar height
                if abs(ls_price - rs_price) / ls_price <= self.tolerance:
                    # Find neckline troughs
                    left_trough = None
                    right_trough = None
                    
                    for t in troughs:
                        if left_shoulder < t < head:
                            left_trough = t
                        elif head < t < right_shoulder:
                            right_trough = t
                    
                    if left_trough and right_trough:
                        neckline = (data['low'].iloc[left_trough] + data['low'].iloc[right_trough]) / 2
                        pattern_height = h_price - neckline
                        target = neckline - pattern_height
                        
                        strength = 80 + (20 * (1 - abs(ls_price - rs_price) / ls_price / self.tolerance))
                        
                        return ChartPattern(
                            pattern_type='head_and_shoulders',
                            start_index=left_shoulder,
                            end_index=right_shoulder,
                            key_points=[
                                (left_shoulder, ls_price),
                                (left_trough, data['low'].iloc[left_trough]),
                                (head, h_price),
                                (right_trough, data['low'].iloc[right_trough]),
                                (right_shoulder, rs_price)
                            ],
                            direction='bearish',
                            strength=strength,
                            target_price=target,
                            stop_loss=h_price * 1.01,
                            confluence_score=0
                        )
        
        # Check inverse H&S (bullish)
        for i in range(len(troughs) - 2):
            left_shoulder = troughs[i]
            head = troughs[i + 1]
            right_shoulder = troughs[i + 2]
            
            ls_price = data['low'].iloc[left_shoulder]
            h_price = data['low'].iloc[head]
            rs_price = data['low'].iloc[right_shoulder]
            
            # Head should be lowest
            if h_price < ls_price and h_price < rs_price:
                # Shoulders should be similar depth
                if abs(ls_price - rs_price) / ls_price <= self.tolerance:
                    # Find neckline peaks
                    left_peak = None
                    right_peak = None
                    
                    for p in peaks:
                        if left_shoulder < p < head:
                            left_peak = p
                        elif head < p < right_shoulder:
                            right_peak = p
                    
                    if left_peak and right_peak:
                        neckline = (data['high'].iloc[left_peak] + data['high'].iloc[right_peak]) / 2
                        pattern_height = neckline - h_price
                        target = neckline + pattern_height
                        
                        strength = 80 + (20 * (1 - abs(ls_price - rs_price) / ls_price / self.tolerance))
                        
                        return ChartPattern(
                            pattern_type='inverse_head_and_shoulders',
                            start_index=left_shoulder,
                            end_index=right_shoulder,
                            key_points=[
                                (left_shoulder, ls_price),
                                (left_peak, data['high'].iloc[left_peak]),
                                (head, h_price),
                                (right_peak, data['high'].iloc[right_peak]),
                                (right_shoulder, rs_price)
                            ],
                            direction='bullish',
                            strength=strength,
                            target_price=target,
                            stop_loss=h_price * 0.99,
                            confluence_score=0
                        )
        
        return None
    
    def calculate_confluence(self, pattern: ChartPattern, data: pd.DataFrame) -> float:
        """
        Calculate confluence score based on other indicators
        
        Args:
            pattern: Chart pattern
            data: OHLC DataFrame with indicator data
            
        Returns:
            Confluence score (0-100)
        """
        score = 0
        factors = 0
        
        # Check if pattern aligns with market structure
        if 'trend' in data.columns:
            current_trend = data['trend'].iloc[pattern.end_index]
            if current_trend == pattern.direction:
                score += 30
            factors += 1
        
        # Check volume confirmation
        if 'volume' in data.columns:
            pattern_volume = data['volume'].iloc[pattern.start_index:pattern.end_index+1].mean()
            recent_volume = data['volume'].iloc[pattern.start_index-20:pattern.start_index].mean()
            if pattern_volume > recent_volume * 1.2:
                score += 20
            factors += 1
        
        # Check if near key levels
        if 'resistance_zone' in data.columns or 'support_zone' in data.columns:
            if pattern.direction == 'bullish' and 'support_zone' in data.columns:
                if data['support_zone'].iloc[pattern.end_index] > 0:
                    score += 25
            elif pattern.direction == 'bearish' and 'resistance_zone' in data.columns:
                if data['resistance_zone'].iloc[pattern.end_index] > 0:
                    score += 25
            factors += 1
        
        # Check order block alignment
        if 'bullish_ob' in data.columns or 'bearish_ob' in data.columns:
            if pattern.direction == 'bullish' and 'bullish_ob' in data.columns:
                if data['bullish_ob'].iloc[pattern.end_index] > 0:
                    score += 25
            elif pattern.direction == 'bearish' and 'bearish_ob' in data.columns:
                if data['bearish_ob'].iloc[pattern.end_index] > 0:
                    score += 25
            factors += 1
        
        return score if factors == 0 else min(100, score * 4 / factors)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pattern recognition for given data
        
        Args:
            data: OHLC DataFrame
            
        Returns:
            DataFrame with pattern analysis
        """
        if len(data) < self.max_pattern_bars:
            return pd.DataFrame()
        
        # Reset patterns
        self.patterns = []
        
        # Search for patterns in different timeframes
        for window in range(self.min_pattern_bars, min(self.max_pattern_bars, len(data)), 5):
            end_idx = len(data) - 1
            start_idx = end_idx - window
            
            # Try to detect each pattern type
            wedge = self.detect_wedge(data, start_idx, end_idx)
            if wedge:
                self.patterns.append(wedge)
            
            triangle = self.detect_triangle(data, start_idx, end_idx)
            if triangle:
                self.patterns.append(triangle)
        
        # Look for double patterns and H&S in recent data
        double = self.detect_double_pattern(data, self.max_pattern_bars)
        if double:
            self.patterns.append(double)
        
        hs = self.detect_head_shoulders(data, self.max_pattern_bars)
        if hs:
            self.patterns.append(hs)
        
        # Calculate confluence for all patterns
        for pattern in self.patterns:
            pattern.confluence_score = self.calculate_confluence(pattern, data)
        
        # Keep only strongest patterns
        self.patterns.sort(key=lambda p: p.strength + p.confluence_score, reverse=True)
        self.patterns = self.patterns[:3]  # Keep top 3
        
        # Create output DataFrame
        result = pd.DataFrame(index=data.index)
        
        # Initialize columns
        result['pattern_type'] = ''
        result['pattern_direction'] = ''
        result['pattern_strength'] = 0.0
        result['pattern_target'] = 0.0
        result['pattern_stop'] = 0.0
        result['pattern_confluence'] = 0.0
        
        # Mark patterns
        for pattern in self.patterns:
            for i in range(pattern.start_index, min(pattern.end_index + 1, len(result))):
                # Update if stronger pattern
                if result.iloc[i, result.columns.get_loc('pattern_strength')] < pattern.strength:
                    result.iloc[i, result.columns.get_loc('pattern_type')] = pattern.pattern_type
                    result.iloc[i, result.columns.get_loc('pattern_direction')] = pattern.direction
                    result.iloc[i, result.columns.get_loc('pattern_strength')] = pattern.strength
                    result.iloc[i, result.columns.get_loc('pattern_target')] = pattern.target_price
                    result.iloc[i, result.columns.get_loc('pattern_stop')] = pattern.stop_loss
                    result.iloc[i, result.columns.get_loc('pattern_confluence')] = pattern.confluence_score
        
        # Add pattern completion percentage
        result['pattern_completion'] = 0.0
        for pattern in self.patterns:
            if pattern.end_index < len(result):
                completion = ((pattern.end_index - pattern.start_index) / 
                            (self.max_pattern_bars)) * 100
                result.iloc[pattern.end_index, result.columns.get_loc('pattern_completion')] = completion
        
        return result
    
    def get_active_patterns(self) -> List[ChartPattern]:
        """
        Get currently active patterns
        
        Returns:
            List of active patterns sorted by strength
        """
        return self.patterns
    
    def get_pattern_summary(self) -> Dict:
        """
        Get summary of detected patterns
        
        Returns:
            Dictionary with pattern statistics
        """
        if not self.patterns:
            return {
                'total_patterns': 0,
                'bullish_patterns': 0,
                'bearish_patterns': 0,
                'avg_strength': 0,
                'avg_confluence': 0,
                'pattern_types': []
            }
        
        bullish = sum(1 for p in self.patterns if p.direction == 'bullish')
        bearish = sum(1 for p in self.patterns if p.direction == 'bearish')
        
        return {
            'total_patterns': len(self.patterns),
            'bullish_patterns': bullish,
            'bearish_patterns': bearish,
            'avg_strength': np.mean([p.strength for p in self.patterns]),
            'avg_confluence': np.mean([p.confluence_score for p in self.patterns]),
            'pattern_types': list(set(p.pattern_type for p in self.patterns))
        }