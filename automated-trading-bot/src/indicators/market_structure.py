"""
Market Structure Indicator - LuxAlgo Price Action Concepts
Detects Break of Structure (BOS), Change of Character (CHoCH), and Equal Highs/Lows
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .base import BaseIndicator


@dataclass
class MarketStructurePoint:
    """Represents a market structure point"""
    index: int
    price: float
    type: str  # 'HH', 'HL', 'LH', 'LL'
    timestamp: pd.Timestamp


@dataclass
class StructureBreak:
    """Represents a structure break event"""
    index: int
    price: float
    type: str  # 'BOS', 'CHoCH', 'CHoCH+'
    direction: str  # 'bullish', 'bearish'
    timestamp: pd.Timestamp
    confirmation_level: float  # Price level that confirmed the break


class MarketStructure(BaseIndicator):
    """
    Advanced Market Structure Analysis based on LuxAlgo Price Action Concepts
    
    Features:
    - Break of Structure (BOS) detection
    - Change of Character (CHoCH) identification  
    - CHoCH+ for more confirmed reversals
    - Equal Highs/Lows (EQH/EQL) detection
    - Internal vs Swing structure differentiation
    - Multi-timeframe scanner support
    """
    
    def __init__(self, 
                 swing_lookback: int = 10,
                 internal_lookback: int = 5,
                 equal_threshold: float = 0.0001,
                 min_structure_points: int = 4):
        """
        Initialize Market Structure indicator
        
        Args:
            swing_lookback: Lookback period for swing highs/lows
            internal_lookback: Lookback period for internal structure
            equal_threshold: Percentage threshold for equal highs/lows
            min_structure_points: Minimum points needed to establish structure
        """
        self.swing_lookback = swing_lookback
        self.internal_lookback = internal_lookback
        self.equal_threshold = equal_threshold
        self.min_structure_points = min_structure_points
        super().__init__()
        
        # Structure tracking
        self.swing_highs = []
        self.swing_lows = []
        self.internal_highs = []
        self.internal_lows = []
        self.structure_breaks = []
        self.current_trend = None  # 'bullish', 'bearish', None
    
    def _calculate_min_periods(self) -> int:
        """Calculate minimum periods required for this indicator"""
        return self.swing_lookback * 2
        
    def find_pivots(self, data: pd.Series, lookback: int) -> Tuple[List[int], List[int]]:
        """
        Find pivot highs and lows in price data
        
        Args:
            data: Price series
            lookback: Lookback period for pivot detection
            
        Returns:
            Tuple of (high_indices, low_indices)
        """
        highs = []
        lows = []
        
        for i in range(lookback, len(data) - lookback):
            # Check for pivot high
            if all(data.iloc[i] >= data.iloc[i-j] for j in range(1, lookback+1)) and \
               all(data.iloc[i] >= data.iloc[i+j] for j in range(1, lookback+1)):
                highs.append(i)
            
            # Check for pivot low
            if all(data.iloc[i] <= data.iloc[i-j] for j in range(1, lookback+1)) and \
               all(data.iloc[i] <= data.iloc[i+j] for j in range(1, lookback+1)):
                lows.append(i)
        
        return highs, lows
    
    def classify_structure_point(self, price: float, index: int, 
                               prev_high: Optional[float], prev_low: Optional[float],
                               is_high: bool) -> str:
        """
        Classify a structure point as HH, HL, LH, or LL
        
        Args:
            price: Current pivot price
            index: Current pivot index
            prev_high: Previous high price
            prev_low: Previous low price
            is_high: Whether current point is a high
            
        Returns:
            Classification string ('HH', 'HL', 'LH', 'LL')
        """
        if is_high:
            if prev_high is None:
                return 'HH'  # First high
            return 'HH' if price > prev_high else 'LH'
        else:
            if prev_low is None:
                return 'LL'  # First low
            return 'HL' if price > prev_low else 'LL'
    
    def detect_bos(self, structure_points: List[MarketStructurePoint]) -> Optional[StructureBreak]:
        """
        Detect Break of Structure (trend continuation)
        
        Args:
            structure_points: List of recent structure points
            
        Returns:
            StructureBreak if BOS detected, None otherwise
        """
        if len(structure_points) < 3:
            return None
        
        latest = structure_points[-1]
        
        # Bullish BOS: New HH after established uptrend
        if latest.type == 'HH' and self.current_trend == 'bullish':
            # Find the previous high that was broken
            for i in range(len(structure_points)-2, -1, -1):
                if structure_points[i].type in ['HH', 'LH']:
                    if latest.price > structure_points[i].price:
                        return StructureBreak(
                            index=latest.index,
                            price=latest.price,
                            type='BOS',
                            direction='bullish',
                            timestamp=latest.timestamp,
                            confirmation_level=structure_points[i].price
                        )
        
        # Bearish BOS: New LL after established downtrend
        elif latest.type == 'LL' and self.current_trend == 'bearish':
            # Find the previous low that was broken
            for i in range(len(structure_points)-2, -1, -1):
                if structure_points[i].type in ['LL', 'HL']:
                    if latest.price < structure_points[i].price:
                        return StructureBreak(
                            index=latest.index,
                            price=latest.price,
                            type='BOS',
                            direction='bearish',
                            timestamp=latest.timestamp,
                            confirmation_level=structure_points[i].price
                        )
        
        return None
    
    def detect_choch(self, structure_points: List[MarketStructurePoint]) -> Optional[StructureBreak]:
        """
        Detect Change of Character (potential trend reversal)
        
        Args:
            structure_points: List of recent structure points
            
        Returns:
            StructureBreak if CHoCH detected, None otherwise
        """
        if len(structure_points) < 4:
            return None
        
        latest = structure_points[-1]
        
        # Bullish CHoCH: First HL after series of LL (downtrend reversal)
        if latest.type == 'HL' and self.current_trend == 'bearish':
            # Check if we had consistent lower lows before
            ll_count = sum(1 for p in structure_points[-4:-1] if p.type == 'LL')
            if ll_count >= 2:
                return StructureBreak(
                    index=latest.index,
                    price=latest.price,
                    type='CHoCH',
                    direction='bullish',
                    timestamp=latest.timestamp,
                    confirmation_level=latest.price
                )
        
        # Bearish CHoCH: First LH after series of HH (uptrend reversal)
        elif latest.type == 'LH' and self.current_trend == 'bullish':
            # Check if we had consistent higher highs before
            hh_count = sum(1 for p in structure_points[-4:-1] if p.type == 'HH')
            if hh_count >= 2:
                return StructureBreak(
                    index=latest.index,
                    price=latest.price,
                    type='CHoCH',
                    direction='bearish',
                    timestamp=latest.timestamp,
                    confirmation_level=latest.price
                )
        
        return None
    
    def detect_choch_plus(self, structure_points: List[MarketStructurePoint], 
                         high_data: pd.Series, low_data: pd.Series) -> Optional[StructureBreak]:
        """
        Detect CHoCH+ (more confirmed reversal with structure break)
        
        Args:
            structure_points: List of recent structure points
            high_data: High price series
            low_data: Low price series
            
        Returns:
            StructureBreak if CHoCH+ detected, None otherwise
        """
        if len(structure_points) < 5:
            return None
        
        # Look for CHoCH followed by structure break confirmation
        for i in range(len(self.structure_breaks)-1, -1, -1):
            if self.structure_breaks[i].type == 'CHoCH':
                choch = self.structure_breaks[i]
                
                # Check if price has broken key level after CHoCH
                if choch.direction == 'bullish':
                    # Look for break above previous high
                    key_level = max(p.price for p in structure_points[-5:-1] if p.type in ['HH', 'LH'])
                    if high_data.iloc[-1] > key_level:
                        return StructureBreak(
                            index=len(high_data)-1,
                            price=high_data.iloc[-1],
                            type='CHoCH+',
                            direction='bullish',
                            timestamp=high_data.index[-1],
                            confirmation_level=key_level
                        )
                
                elif choch.direction == 'bearish':
                    # Look for break below previous low
                    key_level = min(p.price for p in structure_points[-5:-1] if p.type in ['LL', 'HL'])
                    if low_data.iloc[-1] < key_level:
                        return StructureBreak(
                            index=len(low_data)-1,
                            price=low_data.iloc[-1],
                            type='CHoCH+',
                            direction='bearish',
                            timestamp=low_data.index[-1],
                            confirmation_level=key_level
                        )
        
        return None
    
    def detect_equal_highs_lows(self, data: pd.DataFrame) -> Dict[str, List[Tuple[int, float]]]:
        """
        Detect Equal Highs (EQH) and Equal Lows (EQL)
        
        Args:
            data: OHLC DataFrame
            
        Returns:
            Dictionary with 'EQH' and 'EQL' lists
        """
        equal_levels = {'EQH': [], 'EQL': []}
        
        # Find equal highs
        if len(self.swing_highs) >= 2:
            for i in range(len(self.swing_highs)-1):
                for j in range(i+1, len(self.swing_highs)):
                    idx1, idx2 = self.swing_highs[i], self.swing_highs[j]
                    price1 = data['high'].iloc[idx1]
                    price2 = data['high'].iloc[idx2]
                    
                    # Check if prices are equal within threshold
                    if abs(price1 - price2) / price1 <= self.equal_threshold:
                        equal_levels['EQH'].append((idx2, (price1 + price2) / 2))
        
        # Find equal lows
        if len(self.swing_lows) >= 2:
            for i in range(len(self.swing_lows)-1):
                for j in range(i+1, len(self.swing_lows)):
                    idx1, idx2 = self.swing_lows[i], self.swing_lows[j]
                    price1 = data['low'].iloc[idx1]
                    price2 = data['low'].iloc[idx2]
                    
                    # Check if prices are equal within threshold
                    if abs(price1 - price2) / price1 <= self.equal_threshold:
                        equal_levels['EQL'].append((idx2, (price1 + price2) / 2))
        
        return equal_levels
    
    def update_trend(self, structure_points: List[MarketStructurePoint]):
        """
        Update current trend based on structure points
        
        Args:
            structure_points: List of recent structure points
        """
        if len(structure_points) < 3:
            return
        
        # Count recent HH/HL vs LH/LL
        bullish_points = sum(1 for p in structure_points[-4:] if p.type in ['HH', 'HL'])
        bearish_points = sum(1 for p in structure_points[-4:] if p.type in ['LH', 'LL'])
        
        if bullish_points >= 3:
            self.current_trend = 'bullish'
        elif bearish_points >= 3:
            self.current_trend = 'bearish'
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market structure for given data
        
        Args:
            data: OHLC DataFrame
            
        Returns:
            DataFrame with market structure analysis
        """
        if len(data) < self.swing_lookback * 2:
            return pd.DataFrame()
        
        # Find swing pivots
        swing_highs, swing_lows = self.find_pivots(data['high'], self.swing_lookback)
        internal_highs, internal_lows = self.find_pivots(data['high'], self.internal_lookback)
        
        self.swing_highs = swing_highs
        self.swing_lows = swing_lows
        self.internal_highs = internal_highs
        self.internal_lows = internal_lows
        
        # Build structure points
        structure_points = []
        prev_high = None
        prev_low = None
        
        # Combine and sort pivots by index
        all_pivots = [(idx, 'high', data['high'].iloc[idx]) for idx in swing_highs] + \
                     [(idx, 'low', data['low'].iloc[idx]) for idx in swing_lows]
        all_pivots.sort(key=lambda x: x[0])
        
        # Classify each pivot
        for idx, pivot_type, price in all_pivots:
            is_high = pivot_type == 'high'
            classification = self.classify_structure_point(
                price, idx, prev_high, prev_low, is_high
            )
            
            structure_points.append(MarketStructurePoint(
                index=idx,
                price=price,
                type=classification,
                timestamp=data.index[idx]
            ))
            
            if is_high:
                prev_high = price
            else:
                prev_low = price
        
        # Detect structure breaks
        self.structure_breaks = []
        for i in range(self.min_structure_points, len(structure_points)):
            recent_points = structure_points[max(0, i-10):i+1]
            
            # Update trend
            self.update_trend(recent_points)
            
            # Detect BOS
            bos = self.detect_bos(recent_points)
            if bos:
                self.structure_breaks.append(bos)
            
            # Detect CHoCH
            choch = self.detect_choch(recent_points)
            if choch:
                self.structure_breaks.append(choch)
        
        # Detect CHoCH+
        if self.structure_breaks:
            choch_plus = self.detect_choch_plus(structure_points, data['high'], data['low'])
            if choch_plus:
                self.structure_breaks.append(choch_plus)
        
        # Detect equal highs/lows
        equal_levels = self.detect_equal_highs_lows(data)
        
        # Create output DataFrame
        result = pd.DataFrame(index=data.index)
        result['trend'] = self.current_trend
        
        # Mark structure breaks
        result['bos_bullish'] = 0
        result['bos_bearish'] = 0
        result['choch_bullish'] = 0
        result['choch_bearish'] = 0
        result['choch_plus_bullish'] = 0
        result['choch_plus_bearish'] = 0
        
        for sb in self.structure_breaks:
            if sb.index < len(result):
                if sb.type == 'BOS':
                    if sb.direction == 'bullish':
                        result.iloc[sb.index, result.columns.get_loc('bos_bullish')] = 1
                    else:
                        result.iloc[sb.index, result.columns.get_loc('bos_bearish')] = 1
                elif sb.type == 'CHoCH':
                    if sb.direction == 'bullish':
                        result.iloc[sb.index, result.columns.get_loc('choch_bullish')] = 1
                    else:
                        result.iloc[sb.index, result.columns.get_loc('choch_bearish')] = 1
                elif sb.type == 'CHoCH+':
                    if sb.direction == 'bullish':
                        result.iloc[sb.index, result.columns.get_loc('choch_plus_bullish')] = 1
                    else:
                        result.iloc[sb.index, result.columns.get_loc('choch_plus_bearish')] = 1
        
        # Mark equal levels
        result['eqh'] = 0.0
        result['eql'] = 0.0
        
        for idx, price in equal_levels['EQH']:
            if idx < len(result):
                result.iloc[idx, result.columns.get_loc('eqh')] = price
        
        for idx, price in equal_levels['EQL']:
            if idx < len(result):
                result.iloc[idx, result.columns.get_loc('eql')] = price
        
        # Mark swing points
        result['swing_high'] = 0.0
        result['swing_low'] = 0.0
        
        for idx in swing_highs:
            if idx < len(result):
                result.iloc[idx, result.columns.get_loc('swing_high')] = data['high'].iloc[idx]
        
        for idx in swing_lows:
            if idx < len(result):
                result.iloc[idx, result.columns.get_loc('swing_low')] = data['low'].iloc[idx]
        
        return result
    
    def get_current_structure(self) -> Dict:
        """
        Get current market structure state
        
        Returns:
            Dictionary with current structure information
        """
        return {
            'trend': self.current_trend,
            'swing_high_count': len(self.swing_highs),
            'swing_low_count': len(self.swing_lows),
            'structure_breaks': len(self.structure_breaks),
            'last_break': self.structure_breaks[-1] if self.structure_breaks else None
        }
    
    def get_key_levels(self, data: pd.DataFrame, lookback: int = 50) -> Dict[str, List[float]]:
        """
        Get key support/resistance levels from structure
        
        Args:
            data: OHLC DataFrame
            lookback: Number of bars to look back
            
        Returns:
            Dictionary with support and resistance levels
        """
        levels = {'support': [], 'resistance': []}
        
        # Recent swing highs as resistance
        for idx in self.swing_highs[-lookback:]:
            if idx < len(data):
                levels['resistance'].append(data['high'].iloc[idx])
        
        # Recent swing lows as support
        for idx in self.swing_lows[-lookback:]:
            if idx < len(data):
                levels['support'].append(data['low'].iloc[idx])
        
        # Sort and remove duplicates
        levels['support'] = sorted(list(set(levels['support'])))
        levels['resistance'] = sorted(list(set(levels['resistance'])), reverse=True)
        
        # Ensure support is always below resistance
        if levels['support'] and levels['resistance']:
            min_resistance = min(levels['resistance'])
            max_support = max(levels['support'])
            
            # Filter out overlapping levels
            if max_support >= min_resistance:
                # Remove support levels that are above the minimum resistance
                levels['support'] = [s for s in levels['support'] if s < min_resistance]
                # Remove resistance levels that are below the maximum support
                levels['resistance'] = [r for r in levels['resistance'] if r > max_support]
        
        return levels