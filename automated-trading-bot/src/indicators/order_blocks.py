"""
Order Blocks Indicator - LuxAlgo Price Action Concepts
Identifies institutional order blocks with volumetric analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .base import BaseIndicator


@dataclass
class OrderBlock:
    """Represents an order block"""
    start_index: int
    end_index: int
    top: float
    bottom: float
    type: str  # 'bullish' or 'bearish'
    volume: float
    strength: float  # 0-100 score
    mitigated: bool = False
    mitigation_index: Optional[int] = None
    breaker_block: bool = False  # True if order block becomes breaker


class OrderBlocks(BaseIndicator):
    """
    Advanced Order Blocks Detection based on LuxAlgo Price Action Concepts
    
    Features:
    - Volumetric order block identification
    - Order block strength scoring
    - Mitigation tracking
    - Breaker block detection (failed order blocks)
    - Real-time updates
    - Order flow imbalance detection
    """
    
    def __init__(self,
                 lookback: int = 10,
                 volume_threshold: float = 1.5,
                 imbalance_threshold: float = 0.7,
                 min_block_size: float = 0.0005,
                 max_blocks: int = 20):
        """
        Initialize Order Blocks indicator
        
        Args:
            lookback: Bars to look back for order block formation
            volume_threshold: Volume multiplier for significant blocks
            imbalance_threshold: Threshold for order flow imbalance (0-1)
            min_block_size: Minimum size as percentage of price
            max_blocks: Maximum number of active blocks to track
        """
        self.lookback = lookback
        self.volume_threshold = volume_threshold
        self.imbalance_threshold = imbalance_threshold
        self.min_block_size = min_block_size
        self.max_blocks = max_blocks
        super().__init__()
        
        self.order_blocks = []
        self.breaker_blocks = []
    
    def _calculate_min_periods(self) -> int:
        """Calculate minimum periods required for this indicator"""
        return self.lookback + 2
        
    def calculate_imbalance(self, data: pd.DataFrame, idx: int) -> float:
        """
        Calculate order flow imbalance for a candle
        
        Args:
            data: OHLC DataFrame with volume
            idx: Candle index
            
        Returns:
            Imbalance ratio (0-1, higher = more imbalanced)
        """
        if idx < 1 or idx >= len(data):
            return 0.0
        
        # Current candle metrics
        body = abs(data['close'].iloc[idx] - data['open'].iloc[idx])
        range_size = data['high'].iloc[idx] - data['low'].iloc[idx]
        
        if range_size == 0:
            return 0.0
        
        # Body to range ratio (strong imbalance)
        body_ratio = body / range_size
        
        # Volume imbalance
        avg_volume = data['volume'].iloc[max(0, idx-20):idx].mean()
        volume_ratio = data['volume'].iloc[idx] / avg_volume if avg_volume > 0 else 1
        
        # Price momentum
        if data['close'].iloc[idx] > data['open'].iloc[idx]:  # Bullish
            upper_wick = data['high'].iloc[idx] - max(data['close'].iloc[idx], data['open'].iloc[idx])
            lower_wick = min(data['close'].iloc[idx], data['open'].iloc[idx]) - data['low'].iloc[idx]
            wick_imbalance = 1 - (upper_wick / range_size) if range_size > 0 else 0
        else:  # Bearish
            upper_wick = data['high'].iloc[idx] - max(data['close'].iloc[idx], data['open'].iloc[idx])
            lower_wick = min(data['close'].iloc[idx], data['open'].iloc[idx]) - data['low'].iloc[idx]
            wick_imbalance = 1 - (lower_wick / range_size) if range_size > 0 else 0
        
        # Combine factors
        imbalance = (body_ratio * 0.4 + min(volume_ratio, 2) / 2 * 0.3 + wick_imbalance * 0.3)
        
        return min(imbalance, 1.0)
    
    def identify_order_block(self, data: pd.DataFrame, idx: int) -> Optional[OrderBlock]:
        """
        Identify potential order block at given index
        
        Args:
            data: OHLC DataFrame with volume
            idx: Current index to check
            
        Returns:
            OrderBlock if found, None otherwise
        """
        if idx < self.lookback or idx >= len(data) - 1:
            return None
        
        # Check for strong move after potential order block
        current_close = data['close'].iloc[idx]
        next_close = data['close'].iloc[idx + 1]
        
        # Calculate average volume
        avg_volume = data['volume'].iloc[max(0, idx-20):idx].mean()
        if avg_volume == 0:
            return None
        
        # Bullish order block: Last bearish candle before strong bullish move
        if next_close > current_close:
            # Look for last bearish candle
            for i in range(idx, max(idx - self.lookback, 0), -1):
                if data['close'].iloc[i] < data['open'].iloc[i]:  # Bearish candle
                    # Check if followed by strong bullish move
                    move_size = (next_close - data['low'].iloc[i]) / data['low'].iloc[i]
                    
                    if move_size > self.min_block_size:
                        # Check volume
                        block_volume = data['volume'].iloc[i]
                        if block_volume > avg_volume * self.volume_threshold:
                            # Check imbalance
                            imbalance = self.calculate_imbalance(data, i)
                            if imbalance > self.imbalance_threshold:
                                # Calculate strength
                                volume_score = min(block_volume / (avg_volume * 2), 1) * 30
                                imbalance_score = imbalance * 30
                                move_score = min(move_size / (self.min_block_size * 3), 1) * 40
                                strength = volume_score + imbalance_score + move_score
                                
                                return OrderBlock(
                                    start_index=i,
                                    end_index=i,
                                    top=data['high'].iloc[i],
                                    bottom=data['low'].iloc[i],
                                    type='bullish',
                                    volume=block_volume,
                                    strength=strength
                                )
                    break
        
        # Bearish order block: Last bullish candle before strong bearish move
        elif next_close < current_close:
            # Look for last bullish candle
            for i in range(idx, max(idx - self.lookback, 0), -1):
                if data['close'].iloc[i] > data['open'].iloc[i]:  # Bullish candle
                    # Check if followed by strong bearish move
                    move_size = (data['high'].iloc[i] - next_close) / data['high'].iloc[i]
                    
                    if move_size > self.min_block_size:
                        # Check volume
                        block_volume = data['volume'].iloc[i]
                        if block_volume > avg_volume * self.volume_threshold:
                            # Check imbalance
                            imbalance = self.calculate_imbalance(data, i)
                            if imbalance > self.imbalance_threshold:
                                # Calculate strength
                                volume_score = min(block_volume / (avg_volume * 2), 1) * 30
                                imbalance_score = imbalance * 30
                                move_score = min(move_size / (self.min_block_size * 3), 1) * 40
                                strength = volume_score + imbalance_score + move_score
                                
                                return OrderBlock(
                                    start_index=i,
                                    end_index=i,
                                    top=data['high'].iloc[i],
                                    bottom=data['low'].iloc[i],
                                    type='bearish',
                                    volume=block_volume,
                                    strength=strength
                                )
                    break
        
        return None
    
    def check_mitigation(self, block: OrderBlock, data: pd.DataFrame, current_idx: int) -> bool:
        """
        Check if order block has been mitigated
        
        Args:
            block: Order block to check
            data: OHLC DataFrame
            current_idx: Current candle index
            
        Returns:
            True if mitigated, False otherwise
        """
        if block.mitigated or current_idx <= block.end_index:
            return block.mitigated
        
        current_high = data['high'].iloc[current_idx]
        current_low = data['low'].iloc[current_idx]
        
        # Bullish order block mitigated if price trades below bottom
        if block.type == 'bullish' and current_low < block.bottom:
            block.mitigated = True
            block.mitigation_index = current_idx
            return True
        
        # Bearish order block mitigated if price trades above top
        elif block.type == 'bearish' and current_high > block.top:
            block.mitigated = True
            block.mitigation_index = current_idx
            return True
        
        return False
    
    def check_breaker_block(self, block: OrderBlock, data: pd.DataFrame) -> bool:
        """
        Check if mitigated order block becomes a breaker block
        
        Args:
            block: Mitigated order block
            data: OHLC DataFrame
            
        Returns:
            True if breaker block formed, False otherwise
        """
        if not block.mitigated or block.mitigation_index is None:
            return False
        
        # Check price action after mitigation
        if block.mitigation_index + 5 < len(data):
            post_mitigation_data = data.iloc[block.mitigation_index:block.mitigation_index + 5]
            
            if block.type == 'bullish':
                # Bullish OB becomes bearish breaker if price continues lower
                if post_mitigation_data['close'].iloc[-1] < block.bottom * 0.995:
                    block.breaker_block = True
                    return True
            else:
                # Bearish OB becomes bullish breaker if price continues higher  
                if post_mitigation_data['close'].iloc[-1] > block.top * 1.005:
                    block.breaker_block = True
                    return True
        
        return False
    
    def rank_order_blocks(self) -> List[OrderBlock]:
        """
        Rank order blocks by strength and recency
        
        Returns:
            Sorted list of order blocks (strongest first)
        """
        active_blocks = [b for b in self.order_blocks if not b.mitigated]
        
        # Sort by strength and recency
        active_blocks.sort(key=lambda b: (b.strength, -b.start_index), reverse=True)
        
        return active_blocks
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate order blocks for given data
        
        Args:
            data: OHLC DataFrame with volume
            
        Returns:
            DataFrame with order block analysis
        """
        if len(data) < self.lookback + 2:
            return pd.DataFrame()
        
        # Reset blocks for fresh calculation
        self.order_blocks = []
        self.breaker_blocks = []
        
        # Identify order blocks
        for i in range(self.lookback, len(data) - 1):
            block = self.identify_order_block(data, i)
            if block:
                # Check if overlaps with existing blocks
                overlap = False
                for existing in self.order_blocks:
                    if (existing.type == block.type and 
                        abs(existing.start_index - block.start_index) < 3):
                        overlap = True
                        break
                
                if not overlap:
                    self.order_blocks.append(block)
        
        # Check mitigation and breaker blocks
        for i in range(len(data)):
            for block in self.order_blocks:
                if self.check_mitigation(block, data, i):
                    if self.check_breaker_block(block, data):
                        self.breaker_blocks.append(block)
        
        # Limit number of tracked blocks
        if len(self.order_blocks) > self.max_blocks:
            self.order_blocks = self.rank_order_blocks()[:self.max_blocks]
        
        # Create output DataFrame
        result = pd.DataFrame(index=data.index)
        
        # Initialize columns
        result['bullish_ob'] = 0.0
        result['bearish_ob'] = 0.0
        result['ob_strength'] = 0.0
        result['breaker_bullish'] = 0.0
        result['breaker_bearish'] = 0.0
        result['ob_volume_ratio'] = 1.0
        
        # Mark order blocks
        for block in self.order_blocks:
            if not block.mitigated:
                # Mark the range of the order block
                for i in range(block.start_index, min(block.end_index + 1, len(result))):
                    if block.type == 'bullish':
                        result.iloc[i, result.columns.get_loc('bullish_ob')] = block.bottom
                        result.iloc[i, result.columns.get_loc('ob_strength')] = block.strength
                    else:
                        result.iloc[i, result.columns.get_loc('bearish_ob')] = block.top
                        result.iloc[i, result.columns.get_loc('ob_strength')] = block.strength
                    
                    # Volume ratio
                    avg_vol = data['volume'].iloc[max(0, i-20):i].mean()
                    if avg_vol > 0:
                        result.iloc[i, result.columns.get_loc('ob_volume_ratio')] = block.volume / avg_vol
        
        # Mark breaker blocks
        for block in self.breaker_blocks:
            if block.mitigation_index and block.mitigation_index < len(result):
                if block.type == 'bullish':  # Becomes bearish breaker
                    result.iloc[block.mitigation_index, result.columns.get_loc('breaker_bearish')] = block.bottom
                else:  # Becomes bullish breaker
                    result.iloc[block.mitigation_index, result.columns.get_loc('breaker_bullish')] = block.top
        
        # Add order flow imbalance
        result['order_flow_imbalance'] = 0.0
        for i in range(len(data)):
            result.iloc[i, result.columns.get_loc('order_flow_imbalance')] = self.calculate_imbalance(data, i)
        
        return result
    
    def get_active_blocks(self, current_price: float) -> Dict[str, List[OrderBlock]]:
        """
        Get currently active order blocks relative to price
        
        Args:
            current_price: Current market price
            
        Returns:
            Dictionary with 'above' and 'below' order blocks
        """
        active = {'above': [], 'below': []}
        
        for block in self.order_blocks:
            if not block.mitigated:
                if block.type == 'bearish' and block.bottom > current_price:
                    active['above'].append(block)
                elif block.type == 'bullish' and block.top < current_price:
                    active['below'].append(block)
        
        # Sort by distance from current price
        active['above'].sort(key=lambda b: b.bottom - current_price)
        active['below'].sort(key=lambda b: current_price - b.top)
        
        return active
    
    def get_nearest_blocks(self, current_price: float, n: int = 3) -> Dict[str, List[Tuple[float, float, float]]]:
        """
        Get nearest order blocks to current price
        
        Args:
            current_price: Current market price
            n: Number of blocks to return
            
        Returns:
            Dictionary with nearest support and resistance blocks as (price, strength, volume_ratio)
        """
        active = self.get_active_blocks(current_price)
        
        nearest = {
            'resistance': [(b.bottom, b.strength, b.volume) for b in active['above'][:n]],
            'support': [(b.top, b.strength, b.volume) for b in active['below'][:n]]
        }
        
        return nearest