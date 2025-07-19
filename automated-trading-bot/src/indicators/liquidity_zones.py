"""
Liquidity Zones Indicator - LuxAlgo Price Action Concepts
Identifies areas of high liquidity concentration and potential liquidity grabs
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .base import BaseIndicator


@dataclass
class LiquidityZone:
    """Represents a liquidity zone"""
    start_index: int
    end_index: int
    price_level: float
    zone_high: float
    zone_low: float
    type: str  # 'resistance', 'support', 'premium', 'discount'
    strength: float  # 0-100
    touches: int
    liquidity_type: str  # 'stop_hunt', 'accumulation', 'distribution'


@dataclass
class LiquidityGrab:
    """Represents a liquidity grab event"""
    index: int
    price: float
    direction: str  # 'bullish' or 'bearish'
    zone_violated: Optional[LiquidityZone]
    grab_depth: float  # How deep into zone
    reversal_strength: float  # 0-100


class LiquidityZones(BaseIndicator):
    """
    Liquidity Zones Detection based on LuxAlgo Price Action Concepts
    
    Features:
    - Trend line liquidity detection
    - Chart pattern liquidity zones
    - Liquidity grab identification
    - Premium/discount zone calculation
    - Stop hunt detection
    - Multi-touch zone validation
    """
    
    def __init__(self,
                 min_touches: int = 2,
                 zone_tolerance: float = 0.002,
                 lookback: int = 100,
                 grab_threshold: float = 0.001,
                 max_zones: int = 10):
        """
        Initialize Liquidity Zones indicator
        
        Args:
            min_touches: Minimum touches to form a zone
            zone_tolerance: Price tolerance for zone (as % of price)
            lookback: Bars to look back for zone formation
            grab_threshold: Minimum penetration for liquidity grab
            max_zones: Maximum number of zones to track
        """
        self.min_touches = min_touches
        self.zone_tolerance = zone_tolerance
        self.lookback = lookback
        self.grab_threshold = grab_threshold
        self.max_zones = max_zones
        super().__init__()
        
        self.liquidity_zones = []
        self.liquidity_grabs = []
    
    def _calculate_min_periods(self) -> int:
        """Calculate minimum periods required for this indicator"""
        return self.lookback
        
    def find_swing_points(self, data: pd.DataFrame, lookback: int = 5) -> Tuple[List[int], List[int]]:
        """
        Find swing highs and lows
        
        Args:
            data: OHLC DataFrame
            lookback: Bars to check for swing
            
        Returns:
            Tuple of (swing_high_indices, swing_low_indices)
        """
        highs = []
        lows = []
        
        for i in range(lookback, len(data) - lookback):
            # Swing high
            if all(data['high'].iloc[i] >= data['high'].iloc[i-j] for j in range(1, lookback+1)) and \
               all(data['high'].iloc[i] >= data['high'].iloc[i+j] for j in range(1, lookback+1)):
                highs.append(i)
            
            # Swing low
            if all(data['low'].iloc[i] <= data['low'].iloc[i-j] for j in range(1, lookback+1)) and \
               all(data['low'].iloc[i] <= data['low'].iloc[i+j] for j in range(1, lookback+1)):
                lows.append(i)
        
        return highs, lows
    
    def cluster_price_levels(self, levels: List[Tuple[int, float]], 
                           tolerance: float) -> List[LiquidityZone]:
        """
        Cluster price levels into zones
        
        Args:
            levels: List of (index, price) tuples
            tolerance: Price tolerance for clustering
            
        Returns:
            List of liquidity zones
        """
        if not levels:
            return []
        
        # Sort by price
        levels.sort(key=lambda x: x[1])
        
        zones = []
        current_cluster = [levels[0]]
        
        for i in range(1, len(levels)):
            idx, price = levels[i]
            cluster_avg = np.mean([l[1] for l in current_cluster])
            
            # Check if within tolerance of cluster
            if abs(price - cluster_avg) / cluster_avg <= tolerance:
                current_cluster.append((idx, price))
            else:
                # Create zone from cluster
                if len(current_cluster) >= self.min_touches:
                    prices = [l[1] for l in current_cluster]
                    indices = [l[0] for l in current_cluster]
                    
                    zone = LiquidityZone(
                        start_index=min(indices),
                        end_index=max(indices),
                        price_level=np.mean(prices),
                        zone_high=max(prices) * (1 + tolerance/2),
                        zone_low=min(prices) * (1 - tolerance/2),
                        type='support' if price < np.mean(prices) else 'resistance',
                        strength=len(current_cluster) * 20,  # Base strength on touches
                        touches=len(current_cluster),
                        liquidity_type='accumulation'
                    )
                    zones.append(zone)
                
                # Start new cluster
                current_cluster = [(idx, price)]
        
        # Handle last cluster
        if len(current_cluster) >= self.min_touches:
            prices = [l[1] for l in current_cluster]
            indices = [l[0] for l in current_cluster]
            
            zone = LiquidityZone(
                start_index=min(indices),
                end_index=max(indices),
                price_level=np.mean(prices),
                zone_high=max(prices) * (1 + tolerance/2),
                zone_low=min(prices) * (1 - tolerance/2),
                type='resistance',
                strength=len(current_cluster) * 20,
                touches=len(current_cluster),
                liquidity_type='distribution'
            )
            zones.append(zone)
        
        return zones
    
    def identify_trendline_liquidity(self, data: pd.DataFrame, 
                                   swing_points: List[int], 
                                   is_high: bool) -> List[LiquidityZone]:
        """
        Identify liquidity along trendlines
        
        Args:
            data: OHLC DataFrame
            swing_points: Swing point indices
            is_high: True for highs, False for lows
            
        Returns:
            List of trendline liquidity zones
        """
        if len(swing_points) < 3:
            return []
        
        zones = []
        
        # Check each pair of swing points for trendline
        for i in range(len(swing_points) - 2):
            p1_idx = swing_points[i]
            p2_idx = swing_points[i + 1]
            
            if is_high:
                p1_price = data['high'].iloc[p1_idx]
                p2_price = data['high'].iloc[p2_idx]
            else:
                p1_price = data['low'].iloc[p1_idx]
                p2_price = data['low'].iloc[p2_idx]
            
            # Calculate trendline slope
            slope = (p2_price - p1_price) / (p2_idx - p1_idx)
            
            # Check how many other points touch this trendline
            touches = []
            for j in range(i + 2, len(swing_points)):
                p3_idx = swing_points[j]
                expected_price = p1_price + slope * (p3_idx - p1_idx)
                
                if is_high:
                    actual_price = data['high'].iloc[p3_idx]
                else:
                    actual_price = data['low'].iloc[p3_idx]
                
                # Check if point touches trendline
                if abs(actual_price - expected_price) / expected_price <= self.zone_tolerance:
                    touches.append((p3_idx, actual_price))
            
            # Create zone if enough touches
            if len(touches) >= 1:  # At least 3 total points
                all_points = [(p1_idx, p1_price), (p2_idx, p2_price)] + touches
                prices = [p[1] for p in all_points]
                
                zone = LiquidityZone(
                    start_index=p1_idx,
                    end_index=max(p[0] for p in all_points),
                    price_level=np.mean(prices),
                    zone_high=max(prices) * (1 + self.zone_tolerance/2),
                    zone_low=min(prices) * (1 - self.zone_tolerance/2),
                    type='resistance' if is_high else 'support',
                    strength=30 + len(all_points) * 15,
                    touches=len(all_points),
                    liquidity_type='stop_hunt'
                )
                zones.append(zone)
        
        return zones
    
    def calculate_premium_discount_zones(self, data: pd.DataFrame, 
                                       lookback: int) -> Tuple[LiquidityZone, LiquidityZone]:
        """
        Calculate premium and discount zones based on range
        
        Args:
            data: OHLC DataFrame
            lookback: Bars to calculate range
            
        Returns:
            Tuple of (premium_zone, discount_zone)
        """
        if len(data) < lookback:
            return None, None
        
        # Get range
        recent_data = data.iloc[-lookback:]
        range_high = recent_data['high'].max()
        range_low = recent_data['low'].min()
        range_mid = (range_high + range_low) / 2
        
        # Premium zone (upper 25% of range)
        premium_low = range_mid + (range_high - range_mid) * 0.5
        premium_zone = LiquidityZone(
            start_index=len(data) - lookback,
            end_index=len(data) - 1,
            price_level=(range_high + premium_low) / 2,
            zone_high=range_high,
            zone_low=premium_low,
            type='premium',
            strength=50,
            touches=0,
            liquidity_type='distribution'
        )
        
        # Discount zone (lower 25% of range)
        discount_high = range_mid - (range_mid - range_low) * 0.5
        discount_zone = LiquidityZone(
            start_index=len(data) - lookback,
            end_index=len(data) - 1,
            price_level=(discount_high + range_low) / 2,
            zone_high=discount_high,
            zone_low=range_low,
            type='discount',
            strength=50,
            touches=0,
            liquidity_type='accumulation'
        )
        
        return premium_zone, discount_zone
    
    def detect_liquidity_grab(self, zone: LiquidityZone, data: pd.DataFrame, 
                            idx: int) -> Optional[LiquidityGrab]:
        """
        Detect if current candle is grabbing liquidity from zone
        
        Args:
            zone: Liquidity zone to check
            data: OHLC DataFrame
            idx: Current candle index
            
        Returns:
            LiquidityGrab if detected, None otherwise
        """
        if idx >= len(data) - 1:
            return None
        
        current = data.iloc[idx]
        next_candle = data.iloc[idx + 1]
        
        # Bearish liquidity grab (sweep highs and reverse)
        if zone.type in ['resistance', 'premium']:
            if current['high'] > zone.zone_high:
                penetration = (current['high'] - zone.zone_high) / zone.zone_high
                
                if penetration >= self.grab_threshold:
                    # Check for reversal
                    if next_candle['close'] < zone.zone_high:
                        reversal_strength = min((zone.zone_high - next_candle['close']) / 
                                               (current['high'] - zone.zone_high) * 50, 100)
                        
                        return LiquidityGrab(
                            index=idx,
                            price=current['high'],
                            direction='bearish',
                            zone_violated=zone,
                            grab_depth=penetration,
                            reversal_strength=reversal_strength
                        )
        
        # Bullish liquidity grab (sweep lows and reverse)
        elif zone.type in ['support', 'discount']:
            if current['low'] < zone.zone_low:
                penetration = (zone.zone_low - current['low']) / zone.zone_low
                
                if penetration >= self.grab_threshold:
                    # Check for reversal
                    if next_candle['close'] > zone.zone_low:
                        reversal_strength = min((next_candle['close'] - zone.zone_low) / 
                                               (zone.zone_low - current['low']) * 50, 100)
                        
                        return LiquidityGrab(
                            index=idx,
                            price=current['low'],
                            direction='bullish',
                            zone_violated=zone,
                            grab_depth=penetration,
                            reversal_strength=reversal_strength
                        )
        
        return None
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate liquidity zones for given data
        
        Args:
            data: OHLC DataFrame
            
        Returns:
            DataFrame with liquidity zone analysis
        """
        if len(data) < self.lookback:
            return pd.DataFrame()
        
        # Reset zones
        self.liquidity_zones = []
        self.liquidity_grabs = []
        
        # Find swing points
        swing_highs, swing_lows = self.find_swing_points(data)
        
        # Create zones from swing point clusters
        high_levels = [(idx, data['high'].iloc[idx]) for idx in swing_highs]
        low_levels = [(idx, data['low'].iloc[idx]) for idx in swing_lows]
        
        resistance_zones = self.cluster_price_levels(high_levels, self.zone_tolerance)
        support_zones = self.cluster_price_levels(low_levels, self.zone_tolerance)
        
        # Add trendline liquidity
        trendline_resistance = self.identify_trendline_liquidity(data, swing_highs, True)
        trendline_support = self.identify_trendline_liquidity(data, swing_lows, False)
        
        # Calculate premium/discount zones
        premium_zone, discount_zone = self.calculate_premium_discount_zones(data, self.lookback)
        
        # Combine all zones
        all_zones = resistance_zones + support_zones + trendline_resistance + trendline_support
        if premium_zone:
            all_zones.append(premium_zone)
        if discount_zone:
            all_zones.append(discount_zone)
        
        # Sort by strength and limit
        all_zones.sort(key=lambda z: z.strength, reverse=True)
        self.liquidity_zones = all_zones[:self.max_zones]
        
        # Detect liquidity grabs
        for i in range(1, len(data) - 1):
            for zone in self.liquidity_zones:
                grab = self.detect_liquidity_grab(zone, data, i)
                if grab:
                    self.liquidity_grabs.append(grab)
        
        # Create output DataFrame
        result = pd.DataFrame(index=data.index)
        
        # Initialize columns
        result['resistance_zone'] = 0.0
        result['support_zone'] = 0.0
        result['premium_zone'] = 0.0
        result['discount_zone'] = 0.0
        result['zone_strength'] = 0.0
        result['liquidity_grab'] = 0
        result['grab_direction'] = ''
        result['grab_strength'] = 0.0
        
        # Mark zones
        for zone in self.liquidity_zones:
            for i in range(zone.start_index, min(zone.end_index + 1, len(result))):
                if zone.type == 'resistance':
                    result.iloc[i, result.columns.get_loc('resistance_zone')] = zone.price_level
                elif zone.type == 'support':
                    result.iloc[i, result.columns.get_loc('support_zone')] = zone.price_level
                elif zone.type == 'premium':
                    result.iloc[i, result.columns.get_loc('premium_zone')] = zone.price_level
                elif zone.type == 'discount':
                    result.iloc[i, result.columns.get_loc('discount_zone')] = zone.price_level
                
                # Update strength if stronger
                current_strength = result.iloc[i, result.columns.get_loc('zone_strength')]
                result.iloc[i, result.columns.get_loc('zone_strength')] = max(current_strength, zone.strength)
        
        # Mark liquidity grabs
        for grab in self.liquidity_grabs:
            if grab.index < len(result):
                result.iloc[grab.index, result.columns.get_loc('liquidity_grab')] = 1
                result.iloc[grab.index, result.columns.get_loc('grab_direction')] = grab.direction
                result.iloc[grab.index, result.columns.get_loc('grab_strength')] = grab.reversal_strength
        
        # Add zone density
        result['zone_density'] = 0.0
        for i in range(len(result)):
            active_zones = sum(1 for z in self.liquidity_zones 
                             if z.start_index <= i <= z.end_index)
            result.iloc[i, result.columns.get_loc('zone_density')] = active_zones
        
        return result
    
    def get_nearest_zones(self, current_price: float) -> Dict[str, List[Tuple[float, float, str]]]:
        """
        Get nearest liquidity zones to current price
        
        Args:
            current_price: Current market price
            
        Returns:
            Dictionary with nearest zones as (price, strength, type)
        """
        above_zones = []
        below_zones = []
        
        for zone in self.liquidity_zones:
            if zone.zone_low > current_price:
                above_zones.append((zone.price_level, zone.strength, zone.type))
            elif zone.zone_high < current_price:
                below_zones.append((zone.price_level, zone.strength, zone.type))
        
        # Sort by distance
        above_zones.sort(key=lambda z: z[0] - current_price)
        below_zones.sort(key=lambda z: current_price - z[0])
        
        return {
            'above': above_zones[:3],
            'below': below_zones[:3]
        }
    
    def get_liquidity_statistics(self) -> Dict:
        """
        Get statistics about liquidity zones and grabs
        
        Returns:
            Dictionary with liquidity statistics
        """
        total_grabs = len(self.liquidity_grabs)
        bullish_grabs = sum(1 for g in self.liquidity_grabs if g.direction == 'bullish')
        bearish_grabs = sum(1 for g in self.liquidity_grabs if g.direction == 'bearish')
        
        avg_grab_strength = np.mean([g.reversal_strength for g in self.liquidity_grabs]) if self.liquidity_grabs else 0
        
        return {
            'total_zones': len(self.liquidity_zones),
            'resistance_zones': sum(1 for z in self.liquidity_zones if z.type == 'resistance'),
            'support_zones': sum(1 for z in self.liquidity_zones if z.type == 'support'),
            'total_grabs': total_grabs,
            'bullish_grabs': bullish_grabs,
            'bearish_grabs': bearish_grabs,
            'avg_grab_strength': avg_grab_strength,
            'avg_zone_touches': np.mean([z.touches for z in self.liquidity_zones]) if self.liquidity_zones else 0
        }