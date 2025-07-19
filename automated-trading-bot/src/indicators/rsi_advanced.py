"""
Advanced RSI Indicator - Based on TradingView Specifications
Includes divergence detection, overbought/oversold zones, and signal generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class RSISignalType(Enum):
    """RSI signal types"""
    OVERSOLD = "oversold"
    OVERBOUGHT = "overbought"
    BULLISH_DIVERGENCE = "bullish_divergence"
    BEARISH_DIVERGENCE = "bearish_divergence"
    MIDLINE_CROSS = "midline_cross"
    HIDDEN_BULLISH_DIV = "hidden_bullish_divergence"
    HIDDEN_BEARISH_DIV = "hidden_bearish_divergence"


@dataclass
class RSISignal:
    """Container for RSI signals"""
    timestamp: pd.Timestamp
    signal_type: RSISignalType
    rsi_value: float
    price: float
    strength: float  # 0-1 signal strength
    additional_data: Dict


class AdvancedRSI:
    """
    Advanced RSI implementation with multiple signal types
    Based on TradingView RSI with enhancements
    """
    
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        """
        Initialize Advanced RSI
        
        Args:
            period: RSI calculation period
            overbought: Overbought threshold (default 70)
            oversold: Oversold threshold (default 30)
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.midline = 50
        
        # Additional levels for fine-tuning
        self.extreme_overbought = 80
        self.extreme_oversold = 20
        
        # Divergence parameters
        self.divergence_lookback = 20  # Bars to look back for divergences
        self.min_pivot_distance = 5    # Minimum bars between pivots
        
    def calculate(self, prices: pd.Series) -> pd.Series:
        """
        Calculate RSI values
        
        Args:
            prices: Series of prices (typically close)
            
        Returns:
            Series of RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        # Use SMA for first calculation
        avg_gains = gains.rolling(window=self.period).mean()
        avg_losses = losses.rolling(window=self.period).mean()
        
        # Use EMA for subsequent calculations (Wilder's smoothing)
        for i in range(self.period, len(prices)):
            avg_gains.iloc[i] = (avg_gains.iloc[i-1] * (self.period - 1) + gains.iloc[i]) / self.period
            avg_losses.iloc[i] = (avg_losses.iloc[i-1] * (self.period - 1) + losses.iloc[i]) / self.period
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses.replace(0, 0.0001)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_with_smoothing(self, prices: pd.Series, smooth_period: int = 3) -> pd.Series:
        """
        Calculate RSI with additional smoothing
        
        Args:
            prices: Price series
            smooth_period: Period for smoothing RSI
            
        Returns:
            Smoothed RSI values
        """
        rsi = self.calculate(prices)
        smoothed_rsi = rsi.rolling(window=smooth_period).mean()
        return smoothed_rsi
    
    def detect_divergences(self, prices: pd.Series, rsi: pd.Series) -> List[RSISignal]:
        """
        Detect RSI divergences (regular and hidden)
        
        Args:
            prices: Price series
            rsi: RSI series
            
        Returns:
            List of divergence signals
        """
        signals = []
        
        # Find price and RSI pivots
        price_highs = self._find_pivots(prices, pivot_type='high')
        price_lows = self._find_pivots(prices, pivot_type='low')
        rsi_highs = self._find_pivots(rsi, pivot_type='high')
        rsi_lows = self._find_pivots(rsi, pivot_type='low')
        
        # Check for regular bearish divergence (price higher high, RSI lower high)
        for i in range(1, len(price_highs)):
            if i < len(rsi_highs):
                price_idx1, price_val1 = price_highs[i-1]
                price_idx2, price_val2 = price_highs[i]
                rsi_idx1, rsi_val1 = rsi_highs[i-1]
                rsi_idx2, rsi_val2 = rsi_highs[i]
                
                if (price_idx2 - price_idx1 >= self.min_pivot_distance and
                    abs(price_idx2 - rsi_idx2) < 3):  # Price and RSI pivots should be close
                    
                    if price_val2 > price_val1 and rsi_val2 < rsi_val1:
                        signals.append(RSISignal(
                            timestamp=prices.index[price_idx2],
                            signal_type=RSISignalType.BEARISH_DIVERGENCE,
                            rsi_value=rsi_val2,
                            price=price_val2,
                            strength=min((price_val2 - price_val1) / price_val1 * 100, 1.0),
                            additional_data={
                                'pivot_distance': price_idx2 - price_idx1,
                                'rsi_difference': rsi_val1 - rsi_val2
                            }
                        ))
        
        # Check for regular bullish divergence (price lower low, RSI higher low)
        for i in range(1, len(price_lows)):
            if i < len(rsi_lows):
                price_idx1, price_val1 = price_lows[i-1]
                price_idx2, price_val2 = price_lows[i]
                rsi_idx1, rsi_val1 = rsi_lows[i-1]
                rsi_idx2, rsi_val2 = rsi_lows[i]
                
                if (price_idx2 - price_idx1 >= self.min_pivot_distance and
                    abs(price_idx2 - rsi_idx2) < 3):
                    
                    if price_val2 < price_val1 and rsi_val2 > rsi_val1:
                        signals.append(RSISignal(
                            timestamp=prices.index[price_idx2],
                            signal_type=RSISignalType.BULLISH_DIVERGENCE,
                            rsi_value=rsi_val2,
                            price=price_val2,
                            strength=min((price_val1 - price_val2) / price_val1 * 100, 1.0),
                            additional_data={
                                'pivot_distance': price_idx2 - price_idx1,
                                'rsi_difference': rsi_val2 - rsi_val1
                            }
                        ))
        
        # Hidden divergences (for trend continuation)
        # Hidden bearish: price lower high, RSI higher high (in downtrend)
        # Hidden bullish: price higher low, RSI lower low (in uptrend)
        
        return signals
    
    def generate_signals(self, prices: pd.Series, volume: Optional[pd.Series] = None) -> List[RSISignal]:
        """
        Generate all RSI-based signals
        
        Args:
            prices: Price series
            volume: Optional volume series for confirmation
            
        Returns:
            List of RSI signals
        """
        signals = []
        
        # Calculate RSI
        rsi = self.calculate(prices)
        
        # Skip if not enough data
        if len(rsi) < self.divergence_lookback:
            return signals
        
        # 1. Oversold/Overbought signals
        for i in range(1, len(rsi)):
            current_rsi = rsi.iloc[i]
            prev_rsi = rsi.iloc[i-1]
            
            # Oversold signal
            if prev_rsi > self.oversold and current_rsi <= self.oversold:
                strength = (self.oversold - current_rsi) / self.oversold
                signals.append(RSISignal(
                    timestamp=prices.index[i],
                    signal_type=RSISignalType.OVERSOLD,
                    rsi_value=current_rsi,
                    price=prices.iloc[i],
                    strength=min(strength * 2, 1.0),  # Scale up strength
                    additional_data={'extreme': current_rsi <= self.extreme_oversold}
                ))
            
            # Overbought signal
            elif prev_rsi < self.overbought and current_rsi >= self.overbought:
                strength = (current_rsi - self.overbought) / (100 - self.overbought)
                signals.append(RSISignal(
                    timestamp=prices.index[i],
                    signal_type=RSISignalType.OVERBOUGHT,
                    rsi_value=current_rsi,
                    price=prices.iloc[i],
                    strength=min(strength * 2, 1.0),
                    additional_data={'extreme': current_rsi >= self.extreme_overbought}
                ))
            
            # Midline cross
            if prev_rsi < self.midline and current_rsi >= self.midline:
                signals.append(RSISignal(
                    timestamp=prices.index[i],
                    signal_type=RSISignalType.MIDLINE_CROSS,
                    rsi_value=current_rsi,
                    price=prices.iloc[i],
                    strength=0.5,  # Moderate strength for midline crosses
                    additional_data={'direction': 'bullish'}
                ))
            elif prev_rsi > self.midline and current_rsi <= self.midline:
                signals.append(RSISignal(
                    timestamp=prices.index[i],
                    signal_type=RSISignalType.MIDLINE_CROSS,
                    rsi_value=current_rsi,
                    price=prices.iloc[i],
                    strength=0.5,
                    additional_data={'direction': 'bearish'}
                ))
        
        # 2. Add divergence signals
        divergence_signals = self.detect_divergences(prices, rsi)
        signals.extend(divergence_signals)
        
        # 3. Sort signals by timestamp
        signals.sort(key=lambda x: x.timestamp)
        
        return signals
    
    def _find_pivots(self, series: pd.Series, pivot_type: str = 'high', 
                    left_bars: int = 5, right_bars: int = 5) -> List[Tuple[int, float]]:
        """
        Find pivot points in a series
        
        Args:
            series: Data series
            pivot_type: 'high' or 'low'
            left_bars: Bars to left that must be lower/higher
            right_bars: Bars to right that must be lower/higher
            
        Returns:
            List of (index, value) tuples for pivots
        """
        pivots = []
        
        for i in range(left_bars, len(series) - right_bars):
            if pivot_type == 'high':
                # Check if current bar is highest
                is_pivot = all(series.iloc[i] > series.iloc[i-j] for j in range(1, left_bars+1))
                is_pivot &= all(series.iloc[i] > series.iloc[i+j] for j in range(1, right_bars+1))
            else:  # low
                # Check if current bar is lowest
                is_pivot = all(series.iloc[i] < series.iloc[i-j] for j in range(1, left_bars+1))
                is_pivot &= all(series.iloc[i] < series.iloc[i+j] for j in range(1, right_bars+1))
            
            if is_pivot:
                pivots.append((i, series.iloc[i]))
        
        return pivots
    
    def get_signal_strength(self, rsi_value: float, price_momentum: float = 0) -> float:
        """
        Calculate overall signal strength based on RSI and price momentum
        
        Args:
            rsi_value: Current RSI value
            price_momentum: Optional price momentum
            
        Returns:
            Signal strength (0-1)
        """
        strength = 0.0
        
        # RSI extremes
        if rsi_value <= self.extreme_oversold:
            strength = 0.9
        elif rsi_value <= self.oversold:
            strength = 0.7
        elif rsi_value >= self.extreme_overbought:
            strength = 0.9
        elif rsi_value >= self.overbought:
            strength = 0.7
        else:
            # Midline proximity
            distance_from_midline = abs(rsi_value - self.midline)
            strength = max(0.3, 1 - distance_from_midline / 50)
        
        # Adjust for price momentum if provided
        if price_momentum != 0:
            momentum_factor = min(abs(price_momentum) / 2, 0.3)  # Max 30% adjustment
            if (rsi_value < 50 and price_momentum > 0) or (rsi_value > 50 and price_momentum < 0):
                # Divergence between RSI and price
                strength += momentum_factor
            else:
                # Confirmation
                strength += momentum_factor * 0.5
        
        return min(strength, 1.0)
    
    def backtest_parameters(self, prices: pd.Series, 
                          period_range: Tuple[int, int] = (10, 20),
                          ob_range: Tuple[int, int] = (65, 80),
                          os_range: Tuple[int, int] = (20, 35)) -> Dict:
        """
        Backtest different RSI parameters
        
        Args:
            prices: Price series
            period_range: Range of periods to test
            ob_range: Range of overbought levels
            os_range: Range of oversold levels
            
        Returns:
            Best parameters and performance metrics
        """
        best_params = {}
        best_score = -float('inf')
        
        for period in range(period_range[0], period_range[1] + 1, 2):
            for ob in range(ob_range[0], ob_range[1] + 1, 5):
                for os in range(os_range[0], os_range[1] + 1, 5):
                    # Create RSI with test parameters
                    test_rsi = AdvancedRSI(period=period, overbought=ob, oversold=os)
                    signals = test_rsi.generate_signals(prices)
                    
                    # Score based on signal quality
                    score = self._score_signals(signals, prices)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'period': period,
                            'overbought': ob,
                            'oversold': os,
                            'score': score,
                            'signal_count': len(signals)
                        }
        
        return best_params
    
    def _score_signals(self, signals: List[RSISignal], prices: pd.Series) -> float:
        """
        Score signals based on subsequent price movement
        
        Args:
            signals: List of RSI signals
            prices: Price series
            
        Returns:
            Score value
        """
        if not signals:
            return 0
        
        total_score = 0
        look_forward = 10  # Bars to look forward
        
        for signal in signals:
            idx = prices.index.get_loc(signal.timestamp)
            if idx + look_forward < len(prices):
                future_prices = prices.iloc[idx:idx+look_forward+1]
                price_change = (future_prices.iloc[-1] - future_prices.iloc[0]) / future_prices.iloc[0]
                
                # Score based on signal type and price movement
                if signal.signal_type in [RSISignalType.OVERSOLD, RSISignalType.BULLISH_DIVERGENCE]:
                    # Expecting price increase
                    total_score += price_change * signal.strength * 100
                elif signal.signal_type in [RSISignalType.OVERBOUGHT, RSISignalType.BEARISH_DIVERGENCE]:
                    # Expecting price decrease
                    total_score += -price_change * signal.strength * 100
        
        return total_score / len(signals) if signals else 0