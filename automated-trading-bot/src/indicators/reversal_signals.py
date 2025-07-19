"""
Reversal Signals Indicator Implementation
Based on LuxAlgo's reversal detection methodology
Optimized for Option-Buying strategies
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import talib

from .base import BaseIndicator


@dataclass
class ReversalSignal:
    """Container for reversal signal data"""
    timestamp: pd.Timestamp
    signal_type: str  # 'bullish' or 'bearish'
    strength: float  # 0-1 signal strength
    price: float
    volume: float
    confluence_score: float  # Combined indicator score
    risk_reward: float
    stop_loss: float
    target_price: float


class ReversalSignalsIndicator(BaseIndicator):
    """
    Advanced Reversal Signals Indicator for Option-Buying
    
    Combines multiple reversal detection methods:
    1. Price Structure Analysis (Higher Highs/Lower Lows)
    2. Volume Confirmation
    3. RSI Divergence
    4. Support/Resistance Levels
    5. Candlestick Patterns
    6. Market Structure Shifts
    """
    
    def __init__(self, lookback_period: int = 20, sensitivity: float = 1.5):
        """
        Initialize Reversal Signals Indicator
        
        Args:
            lookback_period: Period for analysis (default: 20)
            sensitivity: Sensitivity factor for signal detection (default: 1.5)
        """
        super().__init__()
        self.lookback_period = lookback_period
        self.sensitivity = sensitivity
        
        # Signal parameters
        self.min_confluence_score = 0.6  # Minimum score for valid signal
        self.volume_threshold = 1.5  # Volume spike threshold
        self.rsi_divergence_threshold = 5  # RSI divergence threshold
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate reversal signals
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with signals and analysis
        """
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        # Initialize results
        results = {
            'bullish_signals': np.zeros(len(data)),
            'bearish_signals': np.zeros(len(data)),
            'signal_strength': np.zeros(len(data)),
            'confluence_score': np.zeros(len(data)),
            'support_levels': np.zeros(len(data)),
            'resistance_levels': np.zeros(len(data))
        }
        
        # Calculate components
        price_structure = self._analyze_price_structure(data)
        volume_signals = self._analyze_volume_patterns(data)
        rsi_divergence = self._detect_rsi_divergence(data)
        support_resistance = self._calculate_support_resistance(data)
        candlestick_patterns = self._detect_candlestick_patterns(data)
        market_structure = self._analyze_market_structure(data)
        
        # Combine signals
        for i in range(self.lookback_period, len(data)):
            # Calculate confluence score
            bullish_score = 0
            bearish_score = 0
            
            # Price structure component (weight: 0.25)
            if price_structure['bullish_reversal'][i]:
                bullish_score += 0.25
            if price_structure['bearish_reversal'][i]:
                bearish_score += 0.25
            
            # Volume confirmation (weight: 0.20)
            if volume_signals['volume_spike'][i]:
                if data['close'].iloc[i] > data['open'].iloc[i]:
                    bullish_score += 0.20
                else:
                    bearish_score += 0.20
            
            # RSI divergence (weight: 0.20)
            if rsi_divergence['bullish_divergence'][i]:
                bullish_score += 0.20
            if rsi_divergence['bearish_divergence'][i]:
                bearish_score += 0.20
            
            # Support/Resistance test (weight: 0.15)
            if self._test_support_bounce(data, i, support_resistance):
                bullish_score += 0.15
            if self._test_resistance_rejection(data, i, support_resistance):
                bearish_score += 0.15
            
            # Candlestick patterns (weight: 0.10)
            if candlestick_patterns['bullish_pattern'][i]:
                bullish_score += 0.10
            if candlestick_patterns['bearish_pattern'][i]:
                bearish_score += 0.10
            
            # Market structure (weight: 0.10)
            if market_structure['trend_shift'][i] == 'bullish':
                bullish_score += 0.10
            elif market_structure['trend_shift'][i] == 'bearish':
                bearish_score += 0.10
            
            # Generate signals
            if bullish_score >= self.min_confluence_score:
                results['bullish_signals'][i] = 1
                results['signal_strength'][i] = bullish_score
                results['confluence_score'][i] = bullish_score
            elif bearish_score >= self.min_confluence_score:
                results['bearish_signals'][i] = 1
                results['signal_strength'][i] = bearish_score
                results['confluence_score'][i] = bearish_score
            
            # Store support/resistance levels
            results['support_levels'][i] = support_resistance['support'][i]
            results['resistance_levels'][i] = support_resistance['resistance'][i]
        
        return results
    
    def _analyze_price_structure(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Analyze price structure for reversal patterns"""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        bullish_reversal = np.zeros(len(data))
        bearish_reversal = np.zeros(len(data))
        
        # Find swing points
        for i in range(self.lookback_period, len(data) - 1):
            # Bullish reversal: Lower low followed by higher high
            recent_lows = low[i-self.lookback_period:i]
            if len(recent_lows) > 0:
                if low[i] < np.min(recent_lows) and close[i] > close[i-1]:
                    # Check for momentum shift
                    if self._check_momentum_shift(data, i, 'bullish'):
                        bullish_reversal[i] = 1
            
            # Bearish reversal: Higher high followed by lower low
            recent_highs = high[i-self.lookback_period:i]
            if len(recent_highs) > 0:
                if high[i] > np.max(recent_highs) and close[i] < close[i-1]:
                    # Check for momentum shift
                    if self._check_momentum_shift(data, i, 'bearish'):
                        bearish_reversal[i] = 1
        
        return {
            'bullish_reversal': bullish_reversal,
            'bearish_reversal': bearish_reversal
        }
    
    def _analyze_volume_patterns(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Analyze volume for confirmation"""
        volume = data['volume'].values
        volume_ma = talib.SMA(volume, timeperiod=self.lookback_period)
        
        volume_spike = np.zeros(len(data))
        
        for i in range(self.lookback_period, len(data)):
            if volume_ma[i] > 0:
                if volume[i] > volume_ma[i] * self.volume_threshold:
                    volume_spike[i] = 1
        
        return {'volume_spike': volume_spike}
    
    def _detect_rsi_divergence(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Detect RSI divergence patterns"""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        # Calculate RSI
        rsi = talib.RSI(close, timeperiod=14)
        
        bullish_divergence = np.zeros(len(data))
        bearish_divergence = np.zeros(len(data))
        
        # Look for divergences
        for i in range(self.lookback_period * 2, len(data)):
            # Find recent peaks and troughs
            price_peaks = self._find_peaks(high[i-self.lookback_period:i])
            price_troughs = self._find_troughs(low[i-self.lookback_period:i])
            rsi_peaks = self._find_peaks(rsi[i-self.lookback_period:i])
            rsi_troughs = self._find_troughs(rsi[i-self.lookback_period:i])
            
            # Bullish divergence: Lower low in price, higher low in RSI
            if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
                if (price_troughs[-1] < price_troughs[-2] and 
                    rsi_troughs[-1] > rsi_troughs[-2] - self.rsi_divergence_threshold):
                    bullish_divergence[i] = 1
            
            # Bearish divergence: Higher high in price, lower high in RSI
            if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
                if (price_peaks[-1] > price_peaks[-2] and 
                    rsi_peaks[-1] < rsi_peaks[-2] + self.rsi_divergence_threshold):
                    bearish_divergence[i] = 1
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }
    
    def _calculate_support_resistance(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate dynamic support and resistance levels"""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        support = np.zeros(len(data))
        resistance = np.zeros(len(data))
        
        for i in range(self.lookback_period, len(data)):
            # Use pivot points method
            pivot = (high[i-1] + low[i-1] + close[i-1]) / 3
            
            # Calculate support and resistance
            support[i] = 2 * pivot - high[i-1]
            resistance[i] = 2 * pivot - low[i-1]
            
            # Alternative: Use recent swing points
            recent_lows = low[i-self.lookback_period:i]
            recent_highs = high[i-self.lookback_period:i]
            
            if len(recent_lows) > 0:
                support[i] = np.percentile(recent_lows, 25)
            if len(recent_highs) > 0:
                resistance[i] = np.percentile(recent_highs, 75)
        
        return {
            'support': support,
            'resistance': resistance
        }
    
    def _detect_candlestick_patterns(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Detect reversal candlestick patterns"""
        open_price = data['open'].values
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        bullish_pattern = np.zeros(len(data))
        bearish_pattern = np.zeros(len(data))
        
        # Detect patterns using TA-Lib
        # Bullish patterns
        hammer = talib.CDLHAMMER(open_price, high, low, close)
        bullish_engulfing = talib.CDLENGULFING(open_price, high, low, close)
        morning_star = talib.CDLMORNINGSTAR(open_price, high, low, close)
        
        # Bearish patterns
        shooting_star = talib.CDLSHOOTINGSTAR(open_price, high, low, close)
        bearish_engulfing = talib.CDLENGULFING(open_price, high, low, close)
        evening_star = talib.CDLEVENINGSTAR(open_price, high, low, close)
        
        # Combine patterns
        for i in range(len(data)):
            if hammer[i] > 0 or bullish_engulfing[i] > 0 or morning_star[i] > 0:
                bullish_pattern[i] = 1
            if shooting_star[i] < 0 or bearish_engulfing[i] < 0 or evening_star[i] < 0:
                bearish_pattern[i] = 1
        
        return {
            'bullish_pattern': bullish_pattern,
            'bearish_pattern': bearish_pattern
        }
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Analyze market structure shifts"""
        close = data['close'].values
        
        # Calculate trend using multiple timeframes
        ema_short = talib.EMA(close, timeperiod=10)
        ema_medium = talib.EMA(close, timeperiod=20)
        ema_long = talib.EMA(close, timeperiod=50)
        
        trend_shift = ['neutral'] * len(data)
        
        for i in range(50, len(data)):
            # Bullish structure shift
            if (ema_short[i] > ema_medium[i] > ema_long[i] and
                ema_short[i-1] <= ema_medium[i-1]):
                trend_shift[i] = 'bullish'
            
            # Bearish structure shift
            elif (ema_short[i] < ema_medium[i] < ema_long[i] and
                  ema_short[i-1] >= ema_medium[i-1]):
                trend_shift[i] = 'bearish'
        
        return {'trend_shift': trend_shift}
    
    def _check_momentum_shift(self, data: pd.DataFrame, idx: int, direction: str) -> bool:
        """Check if momentum is shifting in the specified direction"""
        close = data['close'].values
        
        if idx < 10:
            return False
        
        # Calculate rate of change
        roc = (close[idx] - close[idx-5]) / close[idx-5] * 100
        
        if direction == 'bullish':
            return roc > 0 and close[idx] > close[idx-1]
        else:
            return roc < 0 and close[idx] < close[idx-1]
    
    def _test_support_bounce(self, data: pd.DataFrame, idx: int, 
                            support_resistance: Dict) -> bool:
        """Test if price bounced from support"""
        low = data['low'].iloc[idx]
        close = data['close'].iloc[idx]
        support = support_resistance['support'][idx]
        
        # Check if low touched support and closed above
        tolerance = support * 0.002  # 0.2% tolerance
        return (abs(low - support) < tolerance and 
                close > support and 
                close > data['open'].iloc[idx])
    
    def _test_resistance_rejection(self, data: pd.DataFrame, idx: int,
                                 support_resistance: Dict) -> bool:
        """Test if price rejected from resistance"""
        high = data['high'].iloc[idx]
        close = data['close'].iloc[idx]
        resistance = support_resistance['resistance'][idx]
        
        # Check if high touched resistance and closed below
        tolerance = resistance * 0.002  # 0.2% tolerance
        return (abs(high - resistance) < tolerance and 
                close < resistance and 
                close < data['open'].iloc[idx])
    
    def _find_peaks(self, values: np.ndarray, min_distance: int = 5) -> List[float]:
        """Find peaks in array"""
        peaks = []
        values = np.array(values)
        
        for i in range(min_distance, len(values) - min_distance):
            if values[i] == np.max(values[i-min_distance:i+min_distance+1]):
                peaks.append(values[i])
        
        return peaks
    
    def _find_troughs(self, values: np.ndarray, min_distance: int = 5) -> List[float]:
        """Find troughs in array"""
        troughs = []
        values = np.array(values)
        
        for i in range(min_distance, len(values) - min_distance):
            if values[i] == np.min(values[i-min_distance:i+min_distance+1]):
                troughs.append(values[i])
        
        return troughs
    
    def generate_option_signals(self, data: pd.DataFrame, 
                              signals: Dict[str, np.ndarray],
                              option_chain: Optional[Dict] = None) -> List[ReversalSignal]:
        """
        Generate option trading signals based on reversal detection
        
        Args:
            data: Price data
            signals: Calculated reversal signals
            option_chain: Current option chain data
            
        Returns:
            List of ReversalSignal objects
        """
        reversal_signals = []
        
        for i in range(len(data)):
            if signals['bullish_signals'][i] == 1:
                signal = self._create_bullish_option_signal(data, i, signals, option_chain)
                if signal:
                    reversal_signals.append(signal)
            
            elif signals['bearish_signals'][i] == 1:
                signal = self._create_bearish_option_signal(data, i, signals, option_chain)
                if signal:
                    reversal_signals.append(signal)
        
        return reversal_signals
    
    def _create_bullish_option_signal(self, data: pd.DataFrame, idx: int,
                                    signals: Dict, option_chain: Optional[Dict]) -> Optional[ReversalSignal]:
        """Create bullish option signal (Call buying)"""
        current_price = data['close'].iloc[idx]
        
        # Calculate targets and stops
        atr = talib.ATR(data['high'].values, data['low'].values, 
                       data['close'].values, timeperiod=14)
        
        if np.isnan(atr[idx]):
            return None
        
        stop_loss = current_price - (2 * atr[idx])
        target_price = current_price + (3 * atr[idx])  # 1.5:1 risk-reward
        risk_reward = (target_price - current_price) / (current_price - stop_loss)
        
        return ReversalSignal(
            timestamp=data.index[idx],
            signal_type='bullish',
            strength=signals['signal_strength'][idx],
            price=current_price,
            volume=data['volume'].iloc[idx],
            confluence_score=signals['confluence_score'][idx],
            risk_reward=risk_reward,
            stop_loss=stop_loss,
            target_price=target_price
        )
    
    def _create_bearish_option_signal(self, data: pd.DataFrame, idx: int,
                                    signals: Dict, option_chain: Optional[Dict]) -> Optional[ReversalSignal]:
        """Create bearish option signal (Put buying)"""
        current_price = data['close'].iloc[idx]
        
        # Calculate targets and stops
        atr = talib.ATR(data['high'].values, data['low'].values, 
                       data['close'].values, timeperiod=14)
        
        if np.isnan(atr[idx]):
            return None
        
        stop_loss = current_price + (2 * atr[idx])
        target_price = current_price - (3 * atr[idx])  # 1.5:1 risk-reward
        risk_reward = (current_price - target_price) / (stop_loss - current_price)
        
        return ReversalSignal(
            timestamp=data.index[idx],
            signal_type='bearish',
            strength=signals['signal_strength'][idx],
            price=current_price,
            volume=data['volume'].iloc[idx],
            confluence_score=signals['confluence_score'][idx],
            risk_reward=risk_reward,
            stop_loss=stop_loss,
            target_price=target_price
        )