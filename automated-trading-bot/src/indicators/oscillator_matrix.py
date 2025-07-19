"""
Oscillator Matrix - Inspired by LuxAlgo
Combines multiple oscillators to create a comprehensive market view
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
try:
    import talib
except ImportError:
    from . import talib_mock as talib


@dataclass
class OscillatorSignal:
    """Container for oscillator matrix signals"""
    timestamp: pd.Timestamp
    oscillator_values: Dict[str, float]
    composite_score: float  # -100 to +100
    signal_strength: str    # 'strong_buy', 'buy', 'neutral', 'sell', 'strong_sell'
    momentum_direction: str # 'bullish', 'bearish', 'neutral'
    divergences: List[str]  # List of oscillators showing divergence


class OscillatorMatrix:
    """
    Multi-oscillator analysis system
    Combines RSI, MACD, Stochastic, CCI, Williams %R, and custom oscillators
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Oscillator Matrix
        
        Args:
            config: Configuration for individual oscillators
        """
        self.config = config or self._default_config()
        
        # Oscillator weights for composite score
        self.weights = {
            'rsi': 0.20,
            'macd': 0.20,
            'stochastic': 0.15,
            'cci': 0.15,
            'williams_r': 0.10,
            'momentum': 0.10,
            'roc': 0.10
        }
        
        # Signal thresholds
        self.thresholds = {
            'strong_buy': -70,
            'buy': -30,
            'sell': 30,
            'strong_sell': 70
        }
        
    def _default_config(self) -> Dict:
        """Default configuration for oscillators"""
        return {
            'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'stochastic': {'k_period': 14, 'd_period': 3, 'overbought': 80, 'oversold': 20},
            'cci': {'period': 20, 'overbought': 100, 'oversold': -100},
            'williams_r': {'period': 14, 'overbought': -20, 'oversold': -80},
            'momentum': {'period': 10},
            'roc': {'period': 12}
        }
    
    def calculate_all_oscillators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all oscillators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with oscillator values
        """
        oscillators = pd.DataFrame(index=data.index)
        
        # 1. RSI
        oscillators['rsi'] = self._calculate_rsi(data['close'])
        oscillators['rsi_normalized'] = self._normalize_oscillator(
            oscillators['rsi'], 0, 100, -100, 100
        )
        
        # 2. MACD
        macd_line, signal_line, histogram = self._calculate_macd(data['close'])
        oscillators['macd_histogram'] = histogram
        oscillators['macd_normalized'] = self._normalize_macd(histogram, data['close'])
        
        # 3. Stochastic
        k_percent, d_percent = self._calculate_stochastic(data)
        oscillators['stochastic_k'] = k_percent
        oscillators['stochastic_d'] = d_percent
        oscillators['stochastic_normalized'] = self._normalize_oscillator(
            k_percent, 0, 100, -100, 100
        )
        
        # 4. CCI (Commodity Channel Index)
        oscillators['cci'] = self._calculate_cci(data)
        oscillators['cci_normalized'] = self._normalize_cci(oscillators['cci'])
        
        # 5. Williams %R
        oscillators['williams_r'] = self._calculate_williams_r(data)
        oscillators['williams_r_normalized'] = self._normalize_oscillator(
            oscillators['williams_r'], -100, 0, -100, 100
        )
        
        # 6. Momentum
        oscillators['momentum'] = self._calculate_momentum(data['close'])
        oscillators['momentum_normalized'] = self._normalize_momentum(
            oscillators['momentum'], data['close']
        )
        
        # 7. Rate of Change (ROC)
        oscillators['roc'] = self._calculate_roc(data['close'])
        oscillators['roc_normalized'] = self._normalize_roc(oscillators['roc'])
        
        # 8. Calculate composite score
        oscillators['composite_score'] = self._calculate_composite_score(oscillators)
        
        return oscillators
    
    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI"""
        return talib.RSI(prices.values, timeperiod=self.config['rsi']['period'])
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD"""
        macd_config = self.config['macd']
        return talib.MACD(
            prices.values,
            fastperiod=macd_config['fast'],
            slowperiod=macd_config['slow'],
            signalperiod=macd_config['signal']
        )
    
    def _calculate_stochastic(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic oscillator"""
        stoch_config = self.config['stochastic']
        k, d = talib.STOCH(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            fastk_period=stoch_config['k_period'],
            slowk_period=stoch_config['d_period'],
            slowd_period=stoch_config['d_period']
        )
        return k, d
    
    def _calculate_cci(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Commodity Channel Index"""
        period = self.config['cci']['period']
        cci_values = talib.CCI(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            timeperiod=period
        )
        return pd.Series(cci_values, index=data.index)
    
    def _calculate_williams_r(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Williams %R"""
        period = self.config['williams_r']['period']
        willr = talib.WILLR(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            timeperiod=period
        )
        return pd.Series(willr, index=data.index)
    
    def _calculate_momentum(self, prices: pd.Series) -> pd.Series:
        """Calculate Momentum"""
        period = self.config['momentum']['period']
        return pd.Series(talib.MOM(prices.values, timeperiod=period), index=prices.index)
    
    def _calculate_roc(self, prices: pd.Series) -> pd.Series:
        """Calculate Rate of Change"""
        period = self.config['roc']['period']
        return pd.Series(talib.ROC(prices.values, timeperiod=period), index=prices.index)
    
    def _normalize_oscillator(self, values: pd.Series, min_val: float, max_val: float,
                            new_min: float = -100, new_max: float = 100) -> pd.Series:
        """Normalize oscillator values to a common scale"""
        normalized = ((values - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
        return normalized
    
    def _normalize_macd(self, histogram: np.ndarray, prices: pd.Series) -> pd.Series:
        """Normalize MACD histogram relative to price"""
        # Use percentage of price for normalization
        price_pct = (histogram / prices.values) * 100
        # Cap at reasonable values
        return pd.Series(np.clip(price_pct * 10, -100, 100), index=prices.index)
    
    def _normalize_cci(self, cci: pd.Series) -> pd.Series:
        """Normalize CCI values"""
        # CCI typically ranges from -200 to +200
        return cci.clip(-200, 200) / 2
    
    def _normalize_momentum(self, momentum: pd.Series, prices: pd.Series) -> pd.Series:
        """Normalize momentum relative to price"""
        # Convert to percentage change
        pct_change = (momentum / prices.shift(self.config['momentum']['period'])) * 100
        return pct_change.clip(-10, 10) * 10  # Scale to -100 to 100
    
    def _normalize_roc(self, roc: pd.Series) -> pd.Series:
        """Normalize ROC values"""
        # ROC is already in percentage, just scale
        return roc.clip(-20, 20) * 5  # Scale to -100 to 100
    
    def _calculate_composite_score(self, oscillators: pd.DataFrame) -> pd.Series:
        """
        Calculate weighted composite score from all oscillators
        
        Args:
            oscillators: DataFrame with normalized oscillator values
            
        Returns:
            Composite score series (-100 to +100)
        """
        composite = pd.Series(0, index=oscillators.index)
        
        # Add weighted contributions
        for osc, weight in self.weights.items():
            col_name = f"{osc}_normalized"
            if col_name in oscillators.columns:
                composite += oscillators[col_name].fillna(0) * weight
        
        return composite
    
    def generate_signals(self, data: pd.DataFrame, lookback: int = 20) -> List[OscillatorSignal]:
        """
        Generate trading signals from oscillator matrix
        
        Args:
            data: OHLCV data
            lookback: Periods to look back for divergence
            
        Returns:
            List of oscillator signals
        """
        # Calculate all oscillators
        oscillators = self.calculate_all_oscillators(data)
        
        signals = []
        
        for i in range(lookback, len(data)):
            # Current values
            current_values = {
                'rsi': oscillators['rsi'].iloc[i],
                'macd': oscillators['macd_histogram'].iloc[i],
                'stochastic': oscillators['stochastic_k'].iloc[i],
                'cci': oscillators['cci'].iloc[i],
                'williams_r': oscillators['williams_r'].iloc[i],
                'momentum': oscillators['momentum'].iloc[i],
                'roc': oscillators['roc'].iloc[i]
            }
            
            composite_score = oscillators['composite_score'].iloc[i]
            
            # Determine signal strength
            if composite_score <= self.thresholds['strong_buy']:
                signal_strength = 'strong_buy'
            elif composite_score <= self.thresholds['buy']:
                signal_strength = 'buy'
            elif composite_score >= self.thresholds['strong_sell']:
                signal_strength = 'strong_sell'
            elif composite_score >= self.thresholds['sell']:
                signal_strength = 'sell'
            else:
                signal_strength = 'neutral'
            
            # Determine momentum direction
            momentum_direction = self._get_momentum_direction(oscillators, i)
            
            # Check for divergences
            divergences = self._check_divergences(data, oscillators, i, lookback)
            
            # Create signal if not neutral or if divergences exist
            if signal_strength != 'neutral' or divergences:
                signals.append(OscillatorSignal(
                    timestamp=data.index[i],
                    oscillator_values=current_values,
                    composite_score=composite_score,
                    signal_strength=signal_strength,
                    momentum_direction=momentum_direction,
                    divergences=divergences
                ))
        
        return signals
    
    def _get_momentum_direction(self, oscillators: pd.DataFrame, idx: int) -> str:
        """
        Determine overall momentum direction
        
        Args:
            oscillators: DataFrame with oscillator values
            idx: Current index
            
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if idx < 5:
            return 'neutral'
        
        # Check recent trend in composite score
        recent_scores = oscillators['composite_score'].iloc[idx-5:idx+1]
        score_change = recent_scores.iloc[-1] - recent_scores.iloc[0]
        
        # Check individual oscillator trends
        bullish_count = 0
        bearish_count = 0
        
        for osc in ['rsi', 'macd_histogram', 'momentum', 'roc']:
            if osc in oscillators.columns:
                recent = oscillators[osc].iloc[idx-5:idx+1]
                if len(recent.dropna()) >= 3:
                    if recent.iloc[-1] > recent.iloc[0]:
                        bullish_count += 1
                    else:
                        bearish_count += 1
        
        # Determine direction
        if score_change > 20 or bullish_count >= 3:
            return 'bullish'
        elif score_change < -20 or bearish_count >= 3:
            return 'bearish'
        else:
            return 'neutral'
    
    def _check_divergences(self, data: pd.DataFrame, oscillators: pd.DataFrame,
                          idx: int, lookback: int) -> List[str]:
        """
        Check for divergences between price and oscillators
        
        Args:
            data: Price data
            oscillators: Oscillator values
            idx: Current index
            lookback: Bars to look back
            
        Returns:
            List of oscillators showing divergence
        """
        divergences = []
        
        if idx < lookback:
            return divergences
        
        # Get price trend
        price_start = data['close'].iloc[idx-lookback]
        price_end = data['close'].iloc[idx]
        price_trend = 'up' if price_end > price_start else 'down'
        
        # Check each oscillator for divergence
        for osc in ['rsi', 'macd_histogram', 'stochastic_k', 'momentum']:
            if osc in oscillators.columns:
                osc_start = oscillators[osc].iloc[idx-lookback]
                osc_end = oscillators[osc].iloc[idx]
                
                if pd.notna(osc_start) and pd.notna(osc_end):
                    osc_trend = 'up' if osc_end > osc_start else 'down'
                    
                    # Divergence exists if trends don't match
                    if price_trend != osc_trend:
                        divergences.append(f"{osc}_divergence")
        
        return divergences
    
    def get_market_condition(self, oscillators: pd.DataFrame, idx: int) -> Dict[str, str]:
        """
        Analyze overall market condition based on oscillator matrix
        
        Args:
            oscillators: DataFrame with oscillator values
            idx: Current index
            
        Returns:
            Dictionary with market condition analysis
        """
        if idx < 0 or idx >= len(oscillators):
            return {'condition': 'unknown'}
        
        composite = oscillators['composite_score'].iloc[idx]
        
        # Count oscillators in different zones
        overbought_count = 0
        oversold_count = 0
        
        # Check RSI
        if 'rsi' in oscillators.columns and pd.notna(oscillators['rsi'].iloc[idx]):
            if oscillators['rsi'].iloc[idx] > self.config['rsi']['overbought']:
                overbought_count += 1
            elif oscillators['rsi'].iloc[idx] < self.config['rsi']['oversold']:
                oversold_count += 1
        
        # Check Stochastic
        if 'stochastic_k' in oscillators.columns and pd.notna(oscillators['stochastic_k'].iloc[idx]):
            if oscillators['stochastic_k'].iloc[idx] > self.config['stochastic']['overbought']:
                overbought_count += 1
            elif oscillators['stochastic_k'].iloc[idx] < self.config['stochastic']['oversold']:
                oversold_count += 1
        
        # Determine condition
        if composite < -50 and oversold_count >= 2:
            condition = 'extremely_oversold'
        elif composite < -20:
            condition = 'oversold'
        elif composite > 50 and overbought_count >= 2:
            condition = 'extremely_overbought'
        elif composite > 20:
            condition = 'overbought'
        else:
            condition = 'neutral'
        
        return {
            'condition': condition,
            'composite_score': composite,
            'overbought_oscillators': overbought_count,
            'oversold_oscillators': oversold_count
        }
    
    def optimize_weights(self, data: pd.DataFrame, lookback: int = 100) -> Dict[str, float]:
        """
        Optimize oscillator weights based on historical performance
        
        Args:
            data: Historical OHLCV data
            lookback: Periods to analyze
            
        Returns:
            Optimized weights dictionary
        """
        # Calculate oscillators
        oscillators = self.calculate_all_oscillators(data)
        
        # Initialize performance tracking
        oscillator_performance = {osc: 0.0 for osc in self.weights.keys()}
        
        # Test each oscillator's predictive power
        for i in range(lookback, len(data) - 10):
            future_return = (data['close'].iloc[i+10] - data['close'].iloc[i]) / data['close'].iloc[i]
            
            # Check each oscillator's signal
            for osc in self.weights.keys():
                norm_col = f"{osc}_normalized"
                if norm_col in oscillators.columns and pd.notna(oscillators[norm_col].iloc[i]):
                    osc_value = oscillators[norm_col].iloc[i]
                    
                    # Oversold should predict positive returns
                    if osc_value < -30 and future_return > 0:
                        oscillator_performance[osc] += abs(future_return)
                    # Overbought should predict negative returns
                    elif osc_value > 30 and future_return < 0:
                        oscillator_performance[osc] += abs(future_return)
                    # Wrong signal
                    elif (osc_value < -30 and future_return < 0) or (osc_value > 30 and future_return > 0):
                        oscillator_performance[osc] -= abs(future_return) * 0.5
        
        # Normalize to create weights
        total_performance = sum(max(0, perf) for perf in oscillator_performance.values())
        if total_performance > 0:
            optimized_weights = {
                osc: max(0.05, perf / total_performance)  # Minimum 5% weight
                for osc, perf in oscillator_performance.items()
            }
            
            # Ensure weights sum to 1
            weight_sum = sum(optimized_weights.values())
            optimized_weights = {osc: w/weight_sum for osc, w in optimized_weights.items()}
        else:
            # Use default weights if optimization fails
            optimized_weights = self.weights.copy()
        
        return optimized_weights