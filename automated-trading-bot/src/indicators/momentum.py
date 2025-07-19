"""
Momentum Indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import talib
from .base import BaseIndicator, IndicatorResult


class MomentumIndicators(BaseIndicator):
    """Collection of momentum indicators"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'rsi_period': 14,
            'stoch_k_period': 14,
            'stoch_d_period': 3,
            'stoch_smooth': 3,
            'cci_period': 20,
            'mfi_period': 14,
            'roc_period': 10,
            'williams_r_period': 14
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def _calculate_min_periods(self) -> int:
        """Calculate minimum periods required"""
        return max(
            self.params['rsi_period'],
            self.params['stoch_k_period'] + self.params['stoch_smooth'],
            self.params['cci_period'],
            self.params['mfi_period']
        )
    
    def calculate(self, df: pd.DataFrame) -> List[IndicatorResult]:
        """Calculate all momentum indicators"""
        self.validate_data(df)
        results = []
        
        # RSI
        results.append(self.calculate_rsi(df))
        
        # Stochastic
        results.append(self.calculate_stochastic(df))
        
        # CCI
        results.append(self.calculate_cci(df))
        
        # MFI
        results.append(self.calculate_mfi(df))
        
        # ROC
        results.append(self.calculate_roc(df))
        
        # Williams %R
        results.append(self.calculate_williams_r(df))
        
        # Momentum Score
        results.append(self.calculate_momentum_score(df, results))
        
        return results
    
    def calculate_rsi(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate Relative Strength Index"""
        close = df['close'].values
        rsi = talib.RSI(close, timeperiod=self.params['rsi_period'])
        
        latest_rsi = rsi[-1]
        
        # Generate signal
        signal, strength = self.generate_signal(latest_rsi, 70, 30)
        
        # Check for divergence
        divergence = self._check_rsi_divergence(df['close'], pd.Series(rsi, index=df.index))
        
        return IndicatorResult(
            name="RSI",
            value=latest_rsi,
            signal=signal,
            strength=strength,
            metadata={
                'series': pd.Series(rsi, index=df.index),
                'overbought': latest_rsi > 70,
                'oversold': latest_rsi < 30,
                'divergence': divergence
            }
        )
    
    def calculate_stochastic(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate Stochastic Oscillator"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        k, d = talib.STOCH(
            high, low, close,
            fastk_period=self.params['stoch_k_period'],
            slowk_period=self.params['stoch_smooth'],
            slowd_period=self.params['stoch_d_period']
        )
        
        latest_k = k[-1]
        latest_d = d[-1]
        
        # Generate signal based on K and D crossover
        if latest_k > latest_d and latest_k < 20:
            signal = "BUY"
            strength = 0.8
        elif latest_k < latest_d and latest_k > 80:
            signal = "SELL"
            strength = 0.8
        else:
            signal, strength = self.generate_signal(latest_k, 80, 20)
        
        return IndicatorResult(
            name="Stochastic",
            value={'k': latest_k, 'd': latest_d},
            signal=signal,
            strength=strength,
            metadata={
                'k_series': pd.Series(k, index=df.index),
                'd_series': pd.Series(d, index=df.index),
                'crossover': self._detect_line_cross(k, d)
            }
        )
    
    def calculate_cci(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate Commodity Channel Index"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        cci = talib.CCI(high, low, close, timeperiod=self.params['cci_period'])
        latest_cci = cci[-1]
        
        # CCI typically ranges from -100 to +100
        if latest_cci > 100:
            signal = "SELL"
            strength = min((latest_cci - 100) / 100, 1.0)
        elif latest_cci < -100:
            signal = "BUY"
            strength = min((abs(latest_cci) - 100) / 100, 1.0)
        else:
            signal = "NEUTRAL"
            strength = 1 - abs(latest_cci) / 100
        
        return IndicatorResult(
            name="CCI",
            value=latest_cci,
            signal=signal,
            strength=strength,
            metadata={
                'series': pd.Series(cci, index=df.index),
                'extreme': abs(latest_cci) > 100,
                'trend': self._calculate_indicator_trend(cci)
            }
        )
    
    def calculate_mfi(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate Money Flow Index"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        mfi = talib.MFI(high, low, close, volume, timeperiod=self.params['mfi_period'])
        latest_mfi = mfi[-1]
        
        # Generate signal (similar to RSI)
        signal, strength = self.generate_signal(latest_mfi, 80, 20)
        
        # Calculate money flow direction
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        money_flow_direction = np.sign(money_flow[-1] - money_flow[-2]) if len(money_flow) > 1 else 0
        
        return IndicatorResult(
            name="MFI",
            value=latest_mfi,
            signal=signal,
            strength=strength,
            metadata={
                'series': pd.Series(mfi, index=df.index),
                'money_flow_direction': money_flow_direction,
                'volume_pressure': 'buying' if latest_mfi > 50 else 'selling'
            }
        )
    
    def calculate_roc(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate Rate of Change"""
        close = df['close'].values
        roc = talib.ROC(close, timeperiod=self.params['roc_period'])
        latest_roc = roc[-1]
        
        # Generate signal based on momentum
        if latest_roc > 5:
            signal = "BUY"
            strength = min(latest_roc / 10, 1.0)
        elif latest_roc < -5:
            signal = "SELL"
            strength = min(abs(latest_roc) / 10, 1.0)
        else:
            signal = "NEUTRAL"
            strength = 1 - abs(latest_roc) / 5
        
        # Calculate acceleration
        roc_change = roc[-1] - roc[-2] if len(roc) > 1 else 0
        
        return IndicatorResult(
            name="ROC",
            value=latest_roc,
            signal=signal,
            strength=strength,
            metadata={
                'series': pd.Series(roc, index=df.index),
                'acceleration': roc_change,
                'momentum_increasing': roc_change > 0
            }
        )
    
    def calculate_williams_r(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate Williams %R"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        williams_r = talib.WILLR(high, low, close, timeperiod=self.params['williams_r_period'])
        latest_wr = williams_r[-1]
        
        # Williams %R ranges from -100 to 0
        # Convert to 0-100 scale for consistency
        normalized_wr = abs(latest_wr)
        
        signal, strength = self.generate_signal(normalized_wr, 80, 20)
        
        return IndicatorResult(
            name="WilliamsR",
            value=latest_wr,
            signal=signal,
            strength=strength,
            metadata={
                'series': pd.Series(williams_r, index=df.index),
                'normalized': normalized_wr,
                'extreme': normalized_wr > 80 or normalized_wr < 20
            }
        )
    
    def calculate_momentum_score(self, df: pd.DataFrame, 
                               individual_results: List[IndicatorResult]) -> IndicatorResult:
        """Calculate composite momentum score from all indicators"""
        
        # Extract signals and strengths
        buy_signals = 0
        sell_signals = 0
        total_strength = 0
        
        indicator_scores = {}
        
        for result in individual_results:
            if result.name == "MomentumScore":
                continue
                
            if result.signal == "BUY":
                buy_signals += result.strength
            elif result.signal == "SELL":
                sell_signals += result.strength
            
            total_strength += result.strength
            indicator_scores[result.name] = {
                'signal': result.signal,
                'strength': result.strength
            }
        
        # Calculate net momentum
        net_momentum = buy_signals - sell_signals
        
        # Determine overall signal
        if net_momentum > 1.5:
            signal = "BUY"
            strength = min(net_momentum / len(individual_results), 1.0)
        elif net_momentum < -1.5:
            signal = "SELL"
            strength = min(abs(net_momentum) / len(individual_results), 1.0)
        else:
            signal = "NEUTRAL"
            strength = 1 - abs(net_momentum) / len(individual_results)
        
        # Calculate momentum quality (agreement between indicators)
        total_indicators = len(individual_results) - 1  # Exclude self
        agreeing_indicators = sum(1 for r in individual_results 
                                if r.signal == signal and r.name != "MomentumScore")
        quality = agreeing_indicators / total_indicators if total_indicators > 0 else 0
        
        return IndicatorResult(
            name="MomentumScore",
            value=net_momentum,
            signal=signal,
            strength=strength,
            metadata={
                'buy_pressure': buy_signals,
                'sell_pressure': sell_signals,
                'quality': quality,
                'indicator_scores': indicator_scores,
                'agreement_ratio': quality
            }
        )
    
    def _check_rsi_divergence(self, price: pd.Series, rsi: pd.Series, 
                             lookback: int = 14) -> Dict[str, Any]:
        """Check for RSI divergence"""
        if len(price) < lookback or len(rsi) < lookback:
            return {'type': None, 'strength': 0}
        
        # Find local peaks and troughs
        price_highs = price.rolling(5).max() == price
        price_lows = price.rolling(5).min() == price
        rsi_highs = rsi.rolling(5).max() == rsi
        rsi_lows = rsi.rolling(5).min() == rsi
        
        # Check for bearish divergence (price high, RSI not confirming)
        recent_price_highs = price[price_highs].tail(2)
        recent_rsi_highs = rsi[rsi_highs].tail(2)
        
        if len(recent_price_highs) >= 2 and len(recent_rsi_highs) >= 2:
            if (recent_price_highs.iloc[-1] > recent_price_highs.iloc[-2] and
                recent_rsi_highs.iloc[-1] < recent_rsi_highs.iloc[-2]):
                return {'type': 'bearish', 'strength': 0.8}
        
        # Check for bullish divergence (price low, RSI not confirming)
        recent_price_lows = price[price_lows].tail(2)
        recent_rsi_lows = rsi[rsi_lows].tail(2)
        
        if len(recent_price_lows) >= 2 and len(recent_rsi_lows) >= 2:
            if (recent_price_lows.iloc[-1] < recent_price_lows.iloc[-2] and
                recent_rsi_lows.iloc[-1] > recent_rsi_lows.iloc[-2]):
                return {'type': 'bullish', 'strength': 0.8}
        
        return {'type': None, 'strength': 0}
    
    def _detect_line_cross(self, line1: np.ndarray, line2: np.ndarray) -> Dict[str, Any]:
        """Detect crossover between two lines"""
        if len(line1) < 2 or len(line2) < 2:
            return {'type': None, 'bars_ago': None}
        
        # Current and previous positions
        curr_above = line1[-1] > line2[-1]
        prev_above = line1[-2] > line2[-2]
        
        if curr_above and not prev_above:
            cross_type = 'bullish'
        elif not curr_above and prev_above:
            cross_type = 'bearish'
        else:
            cross_type = None
        
        return {
            'type': cross_type,
            'value_diff': line1[-1] - line2[-1],
            'bars_ago': 0 if cross_type else None
        }
    
    def _calculate_indicator_trend(self, indicator: np.ndarray, periods: int = 5) -> str:
        """Calculate short-term trend of indicator"""
        if len(indicator) < periods:
            return 'neutral'
        
        recent = indicator[-periods:]
        slope = np.polyfit(range(periods), recent, 1)[0]
        
        if slope > 0.5:
            return 'rising'
        elif slope < -0.5:
            return 'falling'
        else:
            return 'neutral'