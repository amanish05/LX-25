"""
Trend Following Indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
try:
    import talib
except ImportError:
    from . import talib_mock as talib
from .base import BaseIndicator, IndicatorResult


class TrendIndicators(BaseIndicator):
    """Collection of trend following indicators"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'ema_periods': [5, 20, 50, 200],
            'sma_periods': [10, 20, 50, 200],
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'adx_period': 14,
            'supertrend_period': 10,
            'supertrend_multiplier': 3
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def _calculate_min_periods(self) -> int:
        """Calculate minimum periods required"""
        return max(
            max(self.params['ema_periods']),
            max(self.params['sma_periods']),
            self.params['macd_slow'] + self.params['macd_signal']
        )
    
    def calculate(self, df: pd.DataFrame) -> List[IndicatorResult]:
        """Calculate all trend indicators"""
        self.validate_data(df)
        results = []
        
        # Moving Averages
        results.extend(self.calculate_moving_averages(df))
        
        # MACD
        results.append(self.calculate_macd(df))
        
        # ADX
        results.append(self.calculate_adx(df))
        
        # Supertrend
        results.append(self.calculate_supertrend(df))
        
        # Trend Strength
        results.append(self.calculate_trend_strength(df))
        
        return results
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> List[IndicatorResult]:
        """Calculate various moving averages"""
        results = []
        close = df['close']
        
        # EMAs
        for period in self.params['ema_periods']:
            if len(df) >= period:
                ema = talib.EMA(close.values, timeperiod=period)
                df[f'EMA_{period}'] = ema
                
                # Generate signal based on price vs EMA
                latest_close = close.iloc[-1]
                latest_ema = ema[-1]
                
                if latest_close > latest_ema:
                    signal = "BUY"
                    strength = min((latest_close - latest_ema) / latest_ema * 100, 1.0)
                else:
                    signal = "SELL"
                    strength = min((latest_ema - latest_close) / latest_ema * 100, 1.0)
                
                results.append(IndicatorResult(
                    name=f"EMA_{period}",
                    value=latest_ema,
                    signal=signal,
                    strength=strength,
                    metadata={
                        'series': pd.Series(ema, index=df.index),
                        'crossover': self._detect_crossover(close, pd.Series(ema, index=df.index))
                    }
                ))
        
        # SMAs
        for period in self.params['sma_periods']:
            if len(df) >= period:
                sma = talib.SMA(close.values, timeperiod=period)
                df[f'SMA_{period}'] = sma
                
                results.append(IndicatorResult(
                    name=f"SMA_{period}",
                    value=sma[-1],
                    metadata={'series': pd.Series(sma, index=df.index)}
                ))
        
        return results
    
    def calculate_macd(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate MACD indicator"""
        close = df['close'].values
        
        macd, signal, hist = talib.MACD(
            close,
            fastperiod=self.params['macd_fast'],
            slowperiod=self.params['macd_slow'],
            signalperiod=self.params['macd_signal']
        )
        
        # Generate signal
        latest_hist = hist[-1]
        prev_hist = hist[-2] if len(hist) > 1 else 0
        
        if latest_hist > 0 and prev_hist <= 0:
            signal_type = "BUY"
            strength = 0.8
        elif latest_hist < 0 and prev_hist >= 0:
            signal_type = "SELL"
            strength = 0.8
        else:
            signal_type = "NEUTRAL"
            strength = abs(latest_hist) / (abs(macd[-1]) + 0.0001)
        
        return IndicatorResult(
            name="MACD",
            value={
                'macd': macd[-1],
                'signal': signal[-1],
                'histogram': hist[-1]
            },
            signal=signal_type,
            strength=strength,
            metadata={
                'macd_series': pd.Series(macd, index=df.index),
                'signal_series': pd.Series(signal, index=df.index),
                'hist_series': pd.Series(hist, index=df.index),
                'divergence': self._detect_divergence(df['close'], pd.Series(macd, index=df.index))
            }
        )
    
    def calculate_adx(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate Average Directional Index"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        adx = talib.ADX(high, low, close, timeperiod=self.params['adx_period'])
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=self.params['adx_period'])
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=self.params['adx_period'])
        
        latest_adx = adx[-1]
        latest_plus_di = plus_di[-1]
        latest_minus_di = minus_di[-1]
        
        # Determine trend strength and direction
        if latest_adx > 25:
            if latest_plus_di > latest_minus_di:
                signal = "BUY"
                strength = min(latest_adx / 50, 1.0)
            else:
                signal = "SELL"
                strength = min(latest_adx / 50, 1.0)
        else:
            signal = "NEUTRAL"
            strength = 0.3
        
        return IndicatorResult(
            name="ADX",
            value=latest_adx,
            signal=signal,
            strength=strength,
            metadata={
                'plus_di': latest_plus_di,
                'minus_di': latest_minus_di,
                'trend_strength': 'strong' if latest_adx > 25 else 'weak',
                'adx_series': pd.Series(adx, index=df.index)
            }
        )
    
    def calculate_supertrend(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate Supertrend indicator"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        period = self.params['supertrend_period']
        multiplier = self.params['supertrend_multiplier']
        
        # Calculate ATR
        atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
        
        # Calculate basic bands
        hl_avg = (high + low) / 2
        
        # Calculate final bands
        up_band = hl_avg + (multiplier * pd.Series(atr, index=df.index))
        dn_band = hl_avg - (multiplier * pd.Series(atr, index=df.index))
        
        # Initialize Supertrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        for i in range(period, len(df)):
            if close.iloc[i] <= up_band.iloc[i]:
                direction.iloc[i] = -1
                supertrend.iloc[i] = up_band.iloc[i]
            else:
                direction.iloc[i] = 1
                supertrend.iloc[i] = dn_band.iloc[i]
            
            # Adjust bands
            if direction.iloc[i] == direction.iloc[i-1]:
                if direction.iloc[i] == -1:
                    supertrend.iloc[i] = min(supertrend.iloc[i], supertrend.iloc[i-1])
                else:
                    supertrend.iloc[i] = max(supertrend.iloc[i], supertrend.iloc[i-1])
        
        # Generate signal
        latest_direction = direction.iloc[-1]
        prev_direction = direction.iloc[-2] if len(direction) > 1 else 0
        
        if latest_direction == 1 and prev_direction == -1:
            signal = "BUY"
            strength = 0.9
        elif latest_direction == -1 and prev_direction == 1:
            signal = "SELL"
            strength = 0.9
        else:
            signal = "BUY" if latest_direction == 1 else "SELL"
            strength = 0.6
        
        return IndicatorResult(
            name="Supertrend",
            value=supertrend.iloc[-1],
            signal=signal,
            strength=strength,
            metadata={
                'direction': latest_direction,
                'series': supertrend,
                'up_band': up_band.iloc[-1],
                'dn_band': dn_band.iloc[-1]
            }
        )
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate overall trend strength composite indicator"""
        close = df['close']
        
        # Multiple timeframe trend alignment
        trends = []
        
        for period in [20, 50, 100]:
            if len(df) >= period:
                sma = talib.SMA(close.values, timeperiod=period)
                trend = 1 if close.iloc[-1] > sma[-1] else -1
                trends.append(trend)
        
        # Calculate trend alignment
        if trends:
            trend_score = sum(trends) / len(trends)
            
            if trend_score > 0.5:
                signal = "BUY"
                strength = trend_score
            elif trend_score < -0.5:
                signal = "SELL"
                strength = abs(trend_score)
            else:
                signal = "NEUTRAL"
                strength = 1 - abs(trend_score)
        else:
            signal = "NEUTRAL"
            strength = 0
            trend_score = 0
        
        # Calculate price momentum
        returns = close.pct_change()
        momentum_5 = returns.iloc[-5:].mean() if len(returns) >= 5 else 0
        momentum_20 = returns.iloc[-20:].mean() if len(returns) >= 20 else 0
        
        return IndicatorResult(
            name="TrendStrength",
            value=trend_score,
            signal=signal,
            strength=strength,
            metadata={
                'aligned_trends': len([t for t in trends if t == trends[0]]) if trends else 0,
                'momentum_5d': momentum_5,
                'momentum_20d': momentum_20,
                'volatility': returns.iloc[-20:].std() if len(returns) >= 20 else 0
            }
        )
    
    def _detect_crossover(self, series1: pd.Series, series2: pd.Series) -> Dict[str, Any]:
        """Detect crossover between two series"""
        if len(series1) < 2 or len(series2) < 2:
            return {'crossover': None, 'bars_since': None}
        
        # Current and previous positions
        curr_pos = series1.iloc[-1] > series2.iloc[-1]
        prev_pos = series1.iloc[-2] > series2.iloc[-2]
        
        if curr_pos and not prev_pos:
            crossover = 'bullish'
        elif not curr_pos and prev_pos:
            crossover = 'bearish'
        else:
            crossover = None
        
        # Find bars since last crossover
        positions = series1 > series2
        changes = positions != positions.shift()
        bars_since = len(positions) - changes[::-1].idxmax() if changes.any() else len(positions)
        
        return {
            'crossover': crossover,
            'bars_since': bars_since,
            'current_spread': abs(series1.iloc[-1] - series2.iloc[-1])
        }
    
    def _detect_divergence(self, price: pd.Series, indicator: pd.Series, 
                          lookback: int = 20) -> Dict[str, Any]:
        """Detect divergence between price and indicator"""
        if len(price) < lookback or len(indicator) < lookback:
            return {'divergence': None, 'strength': 0}
        
        # Get recent peaks and troughs
        price_recent = price.iloc[-lookback:]
        indicator_recent = indicator.iloc[-lookback:]
        
        # Simple divergence detection
        price_trend = 1 if price_recent.iloc[-1] > price_recent.iloc[0] else -1
        indicator_trend = 1 if indicator_recent.iloc[-1] > indicator_recent.iloc[0] else -1
        
        if price_trend > 0 and indicator_trend < 0:
            divergence = 'bearish'
            strength = 0.7
        elif price_trend < 0 and indicator_trend > 0:
            divergence = 'bullish'
            strength = 0.7
        else:
            divergence = None
            strength = 0
        
        return {
            'divergence': divergence,
            'strength': strength,
            'price_change': (price_recent.iloc[-1] - price_recent.iloc[0]) / price_recent.iloc[0],
            'indicator_change': (indicator_recent.iloc[-1] - indicator_recent.iloc[0]) / (abs(indicator_recent.iloc[0]) + 0.0001)
        }