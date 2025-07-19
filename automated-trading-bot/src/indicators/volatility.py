"""
Volatility Indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import talib
from .base import BaseIndicator, IndicatorResult


class VolatilityIndicators(BaseIndicator):
    """Collection of volatility indicators"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'atr_period': 14,
            'bb_period': 20,
            'bb_std': 2,
            'keltner_period': 20,
            'keltner_atr_mult': 2,
            'donchian_period': 20,
            'chaikin_period': 10,
            'historical_vol_period': 20
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def _calculate_min_periods(self) -> int:
        """Calculate minimum periods required"""
        return max(
            self.params['atr_period'],
            self.params['bb_period'],
            self.params['keltner_period'],
            self.params['donchian_period'],
            self.params['historical_vol_period']
        )
    
    def calculate(self, df: pd.DataFrame) -> List[IndicatorResult]:
        """Calculate all volatility indicators"""
        self.validate_data(df)
        results = []
        
        # ATR
        results.append(self.calculate_atr(df))
        
        # Bollinger Bands
        results.append(self.calculate_bollinger_bands(df))
        
        # Keltner Channels
        results.append(self.calculate_keltner_channels(df))
        
        # Donchian Channels
        results.append(self.calculate_donchian_channels(df))
        
        # Historical Volatility
        results.append(self.calculate_historical_volatility(df))
        
        # Chaikin Volatility
        results.append(self.calculate_chaikin_volatility(df))
        
        # Volatility Regime
        results.append(self.calculate_volatility_regime(df, results))
        
        return results
    
    def calculate_atr(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate Average True Range"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        atr = talib.ATR(high, low, close, timeperiod=self.params['atr_period'])
        latest_atr = atr[-1]
        
        # Calculate ATR as percentage of price
        atr_pct = (latest_atr / close[-1]) * 100
        
        # Determine volatility level
        if atr_pct > 3:
            volatility = "high"
            strength = min(atr_pct / 5, 1.0)
        elif atr_pct < 1:
            volatility = "low"
            strength = 1 - atr_pct
        else:
            volatility = "medium"
            strength = 0.5
        
        # Calculate ATR expansion/contraction
        atr_change = (atr[-1] - atr[-5]) / atr[-5] if len(atr) > 5 else 0
        
        return IndicatorResult(
            name="ATR",
            value=latest_atr,
            signal="NEUTRAL",  # ATR doesn't give directional signals
            strength=strength,
            metadata={
                'series': pd.Series(atr, index=df.index),
                'atr_pct': atr_pct,
                'volatility_level': volatility,
                'expanding': atr_change > 0.1,
                'contracting': atr_change < -0.1
            }
        )
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate Bollinger Bands"""
        close = df['close'].values
        
        upper, middle, lower = talib.BBANDS(
            close,
            timeperiod=self.params['bb_period'],
            nbdevup=self.params['bb_std'],
            nbdevdn=self.params['bb_std']
        )
        
        latest_close = close[-1]
        latest_upper = upper[-1]
        latest_middle = middle[-1]
        latest_lower = lower[-1]
        
        # Calculate band width and position
        band_width = (latest_upper - latest_lower) / latest_middle
        bb_position = (latest_close - latest_lower) / (latest_upper - latest_lower)
        
        # Generate signal based on band position
        if bb_position > 0.95:
            signal = "SELL"
            strength = 0.8
        elif bb_position < 0.05:
            signal = "BUY"
            strength = 0.8
        else:
            signal = "NEUTRAL"
            strength = 1 - abs(bb_position - 0.5) * 2
        
        # Detect squeeze
        band_width_ma = pd.Series(upper - lower).rolling(20).mean()
        squeeze = band_width < band_width_ma.iloc[-1] * 0.8 if len(band_width_ma) > 0 else False
        
        return IndicatorResult(
            name="BollingerBands",
            value={
                'upper': latest_upper,
                'middle': latest_middle,
                'lower': latest_lower
            },
            signal=signal,
            strength=strength,
            metadata={
                'band_width': band_width,
                'bb_position': bb_position,
                'squeeze': squeeze,
                'upper_series': pd.Series(upper, index=df.index),
                'middle_series': pd.Series(middle, index=df.index),
                'lower_series': pd.Series(lower, index=df.index)
            }
        )
    
    def calculate_keltner_channels(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate Keltner Channels"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Calculate middle line (EMA)
        middle = talib.EMA(close, timeperiod=self.params['keltner_period'])
        
        # Calculate ATR
        atr = talib.ATR(high, low, close, timeperiod=self.params['atr_period'])
        
        # Calculate bands
        upper = middle + (atr * self.params['keltner_atr_mult'])
        lower = middle - (atr * self.params['keltner_atr_mult'])
        
        latest_close = close[-1]
        latest_upper = upper[-1]
        latest_middle = middle[-1]
        latest_lower = lower[-1]
        
        # Calculate position within channels
        kc_position = (latest_close - latest_lower) / (latest_upper - latest_lower)
        
        # Generate signal
        if latest_close > latest_upper:
            signal = "BUY"  # Breakout
            strength = 0.9
        elif latest_close < latest_lower:
            signal = "SELL"  # Breakdown
            strength = 0.9
        else:
            signal = "NEUTRAL"
            strength = 0.5
        
        return IndicatorResult(
            name="KeltnerChannels",
            value={
                'upper': latest_upper,
                'middle': latest_middle,
                'lower': latest_lower
            },
            signal=signal,
            strength=strength,
            metadata={
                'kc_position': kc_position,
                'channel_width': latest_upper - latest_lower,
                'breakout': latest_close > latest_upper,
                'breakdown': latest_close < latest_lower
            }
        )
    
    def calculate_donchian_channels(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate Donchian Channels"""
        period = self.params['donchian_period']
        
        upper = df['high'].rolling(period).max()
        lower = df['low'].rolling(period).min()
        middle = (upper + lower) / 2
        
        latest_close = df['close'].iloc[-1]
        latest_upper = upper.iloc[-1]
        latest_lower = lower.iloc[-1]
        latest_middle = middle.iloc[-1]
        
        # Calculate position
        dc_position = (latest_close - latest_lower) / (latest_upper - latest_lower)
        
        # Check for new highs/lows
        new_high = latest_close >= latest_upper
        new_low = latest_close <= latest_lower
        
        # Generate signal
        if new_high:
            signal = "BUY"
            strength = 0.9
        elif new_low:
            signal = "SELL"
            strength = 0.9
        elif dc_position > 0.8:
            signal = "BUY"
            strength = 0.6
        elif dc_position < 0.2:
            signal = "SELL"
            strength = 0.6
        else:
            signal = "NEUTRAL"
            strength = 0.5
        
        return IndicatorResult(
            name="DonchianChannels",
            value={
                'upper': latest_upper,
                'middle': latest_middle,
                'lower': latest_lower
            },
            signal=signal,
            strength=strength,
            metadata={
                'dc_position': dc_position,
                'channel_width': latest_upper - latest_lower,
                'new_high': new_high,
                'new_low': new_low,
                'days_since_high': self._days_since_extreme(df['high'], latest_upper),
                'days_since_low': self._days_since_extreme(df['low'], latest_lower, is_low=True)
            }
        )
    
    def calculate_historical_volatility(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate Historical Volatility"""
        close = df['close']
        returns = close.pct_change().dropna()
        
        # Calculate rolling volatility
        period = self.params['historical_vol_period']
        rolling_vol = returns.rolling(period).std() * np.sqrt(252)  # Annualized
        
        latest_vol = rolling_vol.iloc[-1]
        
        # Calculate volatility percentile
        vol_percentile = self.rolling_rank(rolling_vol, 252).iloc[-1] if len(rolling_vol) > 252 else 0.5
        
        # Determine volatility regime
        if vol_percentile > 0.8:
            regime = "high"
            strength = vol_percentile
        elif vol_percentile < 0.2:
            regime = "low"
            strength = 1 - vol_percentile
        else:
            regime = "normal"
            strength = 0.5
        
        # Calculate volatility trend
        vol_change = (rolling_vol.iloc[-1] - rolling_vol.iloc[-5]) / rolling_vol.iloc[-5] if len(rolling_vol) > 5 else 0
        
        return IndicatorResult(
            name="HistoricalVolatility",
            value=latest_vol,
            signal="NEUTRAL",
            strength=strength,
            metadata={
                'series': rolling_vol,
                'percentile': vol_percentile,
                'regime': regime,
                'trend': 'increasing' if vol_change > 0.1 else 'decreasing' if vol_change < -0.1 else 'stable',
                'annualized': latest_vol
            }
        )
    
    def calculate_chaikin_volatility(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate Chaikin Volatility"""
        high = df['high']
        low = df['low']
        
        # High-Low spread
        hl_spread = high - low
        
        # EMA of High-Low spread
        period = self.params['chaikin_period']
        ema_spread = talib.EMA(hl_spread.values, timeperiod=period)
        
        # Rate of change of EMA
        roc = ((ema_spread - np.roll(ema_spread, period)) / np.roll(ema_spread, period)) * 100
        
        latest_chaikin = roc[-1] if not np.isnan(roc[-1]) else 0
        
        # Interpret Chaikin Volatility
        if latest_chaikin > 10:
            vol_trend = "expanding"
            strength = min(latest_chaikin / 20, 1.0)
        elif latest_chaikin < -10:
            vol_trend = "contracting"
            strength = min(abs(latest_chaikin) / 20, 1.0)
        else:
            vol_trend = "stable"
            strength = 0.5
        
        return IndicatorResult(
            name="ChaikinVolatility",
            value=latest_chaikin,
            signal="NEUTRAL",
            strength=strength,
            metadata={
                'series': pd.Series(roc, index=df.index),
                'volatility_trend': vol_trend,
                'ema_spread': ema_spread[-1]
            }
        )
    
    def calculate_volatility_regime(self, df: pd.DataFrame, 
                                  individual_results: List[IndicatorResult]) -> IndicatorResult:
        """Determine overall volatility regime"""
        
        # Extract volatility metrics
        atr_result = next((r for r in individual_results if r.name == "ATR"), None)
        bb_result = next((r for r in individual_results if r.name == "BollingerBands"), None)
        hist_vol_result = next((r for r in individual_results if r.name == "HistoricalVolatility"), None)
        
        # Compile regime indicators
        regime_scores = {
            'high_vol': 0,
            'low_vol': 0,
            'expanding': 0,
            'contracting': 0
        }
        
        # ATR analysis
        if atr_result:
            if atr_result.metadata['volatility_level'] == 'high':
                regime_scores['high_vol'] += 1
            elif atr_result.metadata['volatility_level'] == 'low':
                regime_scores['low_vol'] += 1
            
            if atr_result.metadata['expanding']:
                regime_scores['expanding'] += 1
            elif atr_result.metadata['contracting']:
                regime_scores['contracting'] += 1
        
        # Bollinger Band squeeze
        if bb_result and bb_result.metadata['squeeze']:
            regime_scores['contracting'] += 1
            regime_scores['low_vol'] += 1
        
        # Historical volatility regime
        if hist_vol_result:
            if hist_vol_result.metadata['regime'] == 'high':
                regime_scores['high_vol'] += 1
            elif hist_vol_result.metadata['regime'] == 'low':
                regime_scores['low_vol'] += 1
        
        # Determine dominant regime
        if regime_scores['high_vol'] >= 2:
            regime = "high_volatility"
            opportunity = "volatility_selling"
        elif regime_scores['low_vol'] >= 2:
            regime = "low_volatility"
            opportunity = "volatility_buying" if regime_scores['contracting'] > 0 else "range_trading"
        elif regime_scores['expanding'] > regime_scores['contracting']:
            regime = "expanding_volatility"
            opportunity = "trend_following"
        else:
            regime = "normal_volatility"
            opportunity = "mixed_strategies"
        
        # Calculate regime strength
        total_signals = sum(regime_scores.values())
        regime_strength = max(regime_scores.values()) / total_signals if total_signals > 0 else 0.5
        
        return IndicatorResult(
            name="VolatilityRegime",
            value=regime,
            signal="NEUTRAL",
            strength=regime_strength,
            metadata={
                'regime_scores': regime_scores,
                'opportunity': opportunity,
                'suitable_strategies': self._get_suitable_strategies(regime),
                'risk_level': 'high' if regime == 'high_volatility' else 'low' if regime == 'low_volatility' else 'medium'
            }
        )
    
    def _days_since_extreme(self, series: pd.Series, extreme_value: float, 
                           is_low: bool = False) -> int:
        """Calculate days since extreme value was reached"""
        if is_low:
            mask = series == extreme_value
        else:
            mask = series == extreme_value
        
        if mask.any():
            last_extreme_idx = mask[::-1].idxmax()
            return len(series) - series.index.get_loc(last_extreme_idx) - 1
        return len(series)
    
    def _get_suitable_strategies(self, regime: str) -> List[str]:
        """Get suitable strategies for volatility regime"""
        strategies_map = {
            'high_volatility': ['iron_condor', 'short_straddle', 'volatility_arbitrage'],
            'low_volatility': ['volatility_expansion', 'breakout', 'calendar_spread'],
            'expanding_volatility': ['trend_following', 'momentum', 'directional_options'],
            'normal_volatility': ['mean_reversion', 'swing_trading', 'covered_calls']
        }
        return strategies_map.get(regime, ['mixed_strategies'])