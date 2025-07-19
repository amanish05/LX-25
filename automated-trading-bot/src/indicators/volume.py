"""
Volume Indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import talib
from .base import BaseIndicator, IndicatorResult


class VolumeIndicators(BaseIndicator):
    """Collection of volume-based indicators"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'volume_ma_period': 20,
            'obv_period': 14,
            'ad_period': 14,
            'cmf_period': 20,
            'vwap_period': 'D',  # D for daily, H for hourly
            'volume_profile_bins': 50,
            'mfi_period': 14
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def _calculate_min_periods(self) -> int:
        """Calculate minimum periods required"""
        return max(
            self.params['volume_ma_period'],
            self.params['cmf_period'],
            self.params['mfi_period']
        )
    
    def calculate(self, df: pd.DataFrame) -> List[IndicatorResult]:
        """Calculate all volume indicators"""
        self.validate_data(df)
        results = []
        
        # Volume Analysis
        results.append(self.calculate_volume_analysis(df))
        
        # On Balance Volume
        results.append(self.calculate_obv(df))
        
        # Accumulation/Distribution
        results.append(self.calculate_ad_line(df))
        
        # Chaikin Money Flow
        results.append(self.calculate_cmf(df))
        
        # VWAP
        results.append(self.calculate_vwap(df))
        
        # Volume Profile
        results.append(self.calculate_volume_profile(df))
        
        # Volume Trend Score
        results.append(self.calculate_volume_trend_score(df, results))
        
        return results
    
    def calculate_volume_analysis(self, df: pd.DataFrame) -> IndicatorResult:
        """Basic volume analysis with moving average"""
        volume = df['volume']
        close = df['close']
        
        # Volume moving average
        vol_ma = talib.SMA(volume.values, timeperiod=self.params['volume_ma_period'])
        
        latest_volume = volume.iloc[-1]
        latest_vol_ma = vol_ma[-1]
        
        # Volume ratio
        vol_ratio = latest_volume / latest_vol_ma if latest_vol_ma > 0 else 1
        
        # Price-Volume trend
        price_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] if len(close) > 1 else 0
        
        # Determine signal based on price-volume relationship
        if vol_ratio > 1.5:
            if price_change > 0:
                signal = "BUY"
                strength = min(vol_ratio / 2, 1.0)
                interpretation = "bullish_surge"
            else:
                signal = "SELL"
                strength = min(vol_ratio / 2, 1.0)
                interpretation = "bearish_surge"
        elif vol_ratio < 0.5:
            signal = "NEUTRAL"
            strength = 0.3
            interpretation = "low_interest"
        else:
            signal = "NEUTRAL"
            strength = 0.5
            interpretation = "normal"
        
        # Calculate volume trend
        vol_trend = self._calculate_trend(pd.Series(volume.values[-10:]))
        
        return IndicatorResult(
            name="VolumeAnalysis",
            value=latest_volume,
            signal=signal,
            strength=strength,
            metadata={
                'volume_ma': latest_vol_ma,
                'volume_ratio': vol_ratio,
                'interpretation': interpretation,
                'volume_trend': vol_trend,
                'price_volume_confirmation': price_change * vol_ratio > 0,
                'average_volume': volume.mean()
            }
        )
    
    def calculate_obv(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate On Balance Volume"""
        close = df['close'].values
        volume = df['volume'].values
        
        obv = talib.OBV(close, volume)
        
        # Calculate OBV trend
        obv_ma = talib.SMA(obv, timeperiod=self.params['obv_period'])
        
        latest_obv = obv[-1]
        latest_obv_ma = obv_ma[-1]
        
        # Generate signal based on OBV trend
        if latest_obv > latest_obv_ma:
            signal = "BUY"
            strength = min(abs(latest_obv - latest_obv_ma) / abs(latest_obv_ma) * 10, 1.0)
        else:
            signal = "SELL"
            strength = min(abs(latest_obv - latest_obv_ma) / abs(latest_obv_ma) * 10, 1.0)
        
        # Check for divergence
        price_trend = self._calculate_trend(df['close'].iloc[-20:])
        obv_trend = self._calculate_trend(pd.Series(obv[-20:]))
        divergence = self._check_divergence(price_trend, obv_trend)
        
        return IndicatorResult(
            name="OBV",
            value=latest_obv,
            signal=signal,
            strength=strength,
            metadata={
                'obv_ma': latest_obv_ma,
                'obv_series': pd.Series(obv, index=df.index),
                'divergence': divergence,
                'trend_strength': abs(obv_trend)
            }
        )
    
    def calculate_ad_line(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate Accumulation/Distribution Line"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        ad = talib.AD(high, low, close, volume)
        
        # Calculate AD trend
        ad_ma = talib.SMA(ad, timeperiod=self.params['ad_period'])
        
        latest_ad = ad[-1]
        latest_ad_ma = ad_ma[-1]
        
        # Generate signal
        if latest_ad > latest_ad_ma:
            signal = "BUY"
            strength = 0.7
            pressure = "accumulation"
        else:
            signal = "SELL"
            strength = 0.7
            pressure = "distribution"
        
        # Calculate rate of change
        ad_roc = (ad[-1] - ad[-5]) / abs(ad[-5]) if len(ad) > 5 and ad[-5] != 0 else 0
        
        return IndicatorResult(
            name="A/D Line",
            value=latest_ad,
            signal=signal,
            strength=strength,
            metadata={
                'ad_ma': latest_ad_ma,
                'pressure': pressure,
                'rate_of_change': ad_roc,
                'ad_series': pd.Series(ad, index=df.index)
            }
        )
    
    def calculate_cmf(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate Chaikin Money Flow"""
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Money Flow Multiplier
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        mf_multiplier = mf_multiplier.fillna(0)
        
        # Money Flow Volume
        mf_volume = mf_multiplier * volume
        
        # CMF
        period = self.params['cmf_period']
        cmf = mf_volume.rolling(period).sum() / volume.rolling(period).sum()
        
        latest_cmf = cmf.iloc[-1]
        
        # Generate signal based on CMF value
        if latest_cmf > 0.05:
            signal = "BUY"
            strength = min(latest_cmf * 5, 1.0)
        elif latest_cmf < -0.05:
            signal = "SELL"
            strength = min(abs(latest_cmf) * 5, 1.0)
        else:
            signal = "NEUTRAL"
            strength = 0.5
        
        # Determine money flow
        if latest_cmf > 0:
            flow = "inflow"
        else:
            flow = "outflow"
        
        return IndicatorResult(
            name="CMF",
            value=latest_cmf,
            signal=signal,
            strength=strength,
            metadata={
                'money_flow': flow,
                'flow_strength': abs(latest_cmf),
                'cmf_series': cmf,
                'threshold_crossed': abs(latest_cmf) > 0.05
            }
        )
    
    def calculate_vwap(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate Volume Weighted Average Price"""
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Typical price
        typical_price = (high + low + close) / 3
        
        # VWAP calculation (using daily reset for intraday)
        # Detect session breaks (simplified - assumes daily data or needs timestamp)
        if 'timestamp' in df.index.names or isinstance(df.index, pd.DatetimeIndex):
            df_copy = df.copy()
            df_copy['date'] = df.index.date
            
            vwap_values = []
            for date, group in df_copy.groupby('date'):
                tp = group['typical_price'] = (group['high'] + group['low'] + group['close']) / 3
                cumulative_tp_vol = (tp * group['volume']).cumsum()
                cumulative_vol = group['volume'].cumsum()
                vwap = cumulative_tp_vol / cumulative_vol
                vwap_values.extend(vwap.values)
            
            vwap = pd.Series(vwap_values, index=df.index)
        else:
            # Simple VWAP for non-time series data
            cumulative_tp_vol = (typical_price * volume).cumsum()
            cumulative_vol = volume.cumsum()
            vwap = cumulative_tp_vol / cumulative_vol
        
        latest_vwap = vwap.iloc[-1]
        latest_close = close.iloc[-1]
        
        # Generate signal based on price vs VWAP
        price_vs_vwap = (latest_close - latest_vwap) / latest_vwap
        
        if price_vs_vwap > 0.01:
            signal = "BUY"
            strength = min(price_vs_vwap * 50, 1.0)
        elif price_vs_vwap < -0.01:
            signal = "SELL"
            strength = min(abs(price_vs_vwap) * 50, 1.0)
        else:
            signal = "NEUTRAL"
            strength = 0.5
        
        # Calculate VWAP bands (1 and 2 standard deviations)
        vwap_std = (typical_price - vwap).rolling(20).std()
        upper_band_1 = vwap + vwap_std
        lower_band_1 = vwap - vwap_std
        upper_band_2 = vwap + 2 * vwap_std
        lower_band_2 = vwap - 2 * vwap_std
        
        return IndicatorResult(
            name="VWAP",
            value=latest_vwap,
            signal=signal,
            strength=strength,
            metadata={
                'price_position': price_vs_vwap,
                'vwap_series': vwap,
                'upper_band_1': upper_band_1.iloc[-1],
                'lower_band_1': lower_band_1.iloc[-1],
                'upper_band_2': upper_band_2.iloc[-1],
                'lower_band_2': lower_band_2.iloc[-1],
                'is_support': latest_close > latest_vwap,
                'is_resistance': latest_close < latest_vwap
            }
        )
    
    def calculate_volume_profile(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate Volume Profile (simplified version)"""
        close = df['close']
        volume = df['volume']
        
        # Create price bins
        price_min = close.min()
        price_max = close.max()
        bins = np.linspace(price_min, price_max, self.params['volume_profile_bins'])
        
        # Calculate volume at each price level
        volume_profile = {}
        for i in range(len(bins) - 1):
            mask = (close >= bins[i]) & (close < bins[i+1])
            vol_at_level = volume[mask].sum()
            volume_profile[f"{bins[i]:.2f}-{bins[i+1]:.2f}"] = vol_at_level
        
        # Find Point of Control (POC) - price level with highest volume
        poc_range = max(volume_profile, key=volume_profile.get)
        poc_value = np.mean([float(x) for x in poc_range.split('-')])
        
        # Value Area (70% of volume)
        sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        total_volume = sum(volume_profile.values())
        value_area_volume = 0
        value_area_levels = []
        
        for level, vol in sorted_levels:
            value_area_volume += vol
            value_area_levels.append(level)
            if value_area_volume >= total_volume * 0.7:
                break
        
        # Current price position relative to POC
        latest_close = close.iloc[-1]
        price_vs_poc = (latest_close - poc_value) / poc_value
        
        # Generate signal
        if abs(price_vs_poc) < 0.01:
            signal = "NEUTRAL"
            strength = 0.8  # High probability of mean reversion
        elif price_vs_poc > 0.02:
            signal = "SELL"
            strength = 0.6
        elif price_vs_poc < -0.02:
            signal = "BUY"
            strength = 0.6
        else:
            signal = "NEUTRAL"
            strength = 0.5
        
        return IndicatorResult(
            name="VolumeProfile",
            value=poc_value,
            signal=signal,
            strength=strength,
            metadata={
                'point_of_control': poc_value,
                'value_area_high': max([float(x.split('-')[1]) for x in value_area_levels]),
                'value_area_low': min([float(x.split('-')[0]) for x in value_area_levels]),
                'price_vs_poc': price_vs_poc,
                'volume_distribution': volume_profile,
                'is_balanced': len(value_area_levels) < len(volume_profile) * 0.3
            }
        )
    
    def calculate_volume_trend_score(self, df: pd.DataFrame, 
                                   individual_results: List[IndicatorResult]) -> IndicatorResult:
        """Calculate composite volume trend score"""
        
        # Extract individual indicator signals
        volume_signals = {}
        buy_pressure = 0
        sell_pressure = 0
        
        for result in individual_results:
            if result.name == "VolumeTrendScore":
                continue
            
            volume_signals[result.name] = {
                'signal': result.signal,
                'strength': result.strength
            }
            
            if result.signal == "BUY":
                buy_pressure += result.strength
            elif result.signal == "SELL":
                sell_pressure += result.strength
        
        # Calculate net pressure
        net_pressure = buy_pressure - sell_pressure
        total_indicators = len(volume_signals)
        
        # Determine overall signal
        if net_pressure > 1:
            signal = "BUY"
            strength = min(net_pressure / total_indicators, 1.0)
        elif net_pressure < -1:
            signal = "SELL"
            strength = min(abs(net_pressure) / total_indicators, 1.0)
        else:
            signal = "NEUTRAL"
            strength = 0.5
        
        # Calculate volume trend quality
        agreeing_indicators = sum(1 for v in volume_signals.values() 
                                if v['signal'] == signal)
        quality = agreeing_indicators / total_indicators if total_indicators > 0 else 0
        
        # Determine market participation
        volume_analysis = next((r for r in individual_results if r.name == "VolumeAnalysis"), None)
        if volume_analysis:
            participation = volume_analysis.metadata.get('interpretation', 'normal')
        else:
            participation = 'unknown'
        
        return IndicatorResult(
            name="VolumeTrendScore",
            value=net_pressure,
            signal=signal,
            strength=strength,
            metadata={
                'buy_pressure': buy_pressure,
                'sell_pressure': sell_pressure,
                'quality': quality,
                'participation': participation,
                'indicator_signals': volume_signals,
                'recommendation': self._get_volume_recommendation(signal, quality, participation)
            }
        )
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate trend direction and strength"""
        if len(series) < 2:
            return 0
        
        # Linear regression slope
        x = np.arange(len(series))
        slope = np.polyfit(x, series.values, 1)[0]
        
        # Normalize by series mean
        mean_val = series.mean()
        if mean_val != 0:
            normalized_slope = slope / mean_val
        else:
            normalized_slope = 0
        
        return normalized_slope
    
    def _check_divergence(self, price_trend: float, indicator_trend: float) -> str:
        """Check for divergence between price and indicator"""
        if price_trend > 0.01 and indicator_trend < -0.01:
            return "bearish_divergence"
        elif price_trend < -0.01 and indicator_trend > 0.01:
            return "bullish_divergence"
        else:
            return "no_divergence"
    
    def _get_volume_recommendation(self, signal: str, quality: float, 
                                 participation: str) -> str:
        """Get trading recommendation based on volume analysis"""
        if signal == "BUY" and quality > 0.7 and participation in ['bullish_surge', 'normal']:
            return "strong_buy"
        elif signal == "SELL" and quality > 0.7 and participation in ['bearish_surge', 'normal']:
            return "strong_sell"
        elif participation == 'low_interest':
            return "avoid_trading"
        elif quality < 0.5:
            return "wait_for_confirmation"
        else:
            return "moderate_" + signal.lower() if signal != "NEUTRAL" else "neutral"