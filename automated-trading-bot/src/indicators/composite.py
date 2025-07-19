"""
Composite Indicators
Combines multiple indicator types for comprehensive analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from .base import BaseIndicator, IndicatorResult
from .trend import TrendIndicators
from .momentum import MomentumIndicators
from .volatility import VolatilityIndicators
from .volume import VolumeIndicators


class CompositeIndicators(BaseIndicator):
    """Composite indicators that combine multiple indicator types"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'use_trend': True,
            'use_momentum': True,
            'use_volatility': True,
            'use_volume': True,
            'signal_threshold': 0.6,  # Minimum agreement for strong signal
            'indicator_weights': {
                'trend': 0.3,
                'momentum': 0.25,
                'volatility': 0.2,
                'volume': 0.25
            }
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
        
        # Initialize sub-indicators
        self.trend_indicators = TrendIndicators(params)
        self.momentum_indicators = MomentumIndicators(params)
        self.volatility_indicators = VolatilityIndicators(params)
        self.volume_indicators = VolumeIndicators(params)
    
    def _calculate_min_periods(self) -> int:
        """Calculate minimum periods required"""
        return max(
            self.trend_indicators._calculate_min_periods(),
            self.momentum_indicators._calculate_min_periods(),
            self.volatility_indicators._calculate_min_periods(),
            self.volume_indicators._calculate_min_periods()
        )
    
    def calculate(self, df: pd.DataFrame) -> List[IndicatorResult]:
        """Calculate all composite indicators"""
        self.validate_data(df)
        results = []
        
        # Calculate individual indicator groups
        all_indicators = {}
        
        if self.params['use_trend']:
            trend_results = self.trend_indicators.calculate(df)
            all_indicators['trend'] = trend_results
        
        if self.params['use_momentum']:
            momentum_results = self.momentum_indicators.calculate(df)
            all_indicators['momentum'] = momentum_results
        
        if self.params['use_volatility']:
            volatility_results = self.volatility_indicators.calculate(df)
            all_indicators['volatility'] = volatility_results
        
        if self.params['use_volume']:
            volume_results = self.volume_indicators.calculate(df)
            all_indicators['volume'] = volume_results
        
        # Calculate composite scores
        results.append(self.calculate_master_signal(all_indicators))
        results.append(self.calculate_market_regime(df, all_indicators))
        results.append(self.calculate_entry_quality(all_indicators))
        results.append(self.calculate_risk_reward_score(df, all_indicators))
        
        return results
    
    def calculate_master_signal(self, all_indicators: Dict[str, List[IndicatorResult]]) -> IndicatorResult:
        """Calculate master trading signal from all indicators"""
        
        weights = self.params['indicator_weights']
        weighted_signals = {
            'BUY': 0,
            'SELL': 0,
            'NEUTRAL': 0
        }
        
        signal_details = {}
        
        # Process each indicator group
        for group_name, indicators in all_indicators.items():
            group_weight = weights.get(group_name, 0.25)
            group_signals = {'BUY': 0, 'SELL': 0, 'NEUTRAL': 0}
            
            for indicator in indicators:
                if indicator.signal in group_signals:
                    group_signals[indicator.signal] += indicator.strength
            
            # Normalize and apply weights
            total_strength = sum(group_signals.values())
            if total_strength > 0:
                for signal_type in group_signals:
                    normalized = group_signals[signal_type] / total_strength
                    weighted_signals[signal_type] += normalized * group_weight
            
            signal_details[group_name] = group_signals
        
        # Determine master signal
        max_signal = max(weighted_signals, key=weighted_signals.get)
        signal_strength = weighted_signals[max_signal]
        
        # Apply threshold
        if signal_strength < self.params['signal_threshold']:
            master_signal = "NEUTRAL"
            confidence = 0.5
        else:
            master_signal = max_signal
            confidence = signal_strength
        
        # Calculate signal quality
        signal_agreement = self._calculate_signal_agreement(all_indicators, master_signal)
        
        return IndicatorResult(
            name="MasterSignal",
            value=master_signal,
            signal=master_signal,
            strength=confidence,
            metadata={
                'weighted_signals': weighted_signals,
                'signal_details': signal_details,
                'agreement_score': signal_agreement,
                'indicators_count': sum(len(ind) for ind in all_indicators.values()),
                'recommendation': self._get_recommendation(master_signal, confidence, signal_agreement)
            }
        )
    
    def calculate_market_regime(self, df: pd.DataFrame, 
                              all_indicators: Dict[str, List[IndicatorResult]]) -> IndicatorResult:
        """Determine current market regime"""
        
        regimes = {
            'trending_up': 0,
            'trending_down': 0,
            'ranging': 0,
            'volatile': 0
        }
        
        # Analyze trend indicators
        if 'trend' in all_indicators:
            for indicator in all_indicators['trend']:
                if indicator.name == "TrendStrength":
                    trend_value = indicator.value
                    if trend_value > 0.5:
                        regimes['trending_up'] += abs(trend_value)
                    elif trend_value < -0.5:
                        regimes['trending_down'] += abs(trend_value)
                    else:
                        regimes['ranging'] += 1 - abs(trend_value)
                
                elif indicator.name == "ADX" and indicator.metadata:
                    if indicator.metadata.get('trend_strength') == 'strong':
                        if indicator.signal == "BUY":
                            regimes['trending_up'] += 0.5
                        else:
                            regimes['trending_down'] += 0.5
                    else:
                        regimes['ranging'] += 0.5
        
        # Analyze volatility
        if 'volatility' in all_indicators:
            for indicator in all_indicators['volatility']:
                if indicator.name == "VolatilityRegime":
                    regime = indicator.value
                    if 'high' in regime:
                        regimes['volatile'] += 1
                    elif 'expanding' in regime:
                        regimes['volatile'] += 0.5
        
        # Determine dominant regime
        dominant_regime = max(regimes, key=regimes.get)
        regime_strength = regimes[dominant_regime] / sum(regimes.values()) if sum(regimes.values()) > 0 else 0
        
        # Get suitable strategies for regime
        strategies = self._get_regime_strategies(dominant_regime)
        
        return IndicatorResult(
            name="MarketRegime",
            value=dominant_regime,
            signal="NEUTRAL",
            strength=regime_strength,
            metadata={
                'regime_scores': regimes,
                'suitable_strategies': strategies,
                'market_condition': self._get_market_condition(dominant_regime),
                'risk_level': self._get_regime_risk_level(dominant_regime)
            }
        )
    
    def calculate_entry_quality(self, all_indicators: Dict[str, List[IndicatorResult]]) -> IndicatorResult:
        """Calculate quality score for potential entry"""
        
        quality_factors = {
            'signal_alignment': 0,
            'momentum_confirmation': 0,
            'volume_confirmation': 0,
            'risk_reward': 0,
            'timing': 0
        }
        
        # Check signal alignment across indicator types
        signals_by_type = {}
        for group_name, indicators in all_indicators.items():
            group_signals = [ind.signal for ind in indicators if ind.signal != "NEUTRAL"]
            if group_signals:
                most_common = max(set(group_signals), key=group_signals.count)
                signals_by_type[group_name] = most_common
        
        # Calculate alignment
        if signals_by_type:
            signal_values = list(signals_by_type.values())
            if signal_values:
                most_common_signal = max(set(signal_values), key=signal_values.count)
                alignment = signal_values.count(most_common_signal) / len(signal_values)
                quality_factors['signal_alignment'] = alignment
        
        # Check momentum confirmation
        if 'momentum' in all_indicators:
            momentum_scores = [ind for ind in all_indicators['momentum'] 
                             if ind.name == "MomentumScore"]
            if momentum_scores:
                quality_factors['momentum_confirmation'] = momentum_scores[0].strength
        
        # Check volume confirmation
        if 'volume' in all_indicators:
            volume_scores = [ind for ind in all_indicators['volume'] 
                           if ind.name == "VolumeTrendScore"]
            if volume_scores:
                quality_factors['volume_confirmation'] = volume_scores[0].strength
        
        # Calculate overall quality
        quality_score = np.mean(list(quality_factors.values()))
        
        # Determine entry recommendation
        if quality_score >= 0.7:
            recommendation = "high_quality_entry"
            strength = quality_score
        elif quality_score >= 0.5:
            recommendation = "moderate_quality_entry"
            strength = quality_score
        else:
            recommendation = "low_quality_entry"
            strength = quality_score
        
        return IndicatorResult(
            name="EntryQuality",
            value=quality_score,
            signal="NEUTRAL",
            strength=strength,
            metadata={
                'quality_factors': quality_factors,
                'recommendation': recommendation,
                'best_factor': max(quality_factors, key=quality_factors.get),
                'worst_factor': min(quality_factors, key=quality_factors.get),
                'entry_checklist': self._generate_entry_checklist(quality_factors)
            }
        )
    
    def calculate_risk_reward_score(self, df: pd.DataFrame, 
                                  all_indicators: Dict[str, List[IndicatorResult]]) -> IndicatorResult:
        """Calculate risk-reward score for current setup"""
        
        close = df['close'].iloc[-1]
        
        # Find support and resistance levels
        support_levels = []
        resistance_levels = []
        
        # From volatility indicators (Bollinger, Keltner, etc.)
        if 'volatility' in all_indicators:
            for indicator in all_indicators['volatility']:
                if indicator.name == "BollingerBands" and indicator.metadata:
                    support_levels.append(indicator.metadata['lower_series'].iloc[-1])
                    resistance_levels.append(indicator.metadata['upper_series'].iloc[-1])
                elif indicator.name == "KeltnerChannels" and indicator.value:
                    support_levels.append(indicator.value['lower'])
                    resistance_levels.append(indicator.value['upper'])
        
        # From volume indicators (VWAP bands)
        if 'volume' in all_indicators:
            for indicator in all_indicators['volume']:
                if indicator.name == "VWAP" and indicator.metadata:
                    support_levels.append(indicator.metadata['lower_band_1'])
                    resistance_levels.append(indicator.metadata['upper_band_1'])
        
        # Calculate risk and reward
        if support_levels and resistance_levels:
            nearest_support = min(support_levels, key=lambda x: abs(x - close))
            nearest_resistance = max(resistance_levels, key=lambda x: abs(x - close))
            
            risk = abs(close - nearest_support)
            reward = abs(nearest_resistance - close)
            
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Determine score
            if risk_reward_ratio >= 3:
                score = 0.9
                assessment = "excellent"
            elif risk_reward_ratio >= 2:
                score = 0.7
                assessment = "good"
            elif risk_reward_ratio >= 1.5:
                score = 0.5
                assessment = "acceptable"
            else:
                score = 0.3
                assessment = "poor"
        else:
            risk_reward_ratio = 1
            score = 0.5
            assessment = "unknown"
            nearest_support = close * 0.98
            nearest_resistance = close * 1.02
        
        return IndicatorResult(
            name="RiskRewardScore",
            value=risk_reward_ratio,
            signal="NEUTRAL",
            strength=score,
            metadata={
                'risk_reward_ratio': risk_reward_ratio,
                'assessment': assessment,
                'stop_loss_level': nearest_support,
                'target_level': nearest_resistance,
                'risk_amount': abs(close - nearest_support),
                'reward_amount': abs(nearest_resistance - close),
                'position_sizing_suggestion': self._calculate_position_size(score)
            }
        )
    
    def _calculate_signal_agreement(self, all_indicators: Dict[str, List[IndicatorResult]], 
                                  target_signal: str) -> float:
        """Calculate agreement score for a signal"""
        total_indicators = 0
        agreeing_indicators = 0
        
        for group_name, indicators in all_indicators.items():
            for indicator in indicators:
                if indicator.signal != "NEUTRAL":
                    total_indicators += 1
                    if indicator.signal == target_signal:
                        agreeing_indicators += indicator.strength
        
        return agreeing_indicators / total_indicators if total_indicators > 0 else 0
    
    def _get_recommendation(self, signal: str, confidence: float, agreement: float) -> str:
        """Get trading recommendation based on signal analysis"""
        if signal == "NEUTRAL":
            return "wait_for_better_setup"
        
        if confidence >= 0.8 and agreement >= 0.7:
            return f"strong_{signal.lower()}_recommended"
        elif confidence >= 0.6 and agreement >= 0.5:
            return f"moderate_{signal.lower()}_possible"
        else:
            return "weak_signal_use_caution"
    
    def _get_regime_strategies(self, regime: str) -> List[str]:
        """Get suitable strategies for market regime"""
        strategies_map = {
            'trending_up': ['trend_following', 'momentum_long', 'breakout_buying'],
            'trending_down': ['trend_following_short', 'momentum_short', 'breakdown_selling'],
            'ranging': ['mean_reversion', 'range_trading', 'iron_condor'],
            'volatile': ['straddle', 'strangle', 'volatility_arbitrage']
        }
        return strategies_map.get(regime, ['adaptive_strategies'])
    
    def _get_market_condition(self, regime: str) -> str:
        """Get market condition description"""
        conditions = {
            'trending_up': 'bullish_trend',
            'trending_down': 'bearish_trend',
            'ranging': 'consolidation',
            'volatile': 'high_volatility'
        }
        return conditions.get(regime, 'undefined')
    
    def _get_regime_risk_level(self, regime: str) -> str:
        """Get risk level for regime"""
        risk_levels = {
            'trending_up': 'medium',
            'trending_down': 'medium_high',
            'ranging': 'low',
            'volatile': 'high'
        }
        return risk_levels.get(regime, 'medium')
    
    def _generate_entry_checklist(self, quality_factors: Dict[str, float]) -> List[str]:
        """Generate entry checklist based on quality factors"""
        checklist = []
        
        if quality_factors['signal_alignment'] >= 0.7:
            checklist.append("✓ Strong signal alignment across indicators")
        else:
            checklist.append("✗ Weak signal alignment - wait for confirmation")
        
        if quality_factors['momentum_confirmation'] >= 0.6:
            checklist.append("✓ Momentum confirms direction")
        else:
            checklist.append("✗ Momentum not confirming - be cautious")
        
        if quality_factors['volume_confirmation'] >= 0.6:
            checklist.append("✓ Volume supports the move")
        else:
            checklist.append("✗ Volume not confirming - low conviction")
        
        return checklist
    
    def _calculate_position_size(self, risk_score: float) -> str:
        """Suggest position size based on risk score"""
        if risk_score >= 0.8:
            return "full_position"
        elif risk_score >= 0.6:
            return "three_quarter_position"
        elif risk_score >= 0.4:
            return "half_position"
        else:
            return "quarter_position_or_skip"