"""
Market Regime Detection and Smart Bot Selection System
Analyzes market conditions to select the most appropriate trading bot
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class MarketRegime:
    """Market regime characteristics"""
    regime_type: str  # 'trending_up', 'trending_down', 'ranging', 'volatile', 'calm'
    volatility_level: str  # 'low', 'medium', 'high', 'extreme'
    iv_rank: float  # 0-100
    trend_strength: float  # -1 to 1 (-1 strong down, 0 neutral, 1 strong up)
    market_structure: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0-1
    timestamp: datetime


@dataclass
class BotRecommendation:
    """Bot recommendation based on market regime"""
    bot_name: str
    score: float  # 0-1 suitability score
    reasons: List[str]
    expected_performance: Dict[str, float]
    risk_level: str  # 'low', 'medium', 'high'


class MarketRegimeDetector:
    """
    Detects current market regime and recommends optimal bot selection
    
    Market Regimes:
    1. Trending Up: Strong upward momentum, good for momentum bots
    2. Trending Down: Strong downward momentum, good for momentum bots (puts)
    3. Ranging: Sideways movement, ideal for option selling
    4. Volatile: High volatility with unclear direction, mixed strategies
    5. Calm: Low volatility, potentially setup for volatility expansion
    """
    
    def __init__(self, config_path: str = 'config/bot_selection_config.json'):
        """Initialize market regime detector"""
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Regime detection parameters
        self.lookback_periods = self.config.get('lookback_periods', {
            'short': 20,
            'medium': 50,
            'long': 200
        })
        
        # Bot suitability matrix
        self.bot_suitability = self.config.get('bot_suitability', {
            'trending_up': {
                'momentum_rider': 0.9,
                'volatility_expander': 0.3,
                'short_straddle': 0.2,
                'iron_condor': 0.3
            },
            'trending_down': {
                'momentum_rider': 0.9,
                'volatility_expander': 0.3,
                'short_straddle': 0.2,
                'iron_condor': 0.3
            },
            'ranging': {
                'momentum_rider': 0.3,
                'volatility_expander': 0.4,
                'short_straddle': 0.8,
                'iron_condor': 0.9
            },
            'volatile': {
                'momentum_rider': 0.6,
                'volatility_expander': 0.7,
                'short_straddle': 0.5,
                'iron_condor': 0.4
            },
            'calm': {
                'momentum_rider': 0.4,
                'volatility_expander': 0.9,
                'short_straddle': 0.7,
                'iron_condor': 0.8
            }
        })
        
        # Historical regime tracking
        self.regime_history = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                'lookback_periods': {'short': 20, 'medium': 50, 'long': 200},
                'volatility_thresholds': {'low': 0.5, 'medium': 1.0, 'high': 2.0},
                'trend_thresholds': {'strong': 0.7, 'moderate': 0.4, 'weak': 0.2},
                'iv_rank_thresholds': {'low': 30, 'medium': 50, 'high': 70}
            }
    
    def detect_market_regime(self, data: pd.DataFrame, iv_data: Optional[Dict] = None) -> MarketRegime:
        """
        Detect current market regime from market data
        
        Args:
            data: DataFrame with OHLCV data
            iv_data: Optional IV rank and percentile data
            
        Returns:
            MarketRegime object with current regime characteristics
        """
        try:
            # Calculate technical indicators
            volatility_metrics = self._calculate_volatility_metrics(data)
            trend_metrics = self._calculate_trend_metrics(data)
            structure_metrics = self._calculate_structure_metrics(data)
            
            # Determine regime type
            regime_type = self._determine_regime_type(
                volatility_metrics, trend_metrics, structure_metrics
            )
            
            # Determine volatility level
            volatility_level = self._categorize_volatility(volatility_metrics['current_volatility'])
            
            # Get IV rank (use provided or calculate)
            iv_rank = 50.0  # Default
            if iv_data and 'iv_rank' in iv_data:
                iv_rank = iv_data['iv_rank']
            
            # Calculate confidence
            confidence = self._calculate_regime_confidence(
                volatility_metrics, trend_metrics, structure_metrics
            )
            
            regime = MarketRegime(
                regime_type=regime_type,
                volatility_level=volatility_level,
                iv_rank=iv_rank,
                trend_strength=trend_metrics['trend_strength'],
                market_structure=structure_metrics['structure'],
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.regime_history.append(regime)
            if len(self.regime_history) > 100:
                self.regime_history.pop(0)
            
            self.logger.info(f"Detected market regime: {regime_type} "
                           f"(volatility: {volatility_level}, trend: {trend_metrics['trend_strength']:.2f})")
            
            return regime
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            # Return default neutral regime
            return MarketRegime(
                regime_type='ranging',
                volatility_level='medium',
                iv_rank=50.0,
                trend_strength=0.0,
                market_structure='neutral',
                confidence=0.3,
                timestamp=datetime.now()
            )
    
    def _calculate_volatility_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility metrics"""
        returns = data['close'].pct_change().dropna()
        
        # Historical volatility (different periods)
        vol_short = returns.tail(self.lookback_periods['short']).std() * np.sqrt(252)
        vol_medium = returns.tail(self.lookback_periods['medium']).std() * np.sqrt(252)
        vol_long = returns.tail(self.lookback_periods['long']).std() * np.sqrt(252)
        
        # ATR-based volatility
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift(1))
        low_close = abs(data['low'] - data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        atr_percent = (atr / data['close'].iloc[-1]) * 100
        
        # Volatility regime change
        vol_ratio = vol_short / vol_medium if vol_medium > 0 else 1.0
        vol_expanding = vol_ratio > 1.2
        vol_contracting = vol_ratio < 0.8
        
        return {
            'current_volatility': vol_short,
            'vol_short': vol_short,
            'vol_medium': vol_medium,
            'vol_long': vol_long,
            'atr_percent': atr_percent,
            'vol_expanding': vol_expanding,
            'vol_contracting': vol_contracting,
            'vol_ratio': vol_ratio
        }
    
    def _calculate_trend_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trend metrics"""
        close_prices = data['close']
        
        # Moving averages
        ma_short = close_prices.rolling(self.lookback_periods['short']).mean()
        ma_medium = close_prices.rolling(self.lookback_periods['medium']).mean()
        ma_long = close_prices.rolling(self.lookback_periods['long']).mean()
        
        # Current price position relative to MAs
        current_price = close_prices.iloc[-1]
        above_short = current_price > ma_short.iloc[-1]
        above_medium = current_price > ma_medium.iloc[-1]
        above_long = current_price > ma_long.iloc[-1] if len(data) > self.lookback_periods['long'] else True
        
        # MA alignment
        ma_bullish = ma_short.iloc[-1] > ma_medium.iloc[-1]
        ma_bearish = ma_short.iloc[-1] < ma_medium.iloc[-1]
        
        # Trend strength (ADX-like calculation)
        high = data['high']
        low = data['low']
        close = data['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        atr14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean().iloc[-1]
        
        # Normalize ADX to trend strength (-1 to 1)
        if plus_di.iloc[-1] > minus_di.iloc[-1]:
            trend_strength = min(adx / 50, 1.0)  # Bullish
        else:
            trend_strength = -min(adx / 50, 1.0)  # Bearish
        
        # Recent price action
        recent_return = (current_price / close_prices.iloc[-self.lookback_periods['short']] - 1) * 100
        
        return {
            'trend_strength': trend_strength,
            'above_short_ma': above_short,
            'above_medium_ma': above_medium,
            'above_long_ma': above_long,
            'ma_bullish': ma_bullish,
            'ma_bearish': ma_bearish,
            'adx': adx,
            'recent_return': recent_return,
            'trending': abs(trend_strength) > 0.4
        }
    
    def _calculate_structure_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market structure metrics"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Identify swing points
        swing_highs = high.rolling(5).max() == high
        swing_lows = low.rolling(5).min() == low
        
        # Recent swings
        recent_highs = high[swing_highs].tail(3).values
        recent_lows = low[swing_lows].tail(3).values
        
        # Structure determination
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            # Higher highs and higher lows = Bullish
            if recent_highs[-1] > recent_highs[-2] and recent_lows[-1] > recent_lows[-2]:
                structure = 'bullish'
            # Lower highs and lower lows = Bearish
            elif recent_highs[-1] < recent_highs[-2] and recent_lows[-1] < recent_lows[-2]:
                structure = 'bearish'
            else:
                structure = 'neutral'
        else:
            structure = 'neutral'
        
        # Range detection
        recent_high = high.tail(self.lookback_periods['short']).max()
        recent_low = low.tail(self.lookback_periods['short']).min()
        range_percent = ((recent_high - recent_low) / recent_low) * 100
        
        # Support/Resistance levels
        support_level = low.tail(self.lookback_periods['medium']).min()
        resistance_level = high.tail(self.lookback_periods['medium']).max()
        
        return {
            'structure': structure,
            'range_percent': range_percent,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'in_range': range_percent < 5.0  # Less than 5% range
        }
    
    def _determine_regime_type(self, volatility: Dict, trend: Dict, structure: Dict) -> str:
        """Determine market regime type from metrics"""
        # Strong trend detection
        if abs(trend['trend_strength']) > 0.6:
            if trend['trend_strength'] > 0:
                return 'trending_up'
            else:
                return 'trending_down'
        
        # Range detection
        if structure['in_range'] and abs(trend['trend_strength']) < 0.3:
            if volatility['current_volatility'] < volatility['vol_medium']:
                return 'calm'
            else:
                return 'ranging'
        
        # Volatile market
        if volatility['vol_expanding'] or volatility['current_volatility'] > volatility['vol_long'] * 1.5:
            return 'volatile'
        
        # Default to ranging
        return 'ranging'
    
    def _categorize_volatility(self, volatility: float) -> str:
        """Categorize volatility level"""
        thresholds = self.config.get('volatility_thresholds', {
            'low': 0.1,
            'medium': 0.2,
            'high': 0.3
        })
        
        if volatility < thresholds['low']:
            return 'low'
        elif volatility < thresholds['medium']:
            return 'medium'
        elif volatility < thresholds['high']:
            return 'high'
        else:
            return 'extreme'
    
    def _calculate_regime_confidence(self, volatility: Dict, trend: Dict, structure: Dict) -> float:
        """Calculate confidence in regime detection"""
        confidence_factors = []
        
        # Trend clarity
        if trend['trending']:
            confidence_factors.append(min(abs(trend['trend_strength']), 1.0))
        else:
            confidence_factors.append(0.5)
        
        # MA alignment
        if trend['ma_bullish'] or trend['ma_bearish']:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        # Structure clarity
        if structure['structure'] != 'neutral':
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Volatility consistency
        vol_consistent = abs(volatility['vol_ratio'] - 1.0) < 0.3
        confidence_factors.append(0.8 if vol_consistent else 0.5)
        
        return np.mean(confidence_factors)
    
    def recommend_bots(self, regime: MarketRegime, 
                      available_capital: float = 100000,
                      risk_tolerance: str = 'medium') -> List[BotRecommendation]:
        """
        Recommend bots based on market regime
        
        Args:
            regime: Current market regime
            available_capital: Available trading capital
            risk_tolerance: 'low', 'medium', 'high'
            
        Returns:
            List of bot recommendations sorted by suitability
        """
        recommendations = []
        
        # Get base suitability scores
        regime_scores = self.bot_suitability.get(regime.regime_type, {})
        
        for bot_name, base_score in regime_scores.items():
            # Adjust score based on additional factors
            adjusted_score = base_score
            reasons = []
            
            # IV rank adjustments
            if bot_name == 'short_straddle':
                if regime.iv_rank > 70:
                    adjusted_score *= 1.3
                    reasons.append(f"High IV rank ({regime.iv_rank:.0f}%) favors premium selling")
                elif regime.iv_rank < 30:
                    adjusted_score *= 0.5
                    reasons.append(f"Low IV rank ({regime.iv_rank:.0f}%) unfavorable for straddles")
            
            elif bot_name == 'volatility_expander':
                if regime.iv_rank < 30:
                    adjusted_score *= 1.4
                    reasons.append(f"Low IV rank ({regime.iv_rank:.0f}%) ideal for volatility expansion")
                elif regime.iv_rank > 70:
                    adjusted_score *= 0.4
                    reasons.append(f"High IV rank ({regime.iv_rank:.0f}%) reduces expansion potential")
            
            # Volatility level adjustments
            if regime.volatility_level == 'extreme':
                if bot_name in ['short_straddle', 'iron_condor']:
                    adjusted_score *= 0.6
                    reasons.append("Extreme volatility increases option selling risk")
                elif bot_name == 'momentum_rider':
                    adjusted_score *= 1.1
                    reasons.append("High volatility provides momentum opportunities")
            
            # Risk tolerance adjustments
            risk_levels = {
                'momentum_rider': 'medium',
                'volatility_expander': 'medium',
                'short_straddle': 'high',
                'iron_condor': 'low'
            }
            
            bot_risk = risk_levels.get(bot_name, 'medium')
            if risk_tolerance == 'low' and bot_risk == 'high':
                adjusted_score *= 0.7
                reasons.append("High risk strategy not suitable for low risk tolerance")
            elif risk_tolerance == 'high' and bot_risk == 'low':
                adjusted_score *= 0.8
                reasons.append("Conservative strategy may underutilize risk tolerance")
            
            # Market structure bonus
            if regime.market_structure == 'bullish' and bot_name == 'momentum_rider':
                adjusted_score *= 1.2
                reasons.append("Bullish structure favors momentum strategies")
            elif regime.market_structure == 'neutral' and bot_name in ['short_straddle', 'iron_condor']:
                adjusted_score *= 1.1
                reasons.append("Neutral structure ideal for option selling")
            
            # Expected performance
            expected_performance = self._estimate_performance(
                bot_name, regime, available_capital
            )
            
            # Add main reason based on regime
            if adjusted_score > 0.7:
                reasons.insert(0, f"Highly suitable for {regime.regime_type} market regime")
            elif adjusted_score > 0.5:
                reasons.insert(0, f"Moderately suitable for {regime.regime_type} market regime")
            else:
                reasons.insert(0, f"Limited suitability for {regime.regime_type} market regime")
            
            recommendation = BotRecommendation(
                bot_name=bot_name,
                score=min(adjusted_score, 1.0),
                reasons=reasons,
                expected_performance=expected_performance,
                risk_level=risk_levels.get(bot_name, 'medium')
            )
            
            recommendations.append(recommendation)
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x.score, reverse=True)
        
        return recommendations
    
    def _estimate_performance(self, bot_name: str, regime: MarketRegime, 
                            capital: float) -> Dict[str, float]:
        """Estimate expected performance for a bot in current regime"""
        # Base performance metrics (historical averages)
        base_performance = {
            'momentum_rider': {
                'win_rate': 0.64,
                'avg_return_per_trade': 0.015,
                'trades_per_day': 3,
                'max_drawdown': 0.15
            },
            'volatility_expander': {
                'win_rate': 0.45,
                'avg_return_per_trade': 0.025,
                'trades_per_day': 0.5,
                'max_drawdown': 0.20
            },
            'short_straddle': {
                'win_rate': 0.74,
                'avg_return_per_trade': 0.008,
                'trades_per_day': 0.2,
                'max_drawdown': 0.25
            },
            'iron_condor': {
                'win_rate': 0.80,
                'avg_return_per_trade': 0.005,
                'trades_per_day': 0.3,
                'max_drawdown': 0.10
            }
        }
        
        perf = base_performance.get(bot_name, {
            'win_rate': 0.5,
            'avg_return_per_trade': 0.01,
            'trades_per_day': 1,
            'max_drawdown': 0.15
        }).copy()
        
        # Adjust based on regime
        if bot_name == 'momentum_rider':
            if regime.regime_type in ['trending_up', 'trending_down']:
                perf['win_rate'] *= 1.1
                perf['avg_return_per_trade'] *= 1.2
            else:
                perf['win_rate'] *= 0.8
                perf['avg_return_per_trade'] *= 0.7
        
        elif bot_name == 'short_straddle':
            if regime.regime_type == 'ranging' and regime.iv_rank > 70:
                perf['win_rate'] *= 1.15
                perf['avg_return_per_trade'] *= 1.3
            elif regime.regime_type in ['trending_up', 'trending_down']:
                perf['win_rate'] *= 0.7
                perf['max_drawdown'] *= 1.5
        
        # Calculate expected metrics
        daily_return = perf['win_rate'] * perf['avg_return_per_trade'] * perf['trades_per_day']
        monthly_return = daily_return * 20  # 20 trading days
        annual_return = monthly_return * 12
        
        return {
            'expected_win_rate': round(perf['win_rate'], 3),
            'expected_daily_return': round(daily_return, 4),
            'expected_monthly_return': round(monthly_return, 3),
            'expected_annual_return': round(annual_return, 3),
            'expected_max_drawdown': round(perf['max_drawdown'], 3),
            'expected_trades_per_day': round(perf['trades_per_day'], 1)
        }
    
    def get_regime_history(self, lookback_days: int = 30) -> List[MarketRegime]:
        """Get historical regime data"""
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        return [r for r in self.regime_history if r.timestamp > cutoff_date]
    
    def analyze_regime_persistence(self) -> Dict[str, Any]:
        """Analyze how long regimes typically persist"""
        if len(self.regime_history) < 10:
            return {'insufficient_data': True}
        
        regime_durations = {}
        current_regime = None
        regime_start = None
        
        for regime in self.regime_history:
            if current_regime != regime.regime_type:
                if current_regime is not None and regime_start is not None:
                    duration = (regime.timestamp - regime_start).total_seconds() / 3600  # Hours
                    if current_regime not in regime_durations:
                        regime_durations[current_regime] = []
                    regime_durations[current_regime].append(duration)
                
                current_regime = regime.regime_type
                regime_start = regime.timestamp
        
        # Calculate statistics
        stats = {}
        for regime_type, durations in regime_durations.items():
            if durations:
                stats[regime_type] = {
                    'avg_duration_hours': np.mean(durations),
                    'min_duration_hours': np.min(durations),
                    'max_duration_hours': np.max(durations),
                    'occurrences': len(durations)
                }
        
        return stats
    
    def save_config(self, filepath: str):
        """Save current configuration"""
        config_data = {
            'lookback_periods': self.lookback_periods,
            'bot_suitability': self.bot_suitability,
            'config': self.config,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        self.logger.info(f"Configuration saved to {filepath}")


# Example usage
if __name__ == "__main__":
    # Create sample market data
    dates = pd.date_range(end=datetime.now(), periods=500, freq='5min')
    
    # Simulate different market regimes
    np.random.seed(42)
    prices = []
    
    # Trending up
    trend_up = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.01, 100)))
    prices.extend(trend_up)
    
    # Ranging
    ranging = prices[-1] + np.random.normal(0, 0.5, 150)
    prices.extend(ranging)
    
    # Volatile
    volatile = prices[-1] * np.exp(np.cumsum(np.random.normal(0, 0.03, 100)))
    prices.extend(volatile)
    
    # Calm
    calm = prices[-1] + np.cumsum(np.random.normal(0, 0.1, 150))
    prices.extend(calm)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.normal(0, 0.5, len(prices))),
        'low': prices - np.abs(np.random.normal(0, 0.5, len(prices))),
        'close': prices + np.random.normal(0, 0.2, len(prices)),
        'volume': np.random.randint(100000, 500000, len(prices))
    }, index=dates[:len(prices)])
    
    # Initialize detector
    detector = MarketRegimeDetector()
    
    # Detect regime for different periods
    print("Market Regime Analysis")
    print("=" * 50)
    
    for i in [100, 250, 400, 499]:
        sample_data = data.iloc[:i+1]
        regime = detector.detect_market_regime(sample_data)
        
        print(f"\nPeriod ending at index {i}:")
        print(f"Regime: {regime.regime_type}")
        print(f"Volatility: {regime.volatility_level}")
        print(f"Trend Strength: {regime.trend_strength:.2f}")
        print(f"Market Structure: {regime.market_structure}")
        print(f"Confidence: {regime.confidence:.2f}")
        
        # Get bot recommendations
        recommendations = detector.recommend_bots(regime)
        
        print("\nBot Recommendations:")
        for rec in recommendations[:3]:  # Top 3
            print(f"\n{rec.bot_name} (Score: {rec.score:.2f})")
            print(f"Risk Level: {rec.risk_level}")
            print("Reasons:")
            for reason in rec.reasons[:2]:
                print(f"  - {reason}")
            print(f"Expected Monthly Return: {rec.expected_performance['expected_monthly_return']:.1%}")