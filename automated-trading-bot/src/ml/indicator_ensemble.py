"""
Indicator Ensemble System with Dynamic Weights
Combines multiple indicators and ML models to generate unified trading signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Import our ML models
from .models.rsi_lstm_model import RSILSTMModel, RSIPattern
from .models.pattern_cnn_model import PatternCNNModel, PatternDetection
from .models.adaptive_thresholds_rl import AdaptiveThresholdsRL, ThresholdAction, MarketState
from .models.price_action_ml_wrapper import MLEnhancedPriceActionSystem

# Import existing indicators
try:
    from ..indicators.rsi_advanced import AdvancedRSI
    from ..indicators.oscillator_matrix import OscillatorMatrix
    from ..indicators.price_action_composite import PriceActionComposite
    from ..indicators.advanced_confirmation import AdvancedConfirmationSystem
    from ..indicators.signal_validator import SignalValidator
except ImportError:
    # Fallback imports for direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from indicators.rsi_advanced import AdvancedRSI
    from indicators.oscillator_matrix import OscillatorMatrix
    from indicators.price_action_composite import PriceActionComposite
    from indicators.advanced_confirmation import AdvancedConfirmationSystem
    from indicators.signal_validator import SignalValidator


@dataclass
class IndicatorSignal:
    """Individual indicator signal"""
    indicator_name: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-1
    confidence: float  # 0-1
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleSignal:
    """Final ensemble signal"""
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-1 (weighted combination)
    confidence: float  # 0-1 (consensus measure)
    consensus_ratio: float  # Percentage of indicators agreeing
    contributing_indicators: List[str]
    individual_signals: List[IndicatorSignal]
    timestamp: datetime
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward_ratio: Optional[float] = None


@dataclass
class EnsembleConfig:
    """Configuration for indicator ensemble"""
    # Weights for different indicator categories
    weights: Dict[str, float] = field(default_factory=lambda: {
        'ml_models': 0.4,  # ML model signals
        'technical_indicators': 0.3,  # Traditional indicators
        'price_action': 0.2,  # Price action analysis
        'confirmation_systems': 0.1  # Confirmation filters
    })
    
    # Individual indicator weights
    indicator_weights: Dict[str, float] = field(default_factory=lambda: {
        'rsi_lstm': 0.15,
        'pattern_cnn': 0.15,
        'adaptive_thresholds': 0.10,
        'advanced_rsi': 0.10,
        'oscillator_matrix': 0.10,
        'price_action_composite': 0.20,
        'advanced_confirmation': 0.10,
        'signal_validator': 0.10
    })
    
    # Ensemble parameters
    min_consensus_ratio: float = 0.6  # Minimum agreement for signal generation
    min_confidence: float = 0.5  # Minimum confidence threshold
    adaptive_weights: bool = True  # Enable dynamic weight adjustment
    performance_window: int = 100  # Window for performance-based weight adjustment


class BaseIndicatorWrapper(ABC):
    """Abstract base class for indicator wrappers"""
    
    def __init__(self, name: str, weight: float):
        self.name = name
        self.weight = weight
        self.performance_history = []
        self.is_enabled = True
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, **kwargs) -> Optional[IndicatorSignal]:
        """Generate signal from indicator"""
        pass
    
    def update_performance(self, success: bool, return_value: float = 0.0):
        """Update performance history"""
        self.performance_history.append({
            'success': success,
            'return': return_value,
            'timestamp': datetime.now()
        })
        
        # Keep only recent history
        if len(self.performance_history) > 200:
            self.performance_history = self.performance_history[-200:]
    
    def get_recent_performance(self, window: int = 50) -> Dict[str, float]:
        """Get recent performance metrics"""
        if len(self.performance_history) == 0:
            return {'win_rate': 0.5, 'avg_return': 0.0, 'consistency': 0.5}
        
        recent = self.performance_history[-window:]
        
        win_rate = sum(1 for r in recent if r['success']) / len(recent)
        avg_return = np.mean([r['return'] for r in recent])
        consistency = 1.0 - np.std([1 if r['success'] else 0 for r in recent])
        
        return {
            'win_rate': win_rate,
            'avg_return': avg_return,
            'consistency': consistency
        }


class MLModelWrapper(BaseIndicatorWrapper):
    """Wrapper for ML model indicators"""
    
    def __init__(self, name: str, model: Any, weight: float):
        super().__init__(name, weight)
        self.model = model
    
    def generate_signal(self, data: pd.DataFrame, **kwargs) -> Optional[IndicatorSignal]:
        """Generate signal from ML model"""
        try:
            if self.name == 'rsi_lstm' and isinstance(self.model, RSILSTMModel):
                return self._generate_rsi_lstm_signal(data, **kwargs)
            elif self.name == 'pattern_cnn' and isinstance(self.model, PatternCNNModel):
                return self._generate_pattern_cnn_signal(data, **kwargs)
            elif self.name == 'adaptive_thresholds' and isinstance(self.model, AdaptiveThresholdsRL):
                return self._generate_adaptive_threshold_signal(data, **kwargs)
        except Exception as e:
            logging.getLogger(__name__).error(f"Error in {self.name}: {e}")
            return None
    
    def _generate_rsi_lstm_signal(self, data: pd.DataFrame, **kwargs) -> Optional[IndicatorSignal]:
        """Generate signal from RSI LSTM model"""
        if len(data) < 25:
            return None
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Get pattern prediction
        rsi_sequence = rsi.tail(25)
        price_sequence = data['close'].tail(25)
        
        pattern = self.model.predict_pattern(rsi_sequence, price_sequence)
        
        # Convert to indicator signal
        if pattern.strength > 0.5:
            if pattern.price_direction > 0:
                signal_type = 'buy'
            elif pattern.price_direction < 0:
                signal_type = 'sell'
            else:
                signal_type = 'hold'
            
            return IndicatorSignal(
                indicator_name=self.name,
                signal_type=signal_type,
                strength=pattern.strength,
                confidence=pattern.confidence,
                timestamp=data.index[-1],
                metadata={
                    'pattern_type': pattern.pattern_type,
                    'predicted_rsi': pattern.rsi_prediction,
                    'price_direction': pattern.price_direction
                }
            )
        
        return None
    
    def _generate_pattern_cnn_signal(self, data: pd.DataFrame, **kwargs) -> Optional[IndicatorSignal]:
        """Generate signal from Pattern CNN model"""
        if len(data) < 50:
            return None
        
        # Get pattern detection
        pattern = self.model.detect_pattern(data)
        
        if pattern.strength > 0.5 and pattern.pattern_type != 'none':
            if pattern.breakout_direction > 0:
                signal_type = 'buy'
            elif pattern.breakout_direction < 0:
                signal_type = 'sell'
            else:
                signal_type = 'hold'
            
            return IndicatorSignal(
                indicator_name=self.name,
                signal_type=signal_type,
                strength=pattern.strength,
                confidence=pattern.confidence,
                timestamp=data.index[-1],
                metadata={
                    'pattern_type': pattern.pattern_type,
                    'target_price': pattern.target_price,
                    'breakout_direction': pattern.breakout_direction
                }
            )
        
        return None
    
    def _generate_adaptive_threshold_signal(self, data: pd.DataFrame, **kwargs) -> Optional[IndicatorSignal]:
        """Generate signal from adaptive thresholds RL model"""
        # Create market state
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std().iloc[-1] if len(data) > 20 else 0.02
        
        current_state = MarketState(
            volatility=volatility,
            trend_strength=0.0,  # Simplified
            volume_ratio=1.0,
            time_of_day=0.5,
            market_regime='ranging',
            recent_performance=0.0,
            current_thresholds=self.model.get_current_thresholds()
        )
        
        # Get threshold adjustments
        threshold_actions = self.model.adapt_thresholds(data.tail(50), current_state)
        
        # Generate signal based on threshold changes
        signal_strength = 0
        total_confidence = 0
        action_count = 0
        
        for action in threshold_actions.values():
            if abs(action.adjustment) > 0.1:  # Significant adjustment
                signal_strength += action.adjustment
                total_confidence += action.confidence
                action_count += 1
        
        if action_count > 0:
            avg_confidence = total_confidence / action_count
            normalized_strength = abs(signal_strength / action_count)
            
            if normalized_strength > 0.3:
                signal_type = 'buy' if signal_strength > 0 else 'sell'
                
                return IndicatorSignal(
                    indicator_name=self.name,
                    signal_type=signal_type,
                    strength=normalized_strength,
                    confidence=avg_confidence,
                    timestamp=data.index[-1],
                    metadata={
                        'threshold_actions': len(threshold_actions),
                        'avg_adjustment': signal_strength / action_count,
                        'market_volatility': volatility
                    }
                )
        
        return None


class TraditionalIndicatorWrapper(BaseIndicatorWrapper):
    """Wrapper for traditional technical indicators"""
    
    def __init__(self, name: str, indicator: Any, weight: float):
        super().__init__(name, weight)
        self.indicator = indicator
    
    def generate_signal(self, data: pd.DataFrame, **kwargs) -> Optional[IndicatorSignal]:
        """Generate signal from traditional indicator"""
        try:
            if self.name == 'advanced_rsi':
                return self._generate_rsi_signal(data)
            elif self.name == 'oscillator_matrix':
                return self._generate_oscillator_signal(data)
            elif self.name == 'price_action_composite':
                return self._generate_price_action_signal(data)
            elif self.name == 'advanced_confirmation':
                return self._generate_confirmation_signal(data, **kwargs)
            elif self.name == 'signal_validator':
                return self._generate_validator_signal(data, **kwargs)
        except Exception as e:
            logging.getLogger(__name__).error(f"Error in {self.name}: {e}")
            return None
    
    def _generate_rsi_signal(self, data: pd.DataFrame) -> Optional[IndicatorSignal]:
        """Generate signal from Advanced RSI"""
        rsi_value = self.indicator.calculate(data['close'])
        
        if pd.isna(rsi_value) or len(data) < 20:
            return None
        
        # Simple RSI signal logic
        if rsi_value < 30:
            signal_type = 'buy'
            strength = (30 - rsi_value) / 30
        elif rsi_value > 70:
            signal_type = 'sell'
            strength = (rsi_value - 70) / 30
        else:
            return None
        
        return IndicatorSignal(
            indicator_name=self.name,
            signal_type=signal_type,
            strength=min(strength, 1.0),
            confidence=0.7,
            timestamp=data.index[-1],
            metadata={'rsi_value': rsi_value}
        )
    
    def _generate_oscillator_signal(self, data: pd.DataFrame) -> Optional[IndicatorSignal]:
        """Generate signal from Oscillator Matrix"""
        oscillator_data = self.indicator.calculate_all_oscillators(data)
        
        if oscillator_data.empty:
            return None
        
        composite_score = oscillator_data['composite_score'].iloc[-1]
        
        if abs(composite_score) > 50:
            signal_type = 'buy' if composite_score > 0 else 'sell'
            strength = abs(composite_score) / 100
            
            return IndicatorSignal(
                indicator_name=self.name,
                signal_type=signal_type,
                strength=strength,
                confidence=0.6,
                timestamp=data.index[-1],
                metadata={'composite_score': composite_score}
            )
        
        return None
    
    def _generate_price_action_signal(self, data: pd.DataFrame) -> Optional[IndicatorSignal]:
        """Generate signal from Price Action Composite"""
        # If using ML-enhanced price action
        if isinstance(self.indicator, MLEnhancedPriceActionSystem):
            analysis = self.indicator.analyze(data)
            composite = analysis['composite_signal']
            
            if composite['direction'] != 'neutral' and composite['strength'] > 0.3:
                return IndicatorSignal(
                    indicator_name=self.name,
                    signal_type='buy' if composite['direction'] == 'bullish' else 'sell',
                    strength=composite['strength'],
                    confidence=composite['confidence'],
                    timestamp=data.index[-1],
                    metadata={
                        'ml_enhanced': True,
                        'bullish_factors': composite['factors']['bullish'],
                        'bearish_factors': composite['factors']['bearish'],
                        'signal_count': composite['factors']['signal_count'],
                        'ml_metrics': analysis.get('ml_metrics', {})
                    }
                )
        else:
            # Traditional price action
            pa_data = self.indicator.calculate(data)
            
            if pa_data.empty:
                return None
            
            signal = pa_data['signal'].iloc[-1]
            strength = pa_data['signal_strength'].iloc[-1]
            confidence_str = pa_data['confidence'].iloc[-1]
            
            # Convert confidence string to float
            confidence_map = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
            confidence = confidence_map.get(confidence_str, 0.5)
            
            if signal != 0 and strength > 40:
                signal_type = 'buy' if signal > 0 else 'sell'
                
                return IndicatorSignal(
                    indicator_name=self.name,
                    signal_type=signal_type,
                    strength=strength / 100,
                    confidence=confidence,
                    timestamp=data.index[-1],
                    metadata={
                        'signal_strength': strength,
                        'confidence_level': confidence_str
                    }
                )
        
        return None
    
    def _generate_confirmation_signal(self, data: pd.DataFrame, **kwargs) -> Optional[IndicatorSignal]:
        """Generate signal from Advanced Confirmation System"""
        # This is typically used as a confirmation layer, not a primary signal generator
        # Return None as it should be used through the confirmation wrapper
        return None
    
    def _generate_validator_signal(self, data: pd.DataFrame, **kwargs) -> Optional[IndicatorSignal]:
        """Generate signal from Signal Validator"""
        # This is typically used as a validation layer, not a primary signal generator
        # Return None as it should be used through the validation wrapper
        return None


class IndicatorEnsemble:
    """
    Main ensemble system that combines all indicators and ML models
    
    Features:
    - Dynamic weight adjustment based on performance
    - Consensus-based signal generation
    - Risk management integration
    - Performance tracking and analysis
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        """Initialize indicator ensemble"""
        self.config = config or EnsembleConfig()
        self.logger = logging.getLogger(__name__)
        
        # Indicator wrappers
        self.indicators: Dict[str, BaseIndicatorWrapper] = {}
        
        # Performance tracking
        self.signal_history = []
        self.performance_metrics = {}
        
        # Weight adaptation
        self.weight_history = []
        self.last_weight_update = datetime.now()
        
        self.logger.info("Initialized Indicator Ensemble")
    
    def add_ml_model(self, name: str, model: Any, weight: Optional[float] = None):
        """Add ML model to ensemble"""
        weight = weight or self.config.indicator_weights.get(name, 0.1)
        wrapper = MLModelWrapper(name, model, weight)
        self.indicators[name] = wrapper
        self.logger.info(f"Added ML model: {name} (weight: {weight})")
    
    def add_traditional_indicator(self, name: str, indicator: Any, weight: Optional[float] = None):
        """Add traditional indicator to ensemble"""
        weight = weight or self.config.indicator_weights.get(name, 0.1)
        wrapper = TraditionalIndicatorWrapper(name, indicator, weight)
        self.indicators[name] = wrapper
        self.logger.info(f"Added traditional indicator: {name} (weight: {weight})")
    
    def generate_ensemble_signal(self, data: pd.DataFrame, **kwargs) -> Optional[EnsembleSignal]:
        """
        Generate ensemble signal from all indicators
        
        Args:
            data: Market data
            **kwargs: Additional parameters for indicators
            
        Returns:
            EnsembleSignal or None if no consensus
        """
        # Collect individual signals
        individual_signals = []
        
        for name, indicator in self.indicators.items():
            if indicator.is_enabled:
                signal = indicator.generate_signal(data, **kwargs)
                if signal:
                    individual_signals.append(signal)
        
        if len(individual_signals) == 0:
            return None
        
        # Calculate weighted consensus
        consensus_result = self._calculate_consensus(individual_signals)
        
        if consensus_result is None:
            return None
        
        signal_type, weighted_strength, weighted_confidence, consensus_ratio = consensus_result
        
        # Check minimum thresholds
        if (consensus_ratio < self.config.min_consensus_ratio or 
            weighted_confidence < self.config.min_confidence):
            return None
        
        # Calculate risk management parameters
        risk_params = self._calculate_risk_parameters(data, individual_signals, signal_type)
        
        # Create ensemble signal
        ensemble_signal = EnsembleSignal(
            signal_type=signal_type,
            strength=weighted_strength,
            confidence=weighted_confidence,
            consensus_ratio=consensus_ratio,
            contributing_indicators=[s.indicator_name for s in individual_signals],
            individual_signals=individual_signals,
            timestamp=data.index[-1],
            target_price=risk_params.get('target_price'),
            stop_loss=risk_params.get('stop_loss'),
            risk_reward_ratio=risk_params.get('risk_reward_ratio')
        )
        
        # Store signal for performance tracking
        self.signal_history.append(ensemble_signal)
        
        # Update weights if adaptive mode is enabled
        if self.config.adaptive_weights:
            self._update_adaptive_weights()
        
        return ensemble_signal
    
    def _calculate_consensus(self, signals: List[IndicatorSignal]) -> Optional[Tuple[str, float, float, float]]:
        """Calculate weighted consensus from individual signals"""
        if not signals:
            return None
        
        # Group signals by type
        signal_groups = {'buy': [], 'sell': [], 'hold': []}
        
        for signal in signals:
            if signal.signal_type in signal_groups:
                signal_groups[signal.signal_type].append(signal)
        
        # Calculate weighted votes for each signal type
        weighted_votes = {}
        
        for signal_type, group_signals in signal_groups.items():
            if not group_signals:
                weighted_votes[signal_type] = 0
                continue
            
            total_weight = 0
            weighted_strength = 0
            weighted_confidence = 0
            
            for signal in group_signals:
                # Get indicator weight
                indicator_weight = self.indicators[signal.indicator_name].weight
                
                # Adjust weight by performance if available
                performance = self.indicators[signal.indicator_name].get_recent_performance()
                performance_multiplier = (
                    performance['win_rate'] * 0.5 + 
                    performance['consistency'] * 0.3 + 
                    min(performance['avg_return'] + 0.5, 1.0) * 0.2
                )
                
                adjusted_weight = indicator_weight * performance_multiplier
                
                total_weight += adjusted_weight
                weighted_strength += signal.strength * adjusted_weight
                weighted_confidence += signal.confidence * adjusted_weight
            
            if total_weight > 0:
                weighted_votes[signal_type] = {
                    'weight': total_weight,
                    'strength': weighted_strength / total_weight,
                    'confidence': weighted_confidence / total_weight,
                    'count': len(group_signals)
                }
            else:
                weighted_votes[signal_type] = {'weight': 0, 'strength': 0, 'confidence': 0, 'count': 0}
        
        # Find winning signal type
        max_weight = 0
        winning_type = 'hold'
        
        for signal_type, vote_data in weighted_votes.items():
            if vote_data['weight'] > max_weight:
                max_weight = vote_data['weight']
                winning_type = signal_type
        
        if winning_type == 'hold' or max_weight == 0:
            return None
        
        # Calculate consensus ratio
        total_signals = len(signals)
        agreeing_signals = weighted_votes[winning_type]['count']
        consensus_ratio = agreeing_signals / total_signals
        
        # Get weighted metrics
        winning_data = weighted_votes[winning_type]
        
        return (
            winning_type,
            winning_data['strength'],
            winning_data['confidence'],
            consensus_ratio
        )
    
    def _calculate_risk_parameters(self, data: pd.DataFrame, signals: List[IndicatorSignal], 
                                 signal_type: str) -> Dict[str, float]:
        """Calculate risk management parameters"""
        current_price = data['close'].iloc[-1]
        
        # Calculate volatility-based stops
        returns = data['close'].pct_change().dropna()
        volatility = returns.std()
        atr = self._calculate_atr(data, 14)
        
        # Base stop loss and target calculations
        if signal_type == 'buy':
            stop_loss = current_price - (atr * 2)
            target_price = current_price + (atr * 3)
        else:  # sell
            stop_loss = current_price + (atr * 2)
            target_price = current_price - (atr * 3)
        
        # Check for ML model targets
        ml_targets = []
        for signal in signals:
            if 'target_price' in signal.metadata and signal.metadata['target_price']:
                ml_targets.append(signal.metadata['target_price'])
        
        # Use ML target if available and reasonable
        if ml_targets:
            avg_ml_target = np.mean(ml_targets)
            if signal_type == 'buy' and avg_ml_target > current_price:
                target_price = avg_ml_target
            elif signal_type == 'sell' and avg_ml_target < current_price:
                target_price = avg_ml_target
        
        # Calculate risk-reward ratio
        risk = abs(current_price - stop_loss)
        reward = abs(target_price - current_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        return {
            'target_price': target_price,
            'stop_loss': stop_loss,
            'risk_reward_ratio': risk_reward_ratio
        }
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(period).mean().iloc[-1]
    
    def _update_adaptive_weights(self):
        """Update indicator weights based on recent performance"""
        if (datetime.now() - self.last_weight_update).seconds < 3600:  # Update hourly
            return
        
        # Calculate performance-based weight adjustments
        performance_scores = {}
        
        for name, indicator in self.indicators.items():
            performance = indicator.get_recent_performance()
            
            # Combined performance score
            score = (
                performance['win_rate'] * 0.5 +
                performance['consistency'] * 0.3 +
                min(performance['avg_return'] + 0.5, 1.0) * 0.2
            )
            
            performance_scores[name] = score
        
        # Normalize scores and adjust weights
        total_score = sum(performance_scores.values())
        
        if total_score > 0:
            for name, indicator in self.indicators.items():
                old_weight = indicator.weight
                normalized_score = performance_scores[name] / total_score
                
                # Gradual weight adjustment (20% towards performance-based weight)
                target_weight = normalized_score * sum(self.config.indicator_weights.values())
                new_weight = old_weight * 0.8 + target_weight * 0.2
                
                indicator.weight = new_weight
        
        self.last_weight_update = datetime.now()
        self.weight_history.append({
            'timestamp': datetime.now(),
            'weights': {name: ind.weight for name, ind in self.indicators.items()},
            'performance_scores': performance_scores.copy()
        })
    
    def update_signal_performance(self, signal_id: str, success: bool, return_value: float = 0.0):
        """Update performance for a specific signal"""
        # Find the signal in history
        for ensemble_signal in self.signal_history:
            if str(ensemble_signal.timestamp) == signal_id:
                # Update performance for all contributing indicators
                for indicator_name in ensemble_signal.contributing_indicators:
                    if indicator_name in self.indicators:
                        self.indicators[indicator_name].update_performance(success, return_value)
                break
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get ensemble summary and performance metrics"""
        summary = {
            'total_indicators': len(self.indicators),
            'enabled_indicators': sum(1 for ind in self.indicators.values() if ind.is_enabled),
            'signals_generated': len(self.signal_history),
            'adaptive_weights_enabled': self.config.adaptive_weights,
            'min_consensus_ratio': self.config.min_consensus_ratio,
            'indicators': {}
        }
        
        # Individual indicator summaries
        for name, indicator in self.indicators.items():
            performance = indicator.get_recent_performance()
            summary['indicators'][name] = {
                'weight': indicator.weight,
                'is_enabled': indicator.is_enabled,
                'performance': performance,
                'signals_count': len(indicator.performance_history)
            }
        
        return summary
    
    def save_ensemble(self, filepath: str):
        """Save ensemble configuration and history"""
        ensemble_data = {
            'config': {
                'weights': self.config.weights,
                'indicator_weights': self.config.indicator_weights,
                'min_consensus_ratio': self.config.min_consensus_ratio,
                'min_confidence': self.config.min_confidence,
                'adaptive_weights': self.config.adaptive_weights
            },
            'indicators': {
                name: {
                    'weight': ind.weight,
                    'is_enabled': ind.is_enabled,
                    'performance_history': ind.performance_history[-50:]  # Last 50 records
                }
                for name, ind in self.indicators.items()
            },
            'ensemble_summary': self.get_ensemble_summary(),
            'weight_history': self.weight_history[-20:],  # Last 20 weight updates
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(ensemble_data, f, indent=2, default=str)
        
        self.logger.info(f"Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath: str):
        """Load ensemble configuration and history"""
        with open(filepath, 'r') as f:
            ensemble_data = json.load(f)
        
        # Update configuration
        config_data = ensemble_data['config']
        self.config.weights = config_data['weights']
        self.config.indicator_weights = config_data['indicator_weights']
        self.config.min_consensus_ratio = config_data['min_consensus_ratio']
        self.config.min_confidence = config_data['min_confidence']
        self.config.adaptive_weights = config_data['adaptive_weights']
        
        # Restore indicator weights and performance
        indicators_data = ensemble_data['indicators']
        for name, ind_data in indicators_data.items():
            if name in self.indicators:
                self.indicators[name].weight = ind_data['weight']
                self.indicators[name].is_enabled = ind_data['is_enabled']
                # Restore recent performance history
                self.indicators[name].performance_history = ind_data.get('performance_history', [])
        
        self.weight_history = ensemble_data.get('weight_history', [])
        
        self.logger.info(f"Ensemble loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # This would normally be run with actual data and models
    print("Indicator Ensemble System")
    print("=" * 50)
    
    # Create sample configuration
    config = EnsembleConfig()
    ensemble = IndicatorEnsemble(config)
    
    # In practice, you would add real models and indicators:
    # ensemble.add_ml_model('rsi_lstm', rsi_lstm_model)
    # ensemble.add_ml_model('pattern_cnn', pattern_cnn_model) 
    # ensemble.add_ml_model('adaptive_thresholds', rl_model)
    # ensemble.add_traditional_indicator('advanced_rsi', advanced_rsi)
    # ensemble.add_traditional_indicator('oscillator_matrix', oscillator_matrix)
    # ensemble.add_traditional_indicator('price_action_composite', price_action)
    
    print("Ensemble initialized successfully!")
    print(f"Configuration: {config}")
    
    # Generate sample data for testing
    dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
    
    sample_data = pd.DataFrame({
        'open': prices + np.random.randn(len(prices)) * 0.5,
        'high': prices + np.abs(np.random.randn(len(prices))) * 1.0,
        'low': prices - np.abs(np.random.randn(len(prices))) * 1.0, 
        'close': prices,
        'volume': np.random.randint(50000, 200000, len(prices))
    }, index=dates)
    
    # Ensure OHLC consistency
    sample_data['high'] = sample_data[['open', 'high', 'close']].max(axis=1)
    sample_data['low'] = sample_data[['open', 'low', 'close']].min(axis=1)
    
    print(f"Sample data shape: {sample_data.shape}")
    print("Ensemble system ready for integration with trading bot!")