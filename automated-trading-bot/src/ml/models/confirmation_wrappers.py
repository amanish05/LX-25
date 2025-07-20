"""
Wrapper classes for integrating confirmation systems with ML ensemble
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

try:
    # Try absolute import first (for when running as module)
    from src.indicators.advanced_confirmation import AdvancedConfirmationSystem
    from src.indicators.signal_validator import SignalValidator
except ImportError:
    # Fall back to relative import
    from ...indicators.advanced_confirmation import AdvancedConfirmationSystem
    from ...indicators.signal_validator import SignalValidator


class MLEnhancedConfirmationSystem:
    """
    ML-enhanced wrapper for Advanced Confirmation System
    Integrates multi-layer confirmations with ML ensemble
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ML-enhanced confirmation system"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Base confirmation system
        self.confirmation_system = AdvancedConfirmationSystem()
        
        # ML enhancement parameters
        self.ml_weight_adjustments = {
            'trendline_break': 1.0,
            'predictive_range': 1.0,
            'fvg_confirmation': 1.2,  # Higher weight for FVG
            'reversal_signals': 1.0,
            'volume_confirmation': 1.1,
            'momentum_alignment': 1.0
        }
        
        # Performance tracking
        self.confirmation_performance = {
            'total_confirmations': 0,
            'successful_confirmations': 0,
            'false_positives': 0,
            'avg_confluence_score': 0.0
        }
    
    def get_confirmations(self, signal_type: str, data: pd.DataFrame, 
                         entry_price: float, ml_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get ML-enhanced confirmations for a signal
        
        Args:
            signal_type: 'buy' or 'sell'
            data: Market data
            entry_price: Proposed entry price
            ml_context: Additional ML context (ensemble signals, etc.)
            
        Returns:
            Enhanced confirmation data with ML adjustments
        """
        # Create primary signal for validation
        primary_signal = {
            'symbol': 'TEST',
            'type': signal_type.upper(),
            'entry_price': entry_price,
            'timestamp': data.index[-1] if not data.empty else pd.Timestamp.now()
        }
        
        # Get base confirmations using validate_signal
        confirmation_result = self.confirmation_system.validate_signal(primary_signal, data)
        
        # Convert to expected format
        if confirmation_result is None:
            confirmations = {
                'total_confirmations': 0,
                'confluence_score': 0.0,
                'confirmations': [],
                'validated': False
            }
        else:
            confirmations = {
                'total_confirmations': len(confirmation_result.confirmations),
                'confluence_score': confirmation_result.confluence_score,
                'confirmations': confirmation_result.confirmations,
                'validated': True,
                'strength': confirmation_result.strength.value,
                'entry_price': confirmation_result.entry_price,
                'stop_loss': confirmation_result.stop_loss,
                'target_price': confirmation_result.target_price
            }
        
        # Update performance tracking (always track attempts)
        self.confirmation_performance['total_confirmations'] += 1
        
        if not confirmations or confirmations['total_confirmations'] == 0:
            return confirmations
        
        # Apply ML enhancements
        enhanced_confirmations = self._apply_ml_enhancements(confirmations, ml_context)
        self.confirmation_performance['avg_confluence_score'] = (
            (self.confirmation_performance['avg_confluence_score'] * 
             (self.confirmation_performance['total_confirmations'] - 1) +
             enhanced_confirmations['confluence_score']) /
            self.confirmation_performance['total_confirmations']
        )
        
        # Add ML metadata
        enhanced_confirmations['ml_enhanced'] = True
        enhanced_confirmations['ml_adjustments'] = {
            'original_score': confirmations['confluence_score'],
            'ml_score': enhanced_confirmations['confluence_score'],
            'confidence_boost': enhanced_confirmations['confluence_score'] - confirmations['confluence_score']
        }
        
        return enhanced_confirmations
    
    def _apply_ml_enhancements(self, confirmations: Dict, ml_context: Optional[Dict]) -> Dict:
        """Apply ML enhancements to confirmations"""
        enhanced = confirmations.copy()
        
        # Recalculate confluence score with ML weights
        total_weight = 0
        weighted_score = 0
        
        for conf_type, conf_data in confirmations['confirmations'].items():
            if isinstance(conf_data, dict) and conf_data.get('confirmed', False):
                weight = self.ml_weight_adjustments.get(conf_type, 1.0)
                confidence = conf_data.get('confidence', 0.5)
                
                # Boost confidence if ML models agree
                if ml_context and 'ensemble_signals' in ml_context:
                    ml_agreement = self._check_ml_agreement(conf_type, ml_context['ensemble_signals'])
                    if ml_agreement > 0.7:
                        confidence *= 1.2  # 20% boost for strong ML agreement
                
                weighted_score += confidence * weight
                total_weight += weight
        
        # Calculate enhanced confluence score
        if total_weight > 0:
            enhanced['confluence_score'] = min(weighted_score / total_weight, 1.0)
        
        # Adjust false positive probability based on ML confidence
        if ml_context and 'ml_confidence' in ml_context:
            ml_confidence = ml_context['ml_confidence']
            # Reduce false positive probability if ML is confident
            enhanced['false_positive_probability'] *= (1 - ml_confidence * 0.3)
        
        # Update signal strength based on enhanced score
        if enhanced['confluence_score'] >= 0.85:
            enhanced['signal_strength'] = 'VERY_STRONG'
        elif enhanced['confluence_score'] >= 0.75:
            enhanced['signal_strength'] = 'STRONG'
        elif enhanced['confluence_score'] >= 0.65:
            enhanced['signal_strength'] = 'MODERATE'
        else:
            enhanced['signal_strength'] = 'WEAK'
        
        return enhanced
    
    def _check_ml_agreement(self, confirmation_type: str, ensemble_signals: List[Dict]) -> float:
        """Check if ML models agree with specific confirmation type"""
        if not ensemble_signals:
            return 0.5
        
        agreement_count = 0
        relevant_signals = 0
        
        for signal in ensemble_signals:
            # Check if signal is relevant to confirmation type
            if confirmation_type == 'momentum_alignment' and 'rsi' in signal.get('indicator_name', '').lower():
                relevant_signals += 1
                if signal.get('strength', 0) > 0.6:
                    agreement_count += 1
            elif confirmation_type == 'volume_confirmation' and 'volume' in str(signal.get('metadata', {})):
                relevant_signals += 1
                if signal.get('metadata', {}).get('volume_confirmed', False):
                    agreement_count += 1
            elif confirmation_type == 'fvg_confirmation' and 'price_action' in signal.get('indicator_name', '').lower():
                relevant_signals += 1
                if signal.get('metadata', {}).get('fvg_detected', False):
                    agreement_count += 1
        
        return agreement_count / relevant_signals if relevant_signals > 0 else 0.5
    
    def update_performance(self, confirmation_id: str, was_successful: bool):
        """Update confirmation performance"""
        if was_successful:
            self.confirmation_performance['successful_confirmations'] += 1
        else:
            self.confirmation_performance['false_positives'] += 1
        
        # Adjust ML weights based on performance
        if self.confirmation_performance['total_confirmations'] % 50 == 0:
            self._adjust_ml_weights()
    
    def _adjust_ml_weights(self):
        """Adjust ML weights based on performance"""
        success_rate = (self.confirmation_performance['successful_confirmations'] /
                       self.confirmation_performance['total_confirmations']
                       if self.confirmation_performance['total_confirmations'] > 0 else 0.5)
        
        # If performance is poor, reduce weights for less reliable confirmations
        if success_rate < 0.4:
            self.ml_weight_adjustments['predictive_range'] *= 0.95
            self.ml_weight_adjustments['reversal_signals'] *= 0.95
        elif success_rate > 0.6:
            # Boost weights for reliable confirmations
            self.ml_weight_adjustments['fvg_confirmation'] *= 1.02
            self.ml_weight_adjustments['volume_confirmation'] *= 1.02
        
        # Normalize weights
        total_weight = sum(self.ml_weight_adjustments.values())
        for key in self.ml_weight_adjustments:
            self.ml_weight_adjustments[key] /= total_weight / 6.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total = self.confirmation_performance['total_confirmations']
        if total == 0:
            return {'status': 'no_data'}
        
        return {
            'total_confirmations': total,
            'success_rate': self.confirmation_performance['successful_confirmations'] / total,
            'false_positive_rate': self.confirmation_performance['false_positives'] / total,
            'avg_confluence_score': self.confirmation_performance['avg_confluence_score'],
            'ml_weight_adjustments': self.ml_weight_adjustments.copy()
        }


class MLEnhancedSignalValidator:
    """
    ML-enhanced wrapper for Signal Validator
    Provides final validation layer with ML context
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ML-enhanced signal validator"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Base signal validator
        self.signal_validator = SignalValidator()
        
        # ML enhancement parameters
        self.ml_validation_rules = {
            'ensemble_consensus': 0.6,  # Minimum ensemble consensus
            'ml_confidence': 0.5,       # Minimum ML confidence
            'confirmation_score': 0.65, # Minimum confirmation score
            'risk_reward_min': 1.5      # Minimum risk-reward ratio
        }
        
        # Adaptive thresholds based on market conditions
        self.adaptive_thresholds = {
            'high_volatility': {
                'ensemble_consensus': 0.7,
                'ml_confidence': 0.6,
                'confirmation_score': 0.75
            },
            'trending': {
                'ensemble_consensus': 0.55,
                'ml_confidence': 0.45,
                'confirmation_score': 0.6
            },
            'ranging': {
                'ensemble_consensus': 0.65,
                'ml_confidence': 0.55,
                'confirmation_score': 0.7
            }
        }
        
        # Validation performance tracking
        self.validation_stats = {
            'total_validations': 0,
            'accepted_signals': 0,
            'rejected_signals': 0,
            'false_positives': 0,
            'true_positives': 0
        }
    
    def validate_ensemble_signal(self, ensemble_signal: Dict, market_data: pd.DataFrame,
                               ml_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Validate ensemble signal with ML enhancements
        
        Args:
            ensemble_signal: Signal from ML ensemble
            market_data: Recent market data
            ml_context: Additional ML context
            
        Returns:
            Validation result with ML enhancements
        """
        # Prepare signal for base validator
        base_signal = {
            'type': ensemble_signal.get('signal_type', 'hold'),
            'strength': ensemble_signal.get('strength', 0),
            'timestamp': ensemble_signal.get('timestamp', datetime.now()),
            'indicator': 'ml_ensemble',
            'metadata': ensemble_signal.get('metadata', {})
        }
        
        # Get base validation
        is_valid, base_validation = self.signal_validator.validate(base_signal, market_data)
        
        # Apply ML validation rules
        ml_validation = self._apply_ml_validation(ensemble_signal, base_validation, ml_context)
        
        # Combine validations
        final_validation = self._combine_validations(base_validation, ml_validation)
        
        # Update statistics
        self.validation_stats['total_validations'] += 1
        if final_validation['recommendation'] in ['STRONG_SIGNAL', 'MODERATE_SIGNAL']:
            self.validation_stats['accepted_signals'] += 1
        else:
            self.validation_stats['rejected_signals'] += 1
        
        return final_validation
    
    def _apply_ml_validation(self, ensemble_signal: Dict, base_validation: Dict,
                           ml_context: Optional[Dict]) -> Dict[str, Any]:
        """Apply ML-specific validation rules"""
        ml_validation = {
            'passed': True,
            'failed_rules': [],
            'ml_score': 0.0,
            'adjustments': {}
        }
        
        # Determine market condition
        market_condition = ml_context.get('market_condition', 'normal') if ml_context else 'normal'
        
        # Get adaptive thresholds
        thresholds = self.adaptive_thresholds.get(market_condition, self.ml_validation_rules)
        
        # Check ensemble consensus
        consensus_ratio = ensemble_signal.get('consensus_ratio', 0)
        if consensus_ratio < thresholds['ensemble_consensus']:
            ml_validation['passed'] = False
            ml_validation['failed_rules'].append({
                'rule': 'ensemble_consensus',
                'reason': f'Consensus ratio {consensus_ratio:.2f} below threshold {thresholds["ensemble_consensus"]}'
            })
        else:
            ml_validation['ml_score'] += 0.25
        
        # Check ML confidence
        ml_confidence = ensemble_signal.get('confidence', 0)
        if ml_confidence < thresholds['ml_confidence']:
            ml_validation['passed'] = False
            ml_validation['failed_rules'].append({
                'rule': 'ml_confidence',
                'reason': f'ML confidence {ml_confidence:.2f} below threshold {thresholds["ml_confidence"]}'
            })
        else:
            ml_validation['ml_score'] += 0.25
        
        # Check confirmation score if available
        if ml_context and 'confirmation_score' in ml_context:
            conf_score = ml_context['confirmation_score']
            if conf_score < thresholds['confirmation_score']:
                ml_validation['passed'] = False
                ml_validation['failed_rules'].append({
                    'rule': 'confirmation_score',
                    'reason': f'Confirmation score {conf_score:.2f} below threshold {thresholds["confirmation_score"]}'
                })
            else:
                ml_validation['ml_score'] += 0.25
        
        # Check risk-reward ratio
        risk_reward = ensemble_signal.get('risk_reward_ratio', 0)
        if risk_reward < self.ml_validation_rules['risk_reward_min']:
            ml_validation['passed'] = False
            ml_validation['failed_rules'].append({
                'rule': 'risk_reward',
                'reason': f'Risk-reward ratio {risk_reward:.2f} below minimum {self.ml_validation_rules["risk_reward_min"]}'
            })
        else:
            ml_validation['ml_score'] += 0.25
        
        # Adjust score based on ML model diversity
        if 'contributing_indicators' in ensemble_signal:
            diversity_bonus = len(ensemble_signal['contributing_indicators']) / 10
            ml_validation['ml_score'] += min(diversity_bonus, 0.2)
        
        return ml_validation
    
    def _combine_validations(self, base_validation: Dict, ml_validation: Dict) -> Dict[str, Any]:
        """Combine base and ML validations"""
        combined = base_validation.copy()
        
        # Add ML validation results
        combined['ml_validation'] = ml_validation
        combined['ml_enhanced'] = True
        
        # Update overall validity
        if not ml_validation['passed']:
            combined['is_valid'] = False
            combined['failed_rules'].extend(ml_validation['failed_rules'])
        
        # Calculate combined score
        base_score = 1.0 - base_validation.get('false_positive_probability', 0.5)
        ml_score = ml_validation['ml_score']
        combined['combined_score'] = (base_score * 0.6 + ml_score * 0.4)
        
        # Update recommendation based on combined score
        if not combined['is_valid']:
            combined['recommendation'] = 'REJECT_SIGNAL'
        elif combined['combined_score'] >= 0.8:
            combined['recommendation'] = 'STRONG_SIGNAL'
        elif combined['combined_score'] >= 0.65:
            combined['recommendation'] = 'MODERATE_SIGNAL'
        elif combined['combined_score'] >= 0.5:
            combined['recommendation'] = 'WEAK_SIGNAL'
        else:
            combined['recommendation'] = 'REJECT_SIGNAL'
        
        # Add enhanced metadata
        combined['metadata'] = {
            'base_validation_score': base_score,
            'ml_validation_score': ml_score,
            'market_condition': combined.get('market_condition', 'normal'),
            'validation_timestamp': datetime.now()
        }
        
        return combined
    
    def update_performance(self, signal_id: str, outcome: str):
        """Update validator performance"""
        if outcome == 'success':
            self.validation_stats['true_positives'] += 1
        elif outcome == 'failure':
            self.validation_stats['false_positives'] += 1
        
        # Update base validator
        outcome_dict = {
            'profitable': (outcome == 'success'),
            'pattern_key': f'ml_ensemble_{outcome}',
            'time_key': f'{datetime.now().hour:02d}:00'
        }
        self.signal_validator.update_signal_outcome(signal_id, outcome_dict)
        
        # Adjust thresholds if needed
        if self.validation_stats['total_validations'] % 100 == 0:
            self._adjust_thresholds()
    
    def _adjust_thresholds(self):
        """Adjust validation thresholds based on performance"""
        total = self.validation_stats['total_validations']
        if total == 0:
            return
        
        # Calculate metrics
        acceptance_rate = self.validation_stats['accepted_signals'] / total
        if self.validation_stats['accepted_signals'] > 0:
            success_rate = (self.validation_stats['true_positives'] / 
                          (self.validation_stats['true_positives'] + self.validation_stats['false_positives']))
        else:
            success_rate = 0.5
        
        # Adjust thresholds
        if success_rate < 0.4:  # Too many false positives
            # Tighten thresholds
            for threshold in self.ml_validation_rules:
                if isinstance(self.ml_validation_rules[threshold], float):
                    self.ml_validation_rules[threshold] = min(
                        self.ml_validation_rules[threshold] * 1.05, 0.9
                    )
        elif success_rate > 0.7 and acceptance_rate < 0.3:  # Too conservative
            # Loosen thresholds
            for threshold in self.ml_validation_rules:
                if isinstance(self.ml_validation_rules[threshold], float):
                    self.ml_validation_rules[threshold] = max(
                        self.ml_validation_rules[threshold] * 0.95, 0.4
                    )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total = self.validation_stats['total_validations']
        if total == 0:
            return {'status': 'no_data'}
        
        accepted = self.validation_stats['accepted_signals']
        true_positives = self.validation_stats['true_positives']
        false_positives = self.validation_stats['false_positives']
        
        return {
            'total_validations': total,
            'acceptance_rate': accepted / total,
            'rejection_rate': self.validation_stats['rejected_signals'] / total,
            'success_rate': true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0,
            'false_positive_rate': false_positives / accepted if accepted > 0 else 0,
            'current_thresholds': self.ml_validation_rules.copy(),
            'ml_enhanced': True
        }


class IntegratedConfirmationValidationSystem:
    """
    Integrated system combining ML-enhanced confirmation and validation
    Provides a complete signal validation pipeline
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize integrated system"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.confirmation_system = MLEnhancedConfirmationSystem(config)
        self.signal_validator = MLEnhancedSignalValidator(config)
        
        # Integration parameters
        self.min_combined_score = config.get('min_combined_score', 0.65)
        self.require_confirmation = config.get('require_confirmation', True)
        
        # Performance tracking
        self.integration_stats = {
            'signals_processed': 0,
            'signals_approved': 0,
            'avg_processing_time': 0.0
        }
    
    def process_ensemble_signal(self, ensemble_signal: Dict, market_data: pd.DataFrame,
                              entry_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Process ensemble signal through complete validation pipeline
        
        Args:
            ensemble_signal: Signal from ML ensemble
            market_data: Market data
            entry_price: Proposed entry price
            
        Returns:
            Complete validation result with confirmations
        """
        start_time = datetime.now()
        
        # Prepare ML context
        ml_context = {
            'ensemble_signals': ensemble_signal.get('individual_signals', []),
            'ml_confidence': ensemble_signal.get('confidence', 0),
            'market_condition': self._detect_market_condition(market_data)
        }
        
        # Get confirmations if entry price provided
        confirmations = None
        if entry_price and self.require_confirmation:
            confirmations = self.confirmation_system.get_confirmations(
                ensemble_signal['signal_type'],
                market_data,
                entry_price,
                ml_context
            )
            
            # Add confirmation score to context
            if confirmations:
                ml_context['confirmation_score'] = confirmations.get('confluence_score', 0)
        
        # Validate signal
        validation = self.signal_validator.validate_ensemble_signal(
            ensemble_signal,
            market_data,
            ml_context
        )
        
        # Combine results
        result = {
            'signal': ensemble_signal,
            'validation': validation,
            'confirmations': confirmations,
            'is_approved': False,
            'combined_score': 0.0,
            'processing_time': (datetime.now() - start_time).total_seconds()
        }
        
        # Calculate combined score
        validation_score = validation.get('combined_score', 0)
        confirmation_score = confirmations.get('confluence_score', 0.5) if confirmations else 0.5
        
        if self.require_confirmation and confirmations:
            result['combined_score'] = validation_score * 0.6 + confirmation_score * 0.4
        else:
            result['combined_score'] = validation_score
        
        # Final approval
        result['is_approved'] = (
            validation['recommendation'] in ['STRONG_SIGNAL', 'MODERATE_SIGNAL'] and
            result['combined_score'] >= self.min_combined_score
        )
        
        # Update statistics
        self.integration_stats['signals_processed'] += 1
        if result['is_approved']:
            self.integration_stats['signals_approved'] += 1
        
        # Update average processing time
        self.integration_stats['avg_processing_time'] = (
            (self.integration_stats['avg_processing_time'] * 
             (self.integration_stats['signals_processed'] - 1) +
             result['processing_time']) /
            self.integration_stats['signals_processed']
        )
        
        return result
    
    def _detect_market_condition(self, market_data: pd.DataFrame) -> str:
        """Simple market condition detection"""
        if len(market_data) < 20:
            return 'normal'
        
        # Calculate volatility
        returns = market_data['close'].pct_change()
        volatility = returns.std()
        
        # Calculate trend
        sma_20 = market_data['close'].rolling(20).mean()
        trend_strength = abs((market_data['close'].iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1])
        
        # Classify market condition
        # Use a simpler approach - check if current volatility is above a basic threshold
        if volatility > 0.02:  # Basic volatility threshold
            return 'high_volatility'
        
        # Check trend strength
        if trend_strength > 0.02:
            return 'trending'
        else:
            return 'ranging'
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get complete system summary"""
        return {
            'integration_stats': self.integration_stats.copy(),
            'confirmation_performance': self.confirmation_system.get_performance_summary(),
            'validation_performance': self.signal_validator.get_performance_summary(),
            'approval_rate': (self.integration_stats['signals_approved'] / 
                            self.integration_stats['signals_processed']
                            if self.integration_stats['signals_processed'] > 0 else 0),
            'avg_processing_time_ms': self.integration_stats['avg_processing_time'] * 1000
        }