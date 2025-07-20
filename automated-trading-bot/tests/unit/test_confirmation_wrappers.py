"""
Unit tests for ML-Enhanced Confirmation and Validation Wrappers
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from ml.models.confirmation_wrappers import (
    MLEnhancedConfirmationSystem,
    MLEnhancedSignalValidator,
    IntegratedConfirmationValidationSystem
)


class TestMLEnhancedConfirmationSystem(unittest.TestCase):
    """Test cases for ML-Enhanced Confirmation System"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.confirmation_system = MLEnhancedConfirmationSystem()
        
        # Create sample market data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
        self.market_data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 5000, 100)
        }, index=dates)
        
        # Ensure OHLC consistency
        self.market_data['high'] = self.market_data[['open', 'high', 'close']].max(axis=1)
        self.market_data['low'] = self.market_data[['open', 'low', 'close']].min(axis=1)
    
    def test_initialization(self):
        """Test confirmation system initialization"""
        self.assertIsNotNone(self.confirmation_system)
        self.assertIsNotNone(self.confirmation_system.confirmation_system)
        self.assertIsInstance(self.confirmation_system.ml_weight_adjustments, dict)
        self.assertEqual(len(self.confirmation_system.ml_weight_adjustments), 6)
    
    def test_get_confirmations(self):
        """Test getting ML-enhanced confirmations"""
        # Test buy signal confirmation
        confirmations = self.confirmation_system.get_confirmations(
            'buy',
            self.market_data,
            100.5,
            ml_context={'ml_confidence': 0.7}
        )
        
        # Check confirmation structure
        self.assertIsInstance(confirmations, dict)
        if confirmations.get('total_confirmations', 0) > 0:
            self.assertIn('confluence_score', confirmations)
            self.assertIn('ml_enhanced', confirmations)
            self.assertTrue(confirmations['ml_enhanced'])
            self.assertIn('ml_adjustments', confirmations)
    
    def test_ml_enhancements(self):
        """Test ML enhancement application"""
        # Create base confirmations
        base_confirmations = {
            'total_confirmations': 4,
            'confluence_score': 0.6,
            'false_positive_probability': 0.3,
            'signal_strength': 'MODERATE',
            'confirmations': {
                'trendline_break': {'confirmed': True, 'confidence': 0.7},
                'volume_confirmation': {'confirmed': True, 'confidence': 0.8},
                'momentum_alignment': {'confirmed': True, 'confidence': 0.6},
                'fvg_confirmation': {'confirmed': False, 'confidence': 0.0}
            }
        }
        
        # Apply ML enhancements
        ml_context = {
            'ml_confidence': 0.8,
            'ensemble_signals': [
                {'indicator_name': 'rsi_lstm', 'strength': 0.7},
                {'indicator_name': 'price_action', 'metadata': {'fvg_detected': True}}
            ]
        }
        
        enhanced = self.confirmation_system._apply_ml_enhancements(
            base_confirmations,
            ml_context
        )
        
        # Check enhancements
        self.assertGreaterEqual(enhanced['confluence_score'], base_confirmations['confluence_score'])
        self.assertLess(enhanced['false_positive_probability'], base_confirmations['false_positive_probability'])
    
    def test_ml_agreement_check(self):
        """Test ML agreement checking"""
        ensemble_signals = [
            {
                'indicator_name': 'rsi_advanced',
                'strength': 0.8,
                'metadata': {}
            },
            {
                'indicator_name': 'price_action_composite',
                'metadata': {'fvg_detected': True, 'volume_confirmed': True}
            }
        ]
        
        # Check momentum alignment agreement
        momentum_agreement = self.confirmation_system._check_ml_agreement(
            'momentum_alignment',
            ensemble_signals
        )
        self.assertIsInstance(momentum_agreement, float)
        self.assertGreaterEqual(momentum_agreement, 0.0)
        self.assertLessEqual(momentum_agreement, 1.0)
    
    def test_performance_tracking(self):
        """Test performance tracking and weight adjustment"""
        # Simulate multiple confirmations
        for i in range(60):
            confirmations = self.confirmation_system.get_confirmations(
                'buy' if i % 2 == 0 else 'sell',
                self.market_data,
                100.0 + i * 0.1
            )
        
        # Update performance
        self.confirmation_system.update_performance('test_1', True)
        self.confirmation_system.update_performance('test_2', False)
        
        # Check performance metrics
        performance = self.confirmation_system.get_performance_summary()
        self.assertIsInstance(performance, dict)
        self.assertIn('total_confirmations', performance)
        self.assertIn('avg_confluence_score', performance)


class TestMLEnhancedSignalValidator(unittest.TestCase):
    """Test cases for ML-Enhanced Signal Validator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.signal_validator = MLEnhancedSignalValidator()
        
        # Create sample market data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
        self.market_data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 5000, 100)
        }, index=dates)
        
        # Ensure OHLC consistency
        self.market_data['high'] = self.market_data[['open', 'high', 'close']].max(axis=1)
        self.market_data['low'] = self.market_data[['open', 'low', 'close']].min(axis=1)
    
    def test_initialization(self):
        """Test signal validator initialization"""
        self.assertIsNotNone(self.signal_validator)
        self.assertIsNotNone(self.signal_validator.signal_validator)
        self.assertIsInstance(self.signal_validator.ml_validation_rules, dict)
        self.assertIsInstance(self.signal_validator.adaptive_thresholds, dict)
    
    def test_validate_ensemble_signal(self):
        """Test ensemble signal validation"""
        ensemble_signal = {
            'signal_type': 'buy',
            'strength': 0.7,
            'confidence': 0.6,
            'consensus_ratio': 0.65,
            'timestamp': datetime.now(),
            'risk_reward_ratio': 2.0,
            'contributing_indicators': ['rsi_lstm', 'pattern_cnn', 'price_action'],
            'metadata': {}
        }
        
        ml_context = {
            'market_condition': 'trending',
            'confirmation_score': 0.7
        }
        
        # Validate signal
        validation = self.signal_validator.validate_ensemble_signal(
            ensemble_signal,
            self.market_data,
            ml_context
        )
        
        # Check validation result
        self.assertIsInstance(validation, dict)
        self.assertIn('is_valid', validation)
        self.assertIn('recommendation', validation)
        self.assertIn('ml_validation', validation)
        self.assertIn('combined_score', validation)
        self.assertTrue(validation['ml_enhanced'])
    
    def test_ml_validation_rules(self):
        """Test ML-specific validation rules"""
        ensemble_signal = {
            'consensus_ratio': 0.5,  # Below threshold
            'confidence': 0.4,  # Below threshold
            'risk_reward_ratio': 1.2,  # Below threshold
            'contributing_indicators': ['rsi_lstm']
        }
        
        # Apply ML validation
        base_validation = {'is_valid': True, 'failed_rules': [], 'false_positive_probability': 0.3}
        ml_validation = self.signal_validator._apply_ml_validation(
            ensemble_signal,
            base_validation,
            {'market_condition': 'normal'}
        )
        
        # Check validation failed
        self.assertFalse(ml_validation['passed'])
        self.assertGreater(len(ml_validation['failed_rules']), 0)
        self.assertLess(ml_validation['ml_score'], 0.5)
    
    def test_adaptive_thresholds(self):
        """Test adaptive threshold application"""
        # Test high volatility thresholds
        ensemble_signal = {
            'consensus_ratio': 0.65,
            'confidence': 0.55,
            'risk_reward_ratio': 2.0,
            'contributing_indicators': ['rsi_lstm', 'pattern_cnn']
        }
        
        # Should pass in normal conditions
        normal_validation = self.signal_validator._apply_ml_validation(
            ensemble_signal,
            {'is_valid': True, 'failed_rules': []},
            {'market_condition': 'normal'}
        )
        
        # Should fail in high volatility
        volatile_validation = self.signal_validator._apply_ml_validation(
            ensemble_signal,
            {'is_valid': True, 'failed_rules': []},
            {'market_condition': 'high_volatility'}
        )
        
        # Check different outcomes
        self.assertGreater(normal_validation['ml_score'], volatile_validation['ml_score'])
    
    def test_performance_tracking_and_adjustment(self):
        """Test performance tracking and threshold adjustment"""
        # Simulate multiple validations
        for i in range(110):
            ensemble_signal = {
                'signal_type': 'buy',
                'strength': 0.6 + i * 0.001,
                'confidence': 0.55 + i * 0.001,
                'consensus_ratio': 0.6 + i * 0.001,
                'risk_reward_ratio': 1.8,
                'timestamp': datetime.now(),
                'contributing_indicators': ['rsi_lstm', 'pattern_cnn']
            }
            
            validation = self.signal_validator.validate_ensemble_signal(
                ensemble_signal,
                self.market_data
            )
            
            # Update performance
            if i < 50:
                self.signal_validator.update_performance(f'signal_{i}', 'failure')
            else:
                self.signal_validator.update_performance(f'signal_{i}', 'success')
        
        # Check threshold adjustment occurred
        performance = self.signal_validator.get_performance_summary()
        self.assertIsInstance(performance, dict)
        self.assertIn('acceptance_rate', performance)
        self.assertIn('success_rate', performance)


class TestIntegratedConfirmationValidationSystem(unittest.TestCase):
    """Test cases for Integrated Confirmation Validation System"""
    
    def setUp(self):
        """Set up test fixtures"""
        config = {
            'min_combined_score': 0.65,
            'require_confirmation': True
        }
        self.integrated_system = IntegratedConfirmationValidationSystem(config)
        
        # Create sample market data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
        self.market_data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 5000, 100)
        }, index=dates)
        
        # Ensure OHLC consistency
        self.market_data['high'] = self.market_data[['open', 'high', 'close']].max(axis=1)
        self.market_data['low'] = self.market_data[['open', 'low', 'close']].min(axis=1)
    
    def test_initialization(self):
        """Test integrated system initialization"""
        self.assertIsNotNone(self.integrated_system)
        self.assertIsNotNone(self.integrated_system.confirmation_system)
        self.assertIsNotNone(self.integrated_system.signal_validator)
        self.assertEqual(self.integrated_system.min_combined_score, 0.65)
        self.assertTrue(self.integrated_system.require_confirmation)
    
    def test_process_ensemble_signal(self):
        """Test complete signal processing pipeline"""
        ensemble_signal = {
            'signal_type': 'buy',
            'strength': 0.75,
            'confidence': 0.7,
            'consensus_ratio': 0.7,
            'timestamp': datetime.now(),
            'risk_reward_ratio': 2.5,
            'contributing_indicators': ['rsi_lstm', 'pattern_cnn', 'price_action'],
            'individual_signals': [
                {'indicator_name': 'rsi_lstm', 'strength': 0.8},
                {'indicator_name': 'pattern_cnn', 'strength': 0.7}
            ]
        }
        
        # Process signal
        result = self.integrated_system.process_ensemble_signal(
            ensemble_signal,
            self.market_data,
            entry_price=100.5
        )
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('signal', result)
        self.assertIn('validation', result)
        self.assertIn('confirmations', result)
        self.assertIn('is_approved', result)
        self.assertIn('combined_score', result)
        self.assertIn('processing_time', result)
        
        # Check processing time is reasonable
        self.assertGreater(result['processing_time'], 0)
        self.assertLess(result['processing_time'], 1.0)  # Should be fast
    
    def test_market_condition_detection(self):
        """Test market condition detection"""
        # Create different market conditions
        
        # High volatility data
        volatile_data = self.market_data.copy()
        volatile_data['close'] = volatile_data['close'] + np.random.normal(0, 5, len(volatile_data))
        
        # Trending data
        trending_data = self.market_data.copy()
        trending_data['close'] = trending_data.index.hour + trending_data.index.minute / 60
        
        # Detect conditions
        normal_condition = self.integrated_system._detect_market_condition(self.market_data)
        volatile_condition = self.integrated_system._detect_market_condition(volatile_data)
        trending_condition = self.integrated_system._detect_market_condition(trending_data)
        
        # Check conditions
        self.assertIn(normal_condition, ['normal', 'ranging', 'trending', 'high_volatility'])
        self.assertEqual(volatile_condition, 'high_volatility')
    
    def test_signal_rejection(self):
        """Test signal rejection scenarios"""
        # Weak signal that should be rejected
        weak_signal = {
            'signal_type': 'buy',
            'strength': 0.3,
            'confidence': 0.3,
            'consensus_ratio': 0.4,
            'timestamp': datetime.now(),
            'risk_reward_ratio': 0.8,
            'contributing_indicators': ['rsi_lstm'],
            'individual_signals': []
        }
        
        result = self.integrated_system.process_ensemble_signal(
            weak_signal,
            self.market_data
        )
        
        # Should be rejected
        self.assertFalse(result['is_approved'])
        self.assertLess(result['combined_score'], self.integrated_system.min_combined_score)
    
    def test_system_summary(self):
        """Test system summary generation"""
        # Process a few signals
        for i in range(5):
            signal = {
                'signal_type': 'buy' if i % 2 == 0 else 'sell',
                'strength': 0.6 + i * 0.05,
                'confidence': 0.55 + i * 0.05,
                'consensus_ratio': 0.6 + i * 0.03,
                'timestamp': datetime.now(),
                'risk_reward_ratio': 1.5 + i * 0.3,
                'contributing_indicators': ['rsi_lstm', 'pattern_cnn'],
                'individual_signals': []
            }
            
            self.integrated_system.process_ensemble_signal(
                signal,
                self.market_data
            )
        
        # Get summary
        summary = self.integrated_system.get_system_summary()
        
        # Check summary structure
        self.assertIsInstance(summary, dict)
        self.assertIn('integration_stats', summary)
        self.assertIn('confirmation_performance', summary)
        self.assertIn('validation_performance', summary)
        self.assertIn('approval_rate', summary)
        self.assertIn('avg_processing_time_ms', summary)
        
        # Check stats
        self.assertEqual(summary['integration_stats']['signals_processed'], 5)
        self.assertGreaterEqual(summary['avg_processing_time_ms'], 0)


if __name__ == '__main__':
    unittest.main()