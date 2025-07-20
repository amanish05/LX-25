"""
Unit tests for ML-Enhanced Price Action Validator
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from ml.models.price_action_ml_validator import (
    PriceActionMLValidator, 
    StructureBreakValidation
)


class TestPriceActionMLValidator(unittest.TestCase):
    """Test cases for Price Action ML Validator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = PriceActionMLValidator()
        
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
    
    def test_validator_initialization(self):
        """Test validator initialization"""
        self.assertIsNotNone(self.validator)
        self.assertIsNotNone(self.validator.config)
        self.assertEqual(self.validator.false_positive_rate, 0.3)
    
    def test_structure_break_validation(self):
        """Test structure break validation"""
        # Create a sample break event
        break_event = {
            'type': 'BOS',
            'confidence': 0.7,
            'price_change': 0.02,
            'levels_broken': 2,
            'swing_strength': 0.8,
            'pattern_clarity': 0.7,
            'confluence_score': 0.6,
            'sr_distance': 0.01
        }
        
        # Validate the break
        validation = self.validator.validate_structure_break(
            break_event, 
            self.market_data
        )
        
        # Check validation result
        self.assertIsInstance(validation, StructureBreakValidation)
        self.assertEqual(validation.break_type, 'BOS')
        self.assertIsInstance(validation.ml_confidence, float)
        self.assertIsInstance(validation.is_valid, bool)
        self.assertIsNotNone(validation.features)
    
    def test_feature_extraction(self):
        """Test feature extraction for structure breaks"""
        break_event = {
            'type': 'CHoCH',
            'confidence': 0.6,
            'price_change': 0.015,
            'levels_broken': 1,
            'swing_strength': 0.6,
            'pattern_clarity': 0.65,
            'confluence_score': 0.55,
            'sr_distance': 0.008
        }
        
        # Extract features
        features = self.validator._extract_break_features(
            break_event,
            self.market_data
        )
        
        # Check required features
        required_features = [
            'momentum_5', 'momentum_10', 'volume_ratio', 'volume_surge',
            'volatility', 'volatility_ratio', 'time_score',
            'break_magnitude', 'levels_broken', 'swing_strength',
            'trend_strength', 'support_resistance_distance',
            'pattern_clarity', 'confluence_score',
            'similar_breaks_success_rate'
        ]
        
        for feature in required_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (int, float))
    
    def test_rule_based_validation(self):
        """Test fallback rule-based validation"""
        features = {
            'momentum_5': 0.01,
            'momentum_10': 0.008,
            'volume_ratio': 1.6,
            'volume_surge': 1.0,
            'volatility': 0.02,
            'volatility_ratio': 1.3,
            'time_score': 0.9,
            'break_magnitude': 0.015,
            'levels_broken': 2,
            'swing_strength': 0.7,
            'trend_strength': 0.6,
            'support_resistance_distance': 0.01,
            'pattern_clarity': 0.75,
            'confluence_score': 0.7,
            'similar_breaks_success_rate': 0.6
        }
        
        # Get rule-based confidence
        confidence = self.validator._rule_based_validation(features)
        
        # Check confidence is valid
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # With these features, confidence should be relatively high
        self.assertGreater(confidence, 0.5)
    
    def test_order_block_validation(self):
        """Test order block validation"""
        order_block = {
            'type': 'bullish',
            'index': 50,
            'price_low': 99.5,
            'price_high': 100.5,
            'volume': 3000,
            'strength': 0.7,
            'confluence_score': 0.6,
            'age_bars': 10,
            'times_tested': 2,
            'at_significant_level': True
        }
        
        # Validate order block
        validation = self.validator.validate_order_block(
            order_block,
            self.market_data
        )
        
        # Check validation result
        self.assertIsInstance(validation, dict)
        self.assertIn('volume_score', validation)
        self.assertIn('structure_score', validation)
        self.assertIn('overall_score', validation)
        self.assertIn('is_valid', validation)
        self.assertIsInstance(validation['is_valid'], bool)
    
    def test_trend_strength_calculation(self):
        """Test trend strength calculation"""
        # Create trending price data
        prices = np.array([100, 101, 102, 103, 104, 105, 104, 106, 107, 108])
        trend_strength = self.validator._calculate_trend_strength(prices)
        
        # Check trend strength
        self.assertIsInstance(trend_strength, float)
        self.assertGreaterEqual(trend_strength, 0.0)
        self.assertLessEqual(trend_strength, 1.0)
        
        # Strong uptrend should have high trend strength
        self.assertGreater(trend_strength, 0.5)
    
    def test_performance_tracking(self):
        """Test performance tracking and updates"""
        # Add some validation results
        for i in range(10):
            self.validator.validation_history.append({
                'id': f'test_{i}',
                'break_type': 'BOS' if i % 2 == 0 else 'CHoCH',
                'ml_confidence': 0.6 + i * 0.02,
                'was_successful': i > 5
            })
        
        # Update performance
        self.validator.update_performance('test_5', True)
        
        # Check false positive rate update
        self.assertIsInstance(self.validator.false_positive_rate, float)
        
        # Get historical success rate
        success_rate = self.validator._get_historical_success_rate('BOS')
        self.assertIsInstance(success_rate, float)
    
    def test_high_volatility_rejection(self):
        """Test validation during high volatility"""
        # Create high volatility data
        volatile_data = self.market_data.copy()
        volatile_data['close'] = volatile_data['close'] + np.random.normal(0, 5, len(volatile_data))
        
        break_event = {
            'type': 'BOS',
            'confidence': 0.5,  # Lower confidence
            'price_change': 0.05,  # Large change
            'levels_broken': 1,
            'swing_strength': 0.4,
            'pattern_clarity': 0.4,
            'confluence_score': 0.4,
            'sr_distance': 0.02
        }
        
        validation = self.validator.validate_structure_break(
            break_event,
            volatile_data
        )
        
        # High volatility with low pattern quality should likely be rejected
        if validation.is_valid:
            self.assertLess(validation.ml_confidence, 0.7)
    
    def test_model_persistence(self):
        """Test model save and load functionality"""
        import tempfile
        
        # Train with some sample data
        training_data = []
        for i in range(60):
            features = {
                'momentum_5': np.random.normal(0, 0.01),
                'momentum_10': np.random.normal(0, 0.01),
                'volume_ratio': np.random.uniform(0.5, 2.0),
                'volume_surge': np.random.choice([0.0, 1.0]),
                'volatility': np.random.uniform(0.01, 0.03),
                'volatility_ratio': np.random.uniform(0.8, 1.2),
                'time_score': np.random.uniform(0.3, 1.0),
                'break_magnitude': np.random.uniform(0.005, 0.02),
                'levels_broken': np.random.randint(1, 4),
                'swing_strength': np.random.uniform(0.3, 0.9),
                'trend_strength': np.random.uniform(0.2, 0.8),
                'support_resistance_distance': np.random.uniform(0.005, 0.02),
                'pattern_clarity': np.random.uniform(0.4, 0.8),
                'confluence_score': np.random.uniform(0.3, 0.8),
                'similar_breaks_success_rate': np.random.uniform(0.3, 0.7)
            }
            training_data.append({
                'features': features,
                'was_successful': np.random.random() > 0.4
            })
        
        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, 'test_model')
            
            # Set up validator for saving
            self.validator.is_trained = True
            self.validator.validation_history = [{'test': 'data'}]
            self.validator.false_positive_rate = 0.25
            
            # Save
            self.validator.save_model(save_path)
            
            # Create new validator and load
            new_validator = PriceActionMLValidator()
            new_validator.load_model(save_path)
            
            # Check loaded state
            self.assertEqual(new_validator.false_positive_rate, 0.25)
            self.assertEqual(len(new_validator.validation_history), 1)
            self.assertTrue(new_validator.is_trained)


if __name__ == '__main__':
    unittest.main()