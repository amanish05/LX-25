"""
Integration tests for Enhanced ML Ensemble System with Price Action and Confirmation
"""

import unittest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from ml.indicator_ensemble import IndicatorEnsemble, EnsembleConfig
from ml.models.price_action_ml_wrapper import MLEnhancedPriceActionSystem
from ml.models.confirmation_wrappers import IntegratedConfirmationValidationSystem
from indicators.rsi_advanced import AdvancedRSI
from indicators.oscillator_matrix import OscillatorMatrix
from indicators.advanced_confirmation import AdvancedConfirmationSystem
from indicators.signal_validator import SignalValidator


class TestMLEnsembleIntegration(unittest.TestCase):
    """Integration tests for ML Ensemble with all enhancements"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once"""
        # Load ML configuration
        config_path = Path('config/ml_models_config.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                cls.ml_config = json.load(f)
        else:
            cls.ml_config = cls._get_default_ml_config()
    
    def setUp(self):
        """Set up test fixtures"""
        # Create ensemble configuration
        ensemble_config = EnsembleConfig(
            weights=self.ml_config['ensemble_config']['weights'],
            indicator_weights=self.ml_config['ensemble_config']['indicator_weights'],
            min_consensus_ratio=0.6,
            min_confidence=0.5,
            adaptive_weights=True
        )
        
        # Initialize ensemble
        self.ensemble = IndicatorEnsemble(ensemble_config)
        
        # Initialize ML-enhanced price action
        self.ml_price_action = MLEnhancedPriceActionSystem(
            self.ml_config.get('price_action_ml_config', {})
        )
        
        # Initialize confirmation validation system
        self.confirmation_validator = IntegratedConfirmationValidationSystem({
            'min_combined_score': 0.65,
            'require_confirmation': True
        })
        
        # Add indicators to ensemble
        self._setup_indicators()
        
        # Create sample market data
        self.market_data = self._create_market_data()
    
    def _setup_indicators(self):
        """Set up indicators in ensemble"""
        # Add traditional indicators
        self.ensemble.add_traditional_indicator(
            'advanced_rsi',
            AdvancedRSI()
        )
        
        self.ensemble.add_traditional_indicator(
            'oscillator_matrix',
            OscillatorMatrix()
        )
        
        # Add ML-enhanced price action
        self.ensemble.add_traditional_indicator(
            'price_action_composite',
            self.ml_price_action
        )
    
    def _create_market_data(self, periods=200):
        """Create realistic market data"""
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')
        
        # Generate realistic price movement
        trend = np.sin(np.linspace(0, 4*np.pi, periods)) * 5
        noise = np.random.randn(periods) * 0.5
        base_price = 100
        
        close_prices = base_price + trend + noise.cumsum()
        
        # Create OHLC data
        data = pd.DataFrame({
            'open': close_prices + np.random.randn(periods) * 0.1,
            'high': close_prices + np.abs(np.random.randn(periods)) * 0.3,
            'low': close_prices - np.abs(np.random.randn(periods)) * 0.3,
            'close': close_prices,
            'volume': np.random.randint(50000, 200000, periods)
        }, index=dates)
        
        # Ensure OHLC consistency
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data
    
    @classmethod
    def _get_default_ml_config(cls):
        """Get default ML configuration"""
        return {
            'ensemble_config': {
                'weights': {
                    'ml_models': 0.4,
                    'technical_indicators': 0.3,
                    'price_action': 0.2,
                    'confirmation_systems': 0.1
                },
                'indicator_weights': {
                    'advanced_rsi': 0.15,
                    'oscillator_matrix': 0.15,
                    'price_action_composite': 0.3
                }
            },
            'price_action_ml_config': {
                'enabled': True,
                'validator_config': {
                    'bos_confidence_threshold': 0.7,
                    'choch_confidence_threshold': 0.75
                }
            }
        }
    
    def test_ensemble_signal_generation(self):
        """Test ensemble signal generation with all components"""
        # Generate ensemble signal
        signal = self.ensemble.generate_ensemble_signal(self.market_data)
        
        if signal:
            # Check signal structure
            self.assertIsNotNone(signal.signal_type)
            self.assertIn(signal.signal_type, ['buy', 'sell', 'hold'])
            self.assertGreater(signal.strength, 0)
            self.assertGreater(signal.confidence, 0)
            self.assertGreater(signal.consensus_ratio, 0)
            self.assertIsInstance(signal.contributing_indicators, list)
            
            # Should have contributions from our indicators
            indicator_names = signal.contributing_indicators
            self.assertTrue(any('rsi' in name for name in indicator_names))
    
    def test_ml_price_action_integration(self):
        """Test ML-enhanced price action integration"""
        # Analyze with ML price action
        analysis = self.ml_price_action.analyze(self.market_data)
        
        # Check analysis structure
        self.assertIsInstance(analysis, dict)
        self.assertIn('components', analysis)
        self.assertIn('composite_signal', analysis)
        self.assertTrue(analysis['ml_enhanced'])
        
        # Check components
        components = analysis['components']
        self.assertIn('structure_breaks', components)
        self.assertIn('order_blocks', components)
        self.assertIn('fair_value_gaps', components)
        self.assertIn('liquidity_zones', components)
        
        # Check ML metrics
        self.assertIn('ml_metrics', analysis)
    
    def test_confirmation_validation_pipeline(self):
        """Test complete confirmation and validation pipeline"""
        # Generate ensemble signal
        ensemble_signal = self.ensemble.generate_ensemble_signal(self.market_data)
        
        if ensemble_signal:
            # Process through confirmation validation
            result = self.confirmation_validator.process_ensemble_signal(
                ensemble_signal.__dict__,
                self.market_data,
                entry_price=self.market_data['close'].iloc[-1]
            )
            
            # Check result
            self.assertIsInstance(result, dict)
            self.assertIn('is_approved', result)
            self.assertIn('combined_score', result)
            self.assertIn('validation', result)
            self.assertIn('confirmations', result)
            
            # Check validation details
            validation = result['validation']
            self.assertIn('recommendation', validation)
            self.assertIn('ml_enhanced', validation)
            self.assertTrue(validation['ml_enhanced'])
    
    def test_adaptive_weight_adjustment(self):
        """Test adaptive weight adjustment in ensemble"""
        # Generate multiple signals and update performance
        for i in range(10):
            signal = self.ensemble.generate_ensemble_signal(
                self.market_data.iloc[i*10:(i+1)*10+50]
            )
            
            if signal:
                # Simulate performance update
                success = np.random.random() > 0.4
                self.ensemble.update_signal_performance(
                    str(signal.timestamp),
                    success,
                    return_value=np.random.normal(0.001, 0.01)
                )
        
        # Check if weights have been tracked
        summary = self.ensemble.get_ensemble_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('indicators', summary)
        
        # Check individual indicator performance
        for indicator_name, indicator_data in summary['indicators'].items():
            self.assertIn('weight', indicator_data)
            self.assertIn('performance', indicator_data)
    
    def test_high_volatility_scenario(self):
        """Test system behavior during high volatility"""
        # Create high volatility data
        volatile_data = self.market_data.copy()
        volatile_data['close'] = volatile_data['close'] * (1 + np.random.normal(0, 0.02, len(volatile_data)))
        volatile_data['high'] = volatile_data['high'] * (1 + np.abs(np.random.normal(0, 0.03, len(volatile_data))))
        volatile_data['low'] = volatile_data['low'] * (1 - np.abs(np.random.normal(0, 0.03, len(volatile_data))))
        
        # Generate signal in volatile conditions
        signal = self.ensemble.generate_ensemble_signal(volatile_data)
        
        if signal:
            # Process through validation
            result = self.confirmation_validator.process_ensemble_signal(
                signal.__dict__,
                volatile_data,
                entry_price=volatile_data['close'].iloc[-1]
            )
            
            # In high volatility, system should be more conservative
            if result['is_approved']:
                self.assertGreater(result['combined_score'], 0.7)  # Higher threshold
    
    def test_trending_market_scenario(self):
        """Test system behavior in trending market"""
        # Create strong uptrend
        periods = 100
        trend_data = self._create_market_data(periods)
        trend_data['close'] = 100 + np.linspace(0, 10, periods) + np.random.randn(periods) * 0.2
        
        # Ensure OHLC consistency
        trend_data['high'] = trend_data['close'] + np.abs(np.random.randn(periods)) * 0.1
        trend_data['low'] = trend_data['close'] - np.abs(np.random.randn(periods)) * 0.1
        trend_data['open'] = trend_data['close'].shift(1).fillna(trend_data['close'].iloc[0])
        
        # Generate signal
        signal = self.ensemble.generate_ensemble_signal(trend_data)
        
        if signal:
            # In strong uptrend, should likely generate buy signals
            if signal.signal_type == 'buy':
                self.assertGreater(signal.strength, 0.5)
    
    def test_ml_validator_false_positive_reduction(self):
        """Test ML validator's ability to reduce false positives"""
        # Create data with potential false signal patterns
        choppy_data = self.market_data.copy()
        
        # Add choppy price action
        for i in range(0, len(choppy_data), 5):
            choppy_data.loc[choppy_data.index[i:i+5], 'close'] += np.random.choice([-1, 1]) * 0.5
        
        # Analyze with ML price action
        analysis = self.ml_price_action.analyze(choppy_data)
        
        # Check structure breaks
        structure_breaks = analysis['components']['structure_breaks']
        if structure_breaks['total_detected'] > 0:
            # ML should filter out many false breaks
            rejection_rate = structure_breaks.get('rejection_rate', 0)
            self.assertGreater(rejection_rate, 0.3)  # Should reject some breaks
    
    def test_ensemble_summary_export(self):
        """Test ensemble summary and configuration export"""
        # Generate some signals first
        for i in range(5):
            self.ensemble.generate_ensemble_signal(
                self.market_data.iloc[i*20:(i+1)*20+50]
            )
        
        # Get summary
        summary = self.ensemble.get_ensemble_summary()
        
        # Check summary completeness
        self.assertIn('total_indicators', summary)
        self.assertIn('enabled_indicators', summary)
        self.assertIn('signals_generated', summary)
        self.assertGreater(summary['total_indicators'], 0)
        
        # Test save/load functionality
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            self.ensemble.save_ensemble(f.name)
            
            # Create new ensemble and load
            new_ensemble = IndicatorEnsemble()
            new_ensemble.load_ensemble(f.name)
            
            # Check configuration loaded correctly
            self.assertEqual(
                new_ensemble.config.min_consensus_ratio,
                self.ensemble.config.min_consensus_ratio
            )
    
    def test_performance_metrics_tracking(self):
        """Test comprehensive performance metrics tracking"""
        # Process multiple signals
        signals_processed = 0
        approved_signals = 0
        
        for i in range(20):
            data_slice = self.market_data.iloc[max(0, i*5):min(len(self.market_data), (i+1)*5+50)]
            if len(data_slice) < 50:
                continue
                
            signal = self.ensemble.generate_ensemble_signal(data_slice)
            
            if signal:
                signals_processed += 1
                
                # Process through validation
                result = self.confirmation_validator.process_ensemble_signal(
                    signal.__dict__,
                    data_slice,
                    entry_price=data_slice['close'].iloc[-1]
                )
                
                if result['is_approved']:
                    approved_signals += 1
        
        # Get system summary
        system_summary = self.confirmation_validator.get_system_summary()
        
        # Check metrics
        self.assertEqual(
            system_summary['integration_stats']['signals_processed'],
            signals_processed
        )
        self.assertEqual(
            system_summary['integration_stats']['signals_approved'],
            approved_signals
        )
        
        # Check approval rate is reasonable
        if signals_processed > 0:
            approval_rate = approved_signals / signals_processed
            self.assertGreater(approval_rate, 0.1)  # Not too restrictive
            self.assertLess(approval_rate, 0.9)     # Not too permissive


if __name__ == '__main__':
    unittest.main()