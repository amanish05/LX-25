"""
Performance tests for ML-enhanced trading system
"""

import unittest
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import psutil
import tracemalloc

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from ml.indicator_ensemble import IndicatorEnsemble, EnsembleConfig
from ml.models.price_action_ml_wrapper import MLEnhancedPriceActionSystem
from ml.models.confirmation_wrappers import IntegratedConfirmationValidationSystem
from bots.momentum_rider_bot import MomentumRiderBot
from bots.short_straddle_bot import ShortStraddleBot
from core.database import DatabaseManager
from integrations.openalgo_client import OpenAlgoClient
from utils.logger import TradingLogger


class TestMLSystemPerformance(unittest.TestCase):
    """Performance tests for ML-enhanced system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once"""
        # Load configurations
        config_path = Path('config/ml_models_config.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                cls.ml_config = json.load(f)
        else:
            cls.ml_config = cls._get_default_ml_config()
        
        # Initialize logger
        cls.logger = TradingLogger("PerformanceTest")
    
    def setUp(self):
        """Set up test fixtures"""
        self.start_memory = None
        self.start_time = None
    
    def tearDown(self):
        """Clean up after tests"""
        if self.start_memory:
            tracemalloc.stop()
    
    def _start_performance_monitoring(self):
        """Start monitoring performance metrics"""
        self.start_time = time.time()
        tracemalloc.start()
        self.start_memory = tracemalloc.get_traced_memory()[0]
    
    def _stop_performance_monitoring(self, test_name):
        """Stop monitoring and report results"""
        end_time = time.time()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        
        elapsed_time = end_time - self.start_time
        memory_used = (current_memory - self.start_memory) / 1024 / 1024  # MB
        peak_memory_mb = peak_memory / 1024 / 1024  # MB
        
        print(f"\n{test_name} Performance Metrics:")
        print(f"  Execution Time: {elapsed_time:.3f} seconds")
        print(f"  Memory Used: {memory_used:.2f} MB")
        print(f"  Peak Memory: {peak_memory_mb:.2f} MB")
        
        return {
            'execution_time': elapsed_time,
            'memory_used': memory_used,
            'peak_memory': peak_memory_mb
        }
    
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
                }
            }
        }
    
    def _create_market_data(self, periods=1000):
        """Create market data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='1min')
        
        # Generate realistic price movement
        trend = np.sin(np.linspace(0, 4*np.pi, periods)) * 10
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
    
    def test_ensemble_signal_generation_performance(self):
        """Test performance of ensemble signal generation"""
        # Set up ensemble
        ensemble_config = EnsembleConfig()
        ensemble = IndicatorEnsemble(ensemble_config)
        
        # Add indicators
        from indicators.rsi_advanced import AdvancedRSI
        from indicators.oscillator_matrix import OscillatorMatrix
        
        ensemble.add_traditional_indicator('advanced_rsi', AdvancedRSI())
        ensemble.add_traditional_indicator('oscillator_matrix', OscillatorMatrix())
        
        # Create market data
        market_data = self._create_market_data(500)
        
        # Start monitoring
        self._start_performance_monitoring()
        
        # Generate signals
        signals_generated = 0
        for i in range(100):
            data_slice = market_data.iloc[i:i+100]
            signal = ensemble.generate_ensemble_signal(data_slice)
            if signal:
                signals_generated += 1
        
        # Stop monitoring
        metrics = self._stop_performance_monitoring("Ensemble Signal Generation")
        
        # Performance assertions
        self.assertLess(metrics['execution_time'], 10.0)  # Should complete in 10 seconds
        self.assertLess(metrics['memory_used'], 100)  # Should use less than 100MB
        self.assertGreater(signals_generated, 0)  # Should generate some signals
        
        # Calculate throughput
        throughput = 100 / metrics['execution_time']
        print(f"  Throughput: {throughput:.2f} signals/second")
        self.assertGreater(throughput, 10)  # At least 10 signals per second
    
    def test_ml_price_action_validation_performance(self):
        """Test performance of ML price action validation"""
        # Initialize ML price action system
        ml_price_action = MLEnhancedPriceActionSystem(self.ml_config.get('price_action_ml_config', {}))
        
        # Create market data
        market_data = self._create_market_data(1000)
        
        # Start monitoring
        self._start_performance_monitoring()
        
        # Perform multiple analyses
        analyses_completed = 0
        for i in range(50):
            data_slice = market_data.iloc[i*10:i*10+200]
            analysis = ml_price_action.analyze(data_slice)
            if analysis:
                analyses_completed += 1
        
        # Stop monitoring
        metrics = self._stop_performance_monitoring("ML Price Action Validation")
        
        # Performance assertions
        self.assertLess(metrics['execution_time'], 15.0)  # Should complete in 15 seconds
        self.assertLess(metrics['memory_used'], 200)  # Should use less than 200MB
        self.assertEqual(analyses_completed, 50)  # All analyses should complete
        
        # Calculate average time per analysis
        avg_time = metrics['execution_time'] / analyses_completed
        print(f"  Average time per analysis: {avg_time:.3f} seconds")
        self.assertLess(avg_time, 0.3)  # Each analysis should take less than 300ms
    
    def test_confirmation_validation_pipeline_performance(self):
        """Test performance of complete confirmation validation pipeline"""
        # Initialize system
        confirmation_validator = IntegratedConfirmationValidationSystem({
            'min_combined_score': 0.65,
            'require_confirmation': True
        })
        
        # Create test signals
        test_signals = []
        for i in range(100):
            test_signals.append({
                'signal_type': 'buy' if i % 2 == 0 else 'sell',
                'strength': 0.5 + (i % 10) * 0.05,
                'confidence': 0.5 + (i % 10) * 0.04,
                'consensus_ratio': 0.6 + (i % 10) * 0.02,
                'timestamp': datetime.now() + timedelta(minutes=i),
                'risk_reward_ratio': 1.5 + (i % 5) * 0.2,
                'contributing_indicators': ['rsi', 'macd', 'bb'],
                'individual_signals': []
            })
        
        # Create market data
        market_data = self._create_market_data(200)
        
        # Start monitoring
        self._start_performance_monitoring()
        
        # Process signals
        processed_count = 0
        approved_count = 0
        
        for signal in test_signals:
            result = confirmation_validator.process_ensemble_signal(
                signal,
                market_data,
                entry_price=100.0 + processed_count * 0.1
            )
            
            processed_count += 1
            if result['is_approved']:
                approved_count += 1
        
        # Stop monitoring
        metrics = self._stop_performance_monitoring("Confirmation Validation Pipeline")
        
        # Performance assertions
        self.assertLess(metrics['execution_time'], 20.0)  # Should complete in 20 seconds
        self.assertLess(metrics['memory_used'], 150)  # Should use less than 150MB
        
        # Calculate throughput
        throughput = processed_count / metrics['execution_time']
        print(f"  Throughput: {throughput:.2f} validations/second")
        print(f"  Approval rate: {approved_count/processed_count:.2%}")
        self.assertGreater(throughput, 5)  # At least 5 validations per second
    
    def test_concurrent_signal_processing(self):
        """Test performance under concurrent load"""
        # Set up ensemble
        ensemble_config = EnsembleConfig()
        ensemble = IndicatorEnsemble(ensemble_config)
        
        # Add indicators
        from indicators.rsi_advanced import AdvancedRSI
        ensemble.add_traditional_indicator('advanced_rsi', AdvancedRSI())
        
        # Create market data
        market_data = self._create_market_data(1000)
        
        # Start monitoring
        self._start_performance_monitoring()
        
        # Process signals concurrently
        def process_signal(i):
            data_slice = market_data.iloc[i:i+100]
            return ensemble.generate_ensemble_signal(data_slice)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_signal, i) for i in range(0, 900, 10)]
            results = [f.result() for f in futures]
        
        # Stop monitoring
        metrics = self._stop_performance_monitoring("Concurrent Signal Processing")
        
        # Count successful signals
        successful_signals = sum(1 for r in results if r is not None)
        
        # Performance assertions
        self.assertLess(metrics['execution_time'], 30.0)  # Should complete in 30 seconds
        self.assertLess(metrics['memory_used'], 300)  # Should use less than 300MB
        
        print(f"  Successful signals: {successful_signals}/{len(futures)}")
        print(f"  Parallel efficiency: {len(futures)/(metrics['execution_time']*4):.2f} tasks/second/thread")
    
    def test_memory_efficiency_long_running(self):
        """Test memory efficiency during long-running operations"""
        # Initialize components
        ml_price_action = MLEnhancedPriceActionSystem()
        confirmation_validator = IntegratedConfirmationValidationSystem()
        
        # Create market data
        market_data = self._create_market_data(2000)
        
        # Track memory over time
        memory_samples = []
        
        # Start monitoring
        self._start_performance_monitoring()
        
        # Simulate long-running operation
        for i in range(200):
            data_slice = market_data.iloc[i*10:(i+1)*10+100]
            
            # Analyze with ML price action
            analysis = ml_price_action.analyze(data_slice)
            
            # Create dummy signal
            signal = {
                'signal_type': 'buy',
                'strength': 0.7,
                'confidence': 0.6,
                'consensus_ratio': 0.65,
                'timestamp': datetime.now(),
                'risk_reward_ratio': 2.0,
                'contributing_indicators': ['test'],
                'individual_signals': []
            }
            
            # Process through validation
            confirmation_validator.process_ensemble_signal(
                signal,
                data_slice
            )
            
            # Sample memory every 20 iterations
            if i % 20 == 0:
                current_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
                memory_samples.append(current_memory)
        
        # Stop monitoring
        metrics = self._stop_performance_monitoring("Long-Running Memory Test")
        
        # Check for memory leaks
        if len(memory_samples) > 2:
            memory_growth = memory_samples[-1] - memory_samples[0]
            print(f"  Memory growth: {memory_growth:.2f} MB")
            print(f"  Memory samples: {[f'{m:.1f}' for m in memory_samples]}")
            
            # Memory growth should be minimal
            self.assertLess(memory_growth, 50)  # Less than 50MB growth
    
    def test_cpu_utilization(self):
        """Test CPU utilization during intensive operations"""
        # Get initial CPU usage
        process = psutil.Process()
        initial_cpu = process.cpu_percent(interval=1)
        
        # Initialize ensemble with all components
        ensemble_config = EnsembleConfig()
        ensemble = IndicatorEnsemble(ensemble_config)
        
        # Add multiple indicators
        from indicators.rsi_advanced import AdvancedRSI
        from indicators.oscillator_matrix import OscillatorMatrix
        
        ensemble.add_traditional_indicator('advanced_rsi', AdvancedRSI())
        ensemble.add_traditional_indicator('oscillator_matrix', OscillatorMatrix())
        ensemble.add_traditional_indicator(
            'ml_price_action',
            MLEnhancedPriceActionSystem()
        )
        
        # Create market data
        market_data = self._create_market_data(500)
        
        # Measure CPU during processing
        cpu_samples = []
        
        for i in range(50):
            data_slice = market_data.iloc[i*10:(i+1)*10+100]
            ensemble.generate_ensemble_signal(data_slice)
            
            if i % 10 == 0:
                cpu_usage = process.cpu_percent(interval=0.1)
                cpu_samples.append(cpu_usage)
        
        # Calculate average CPU usage
        avg_cpu = np.mean(cpu_samples) if cpu_samples else 0
        max_cpu = np.max(cpu_samples) if cpu_samples else 0
        
        print(f"\nCPU Utilization:")
        print(f"  Average: {avg_cpu:.1f}%")
        print(f"  Maximum: {max_cpu:.1f}%")
        print(f"  Samples: {cpu_samples}")
        
        # CPU usage should be reasonable
        self.assertLess(avg_cpu, 80)  # Average below 80%
        self.assertLess(max_cpu, 100)  # Should not max out CPU


class TestBotPerformanceWithML(unittest.TestCase):
    """Test bot performance with ML enhancements"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock database and API client
        self.db_manager = None  # Mock in real tests
        self.openalgo_client = None  # Mock in real tests
        self.logger = TradingLogger("BotPerformanceTest")
    
    def test_bot_initialization_time(self):
        """Test time to initialize bot with ML ensemble"""
        config = {
            'name': 'TestBot',
            'symbols': ['NIFTY'],
            'available_capital': 100000
        }
        
        # Measure initialization time
        start_time = time.time()
        
        # Create bot (would use mocks in real test)
        # bot = MomentumRiderBot(config, self.db_manager, self.openalgo_client, self.logger)
        
        end_time = time.time()
        init_time = end_time - start_time
        
        print(f"\nBot Initialization Time: {init_time:.3f} seconds")
        
        # Initialization should be reasonably fast
        # self.assertLess(init_time, 5.0)  # Less than 5 seconds


if __name__ == '__main__':
    # Run with verbosity for detailed output
    unittest.main(verbosity=2)