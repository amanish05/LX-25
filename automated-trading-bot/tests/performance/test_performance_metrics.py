"""
Performance tests for the trading bot system
Tests processing speed, resource usage, and system performance
"""

import pytest
import time
import pandas as pd
import numpy as np
from datetime import datetime
import psutil
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.indicators.momentum import MomentumIndicators
from src.indicators.volatility import VolatilityIndicators
from src.indicators.price_action_composite import PriceActionComposite
from src.indicators.trend import TrendIndicators
from src.indicators.volume import VolumeIndicators


class TestPerformanceMetrics:
    """Test system performance metrics"""
    
    @pytest.fixture
    def large_dataset(self):
        """Generate large dataset for performance testing"""
        # 1 day of 1-minute data = 1440 periods
        periods = 1440
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='1min')
        
        # Generate realistic price movement
        base_price = 20000
        prices = base_price + np.cumsum(np.random.randn(periods) * 10)
        
        data = pd.DataFrame({
            'open': prices + np.random.uniform(-5, 5, periods),
            'high': prices + np.random.uniform(5, 20, periods),
            'low': prices - np.random.uniform(5, 20, periods),
            'close': prices,
            'volume': np.random.randint(50000, 200000, periods)
        }, index=dates)
        
        # Ensure float64
        for col in data.columns:
            data[col] = data[col].astype(np.float64)
        
        return data
    
    @pytest.fixture
    def medium_dataset(self):
        """Generate medium dataset for standard testing"""
        periods = 500
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')
        
        prices = 20000 + np.cumsum(np.random.randn(periods) * 20)
        
        data = pd.DataFrame({
            'open': prices + np.random.uniform(-10, 10, periods),
            'high': prices + np.random.uniform(10, 30, periods),
            'low': prices - np.random.uniform(10, 30, periods),
            'close': prices,
            'volume': np.random.randint(100000, 300000, periods).astype(float)
        }, index=dates)
        
        for col in data.columns:
            data[col] = data[col].astype(np.float64)
        
        return data
    
    def test_indicator_calculation_speed(self, large_dataset):
        """Test speed of indicator calculations"""
        # Test momentum indicators
        momentum = MomentumIndicators()
        start = time.time()
        momentum_results = momentum.calculate(large_dataset)
        momentum_time = time.time() - start
        
        assert momentum_time < 2.0  # Should complete in under 2 seconds
        assert len(momentum_results) > 0
        
        # Test volatility indicators
        volatility = VolatilityIndicators()
        start = time.time()
        volatility_results = volatility.calculate(large_dataset)
        volatility_time = time.time() - start
        
        assert volatility_time < 2.0
        assert len(volatility_results) > 0
        
        print(f"\nPerformance Results:")
        print(f"Momentum calculation: {momentum_time:.3f}s")
        print(f"Volatility calculation: {volatility_time:.3f}s")
    
    def test_price_action_performance(self, large_dataset):
        """Test price action composite performance"""
        pa = PriceActionComposite()
        
        start = time.time()
        result = pa.calculate(large_dataset)
        pa_time = time.time() - start
        
        assert pa_time < 5.0  # Should complete in under 5 seconds
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(large_dataset)
        
        print(f"Price Action calculation: {pa_time:.3f}s for {len(large_dataset)} periods")
    
    def test_all_indicators_performance(self, medium_dataset):
        """Test all indicators together"""
        indicators = {
            'Momentum': MomentumIndicators(),
            'Volatility': VolatilityIndicators(), 
            'Trend': TrendIndicators(),
            'Volume': VolumeIndicators()
        }
        
        total_start = time.time()
        results = {}
        
        for name, indicator in indicators.items():
            start = time.time()
            try:
                result = indicator.calculate(medium_dataset)
                elapsed = time.time() - start
                results[name] = {
                    'time': elapsed,
                    'count': len(result) if hasattr(result, '__len__') else 1
                }
            except Exception as e:
                results[name] = {
                    'time': 0,
                    'error': str(e)
                }
        
        total_time = time.time() - total_start
        
        print("\nAll Indicators Performance:")
        for name, result in results.items():
            if 'error' in result:
                print(f"{name}: ERROR - {result['error']}")
            else:
                print(f"{name}: {result['time']:.3f}s ({result['count']} results)")
        
        print(f"\nTotal time: {total_time:.3f}s")
        assert total_time < 10.0  # All indicators should complete in under 10 seconds
    
    def test_memory_efficiency(self, medium_dataset):
        """Test memory usage efficiency"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run calculations
        momentum = MomentumIndicators()
        momentum.calculate(medium_dataset)
        
        volatility = VolatilityIndicators()
        volatility.calculate(medium_dataset)
        
        pa = PriceActionComposite()
        pa.calculate(medium_dataset)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory Usage:")
        print(f"Initial: {initial_memory:.1f} MB")
        print(f"Final: {final_memory:.1f} MB")
        print(f"Increase: {memory_increase:.1f} MB")
        
        # Memory increase should be reasonable
        assert memory_increase < 500  # Less than 500MB increase
    
    def test_real_time_processing(self):
        """Test real-time data processing speed"""
        # Simulate real-time data stream
        periods = 100
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='5s')
        
        momentum = MomentumIndicators()
        processing_times = []
        
        for i in range(20, periods):
            # Simulate growing dataset
            data = pd.DataFrame({
                'open': np.random.randn(i) * 100 + 20000,
                'high': np.random.randn(i) * 100 + 20100,
                'low': np.random.randn(i) * 100 + 19900,
                'close': np.random.randn(i) * 100 + 20000,
                'volume': np.random.randint(50000, 150000, i).astype(float)
            }, index=dates[:i])
            
            for col in data.columns:
                data[col] = data[col].astype(np.float64)
            
            start = time.time()
            momentum.calculate(data)
            processing_times.append(time.time() - start)
        
        avg_time = np.mean(processing_times)
        max_time = np.max(processing_times)
        
        print(f"\nReal-time Processing:")
        print(f"Average time: {avg_time*1000:.1f}ms")
        print(f"Max time: {max_time*1000:.1f}ms")
        
        # Should be fast enough for real-time
        assert avg_time < 0.1  # Less than 100ms average
        assert max_time < 0.5  # Less than 500ms max
    
    def test_enhanced_system_performance(self):
        """Test the enhanced system performance metrics from the old file"""
        # This incorporates the analysis from test_enhanced_system.py
        periods = 1000
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')
        
        # Generate test scenarios
        scenarios = {
            'trending': np.cumsum(np.random.randn(periods)) * 50 + 20000,
            'volatile': 20000 + np.random.randn(periods) * 200,
            'ranging': 20000 + np.sin(np.linspace(0, 20*np.pi, periods)) * 100
        }
        
        results = {}
        
        for scenario_name, prices in scenarios.items():
            data = pd.DataFrame({
                'open': prices + np.random.uniform(-20, 20, periods),
                'high': prices + np.random.uniform(20, 50, periods),
                'low': prices - np.random.uniform(20, 50, periods),
                'close': prices,
                'volume': np.random.randint(100000, 500000, periods).astype(float)
            }, index=dates)
            
            for col in data.columns:
                data[col] = data[col].astype(np.float64)
            
            # Test Price Action Composite
            pa = PriceActionComposite()
            start = time.time()
            pa_result = pa.calculate(data)
            elapsed = time.time() - start
            
            signal_count = (pa_result['signal'] != 0).sum()
            avg_strength = pa_result[pa_result['signal'] != 0]['signal_strength'].mean() if signal_count > 0 else 0
            
            results[scenario_name] = {
                'time': elapsed,
                'signals': signal_count,
                'avg_strength': avg_strength
            }
        
        print("\nEnhanced System Performance by Market Condition:")
        for scenario, result in results.items():
            print(f"\n{scenario.upper()} Market:")
            print(f"  Processing time: {result['time']:.3f}s")
            print(f"  Signals generated: {result['signals']}")
            print(f"  Average signal strength: {result['avg_strength']:.1f}")
        
        # Performance assertions
        for result in results.values():
            assert result['time'] < 3.0  # Should process quickly
            
    def test_backtest_performance(self):
        """Test backtesting performance"""
        # Simulate 30 days of 5-minute data
        periods = 30 * 24 * 12  # 8640 periods
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')
        
        print(f"\nBacktest Performance Test ({periods} periods):")
        
        # Generate data
        start = time.time()
        prices = 20000 + np.cumsum(np.random.randn(periods) * 10)
        data = pd.DataFrame({
            'open': prices + np.random.uniform(-10, 10, periods),
            'high': prices + np.random.uniform(10, 30, periods),
            'low': prices - np.random.uniform(10, 30, periods),
            'close': prices,
            'volume': np.random.randint(100000, 500000, periods).astype(float)
        }, index=dates)
        
        for col in data.columns:
            data[col] = data[col].astype(np.float64)
        
        data_gen_time = time.time() - start
        print(f"Data generation: {data_gen_time:.3f}s")
        
        # Run indicators
        pa = PriceActionComposite()
        start = time.time()
        results = pa.calculate(data)
        calc_time = time.time() - start
        
        print(f"Indicator calculation: {calc_time:.3f}s")
        print(f"Rate: {periods/calc_time:.0f} periods/second")
        
        # Should handle large datasets efficiently
        assert calc_time < 30.0  # Should complete 30 days in under 30 seconds
        assert periods/calc_time > 200  # Should process at least 200 periods/second