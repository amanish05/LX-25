"""
Integration test for ML ensemble system
Tests the complete flow from market data to ML-enhanced bot signals
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from src.ml.indicator_ensemble import IndicatorEnsemble, EnsembleConfig
from src.bots.momentum_rider_bot import MomentumRiderBot
from src.bots.short_straddle_bot import ShortStraddleBot
from src.bot_selection.market_regime_detector import MarketRegimeDetector
from src.bot_selection.smart_bot_orchestrator import SmartBotOrchestrator
from src.indicators.price_action_composite import PriceActionComposite
from src.indicators.rsi_advanced import AdvancedRSI
from src.indicators.oscillator_matrix import OscillatorMatrix


class MockDBManager:
    """Mock database manager for testing"""
    async def create_position(self, *args, **kwargs):
        return {"id": 1}
    
    async def update_position(self, *args, **kwargs):
        pass
    
    async def get_positions(self, *args, **kwargs):
        return []


class MockOpenAlgoClient:
    """Mock OpenAlgo client for testing"""
    async def place_order(self, *args, **kwargs):
        return {"status": "success", "orderid": "12345"}
    
    async def get_quote(self, *args, **kwargs):
        return {"ltp": 20000 + np.random.uniform(-100, 100)}
    
    async def get_option_chain(self, *args, **kwargs):
        return {}


def generate_test_data(periods: int = 500) -> pd.DataFrame:
    """Generate realistic test market data"""
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')
    
    # Generate price with trends and volatility
    np.random.seed(42)
    trend = np.cumsum(np.random.normal(0.0001, 0.01, periods))
    noise = np.random.normal(0, 0.005, periods)
    prices = 20000 * np.exp(trend + noise)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, periods)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, periods))),
        'close': prices,
        'volume': np.random.randint(100000, 500000, periods).astype(float)
    }, index=dates)
    
    # Ensure OHLC consistency
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


class TestMLEnsembleIntegration:
    """Test ML ensemble integration with trading bots"""
    
    def setup_method(self):
        """Setup test environment"""
        self.db_manager = MockDBManager()
        self.openalgo_client = MockOpenAlgoClient()
        self.test_data = generate_test_data()
    
    def test_indicator_ensemble_creation(self):
        """Test creating indicator ensemble with all components"""
        # Create ensemble configuration
        config = EnsembleConfig(
            weights={
                'ml_models': 0.4,
                'technical_indicators': 0.3,
                'price_action': 0.2,
                'confirmation_systems': 0.1
            },
            min_consensus_ratio=0.6,
            min_confidence=0.5
        )
        
        # Initialize ensemble
        ensemble = IndicatorEnsemble(config)
        
        # Add traditional indicators
        ensemble.add_traditional_indicator('price_action', PriceActionComposite())
        ensemble.add_traditional_indicator('advanced_rsi', AdvancedRSI())
        ensemble.add_traditional_indicator('oscillator_matrix', OscillatorMatrix())
        
        # Verify ensemble setup
        assert len(ensemble.traditional_indicators) == 3
        assert ensemble.config.weights['ml_models'] == 0.4
        
        # Generate ensemble signal
        signal = ensemble.generate_ensemble_signal(self.test_data)
        
        # Verify signal structure
        assert 'signal_type' in signal
        assert 'strength' in signal
        assert 'confidence' in signal
        assert 'consensus_ratio' in signal
        assert signal['strength'] >= 0 and signal['strength'] <= 1
    
    @pytest.mark.asyncio
    async def test_momentum_bot_with_ml(self):
        """Test MomentumRiderBot with ML ensemble integration"""
        # Bot configuration
        config = {
            'bot_type': 'momentum_rider',
            'name': 'Test Momentum Bot',
            'symbols': ['NIFTY'],
            'available_capital': 100000,
            'entry_conditions': {
                'momentum_threshold': 0.45,
                'volume_multiplier': 2.0
            },
            'exit_conditions': {
                'stop_loss_pct': 2.0,
                'take_profit_pct': 3.0
            },
            'position_sizing': {
                'base_size_pct': 1.0,
                'max_size_pct': 2.0
            }
        }
        
        # Create bot
        bot = MomentumRiderBot(config, self.db_manager, self.openalgo_client)
        await bot.initialize()
        
        # Test signal generation with ML
        signal = await bot.generate_signals('NIFTY', {
            'symbol': 'NIFTY',
            'ltp': self.test_data['close'].iloc[-1],
            'volume': self.test_data['volume'].iloc[-1],
            'data': self.test_data
        })
        
        if signal:
            # Verify ML enhancement
            assert 'ml_enhanced' in signal or 'ensemble_confidence' in signal
            assert 'strength' in signal
            assert signal['strength'] >= 0 and signal['strength'] <= 1
    
    @pytest.mark.asyncio
    async def test_straddle_bot_ml_filtering(self):
        """Test ShortStraddleBot ML filtering for directional signals"""
        # Bot configuration
        config = {
            'bot_type': 'short_straddle',
            'name': 'Test Straddle Bot',
            'symbols': ['NIFTY'],
            'available_capital': 200000,
            'entry_conditions': {
                'iv_rank_min': 72,
                'dte_min': 30,
                'dte_max': 45
            },
            'exit_conditions': {
                'profit_target_pct': 25,
                'stop_loss_multiplier': 1.5
            }
        }
        
        # Create bot
        bot = ShortStraddleBot(config, self.db_manager, self.openalgo_client)
        await bot.initialize()
        
        # Create a signal with strong directional ML prediction
        test_signal = {
            'symbol': 'NIFTY',
            'type': 'SHORT_STRADDLE',
            'ml_enhanced': True,
            'ensemble_confidence': 0.8,
            'ensemble_metadata': {
                'signal_type': 'BUY',  # Strong directional signal
                'strength': 0.75
            },
            'iv_rank': 75,
            'total_premium': 300
        }
        
        # Test ML filtering
        should_enter = await bot.should_enter_position(test_signal)
        
        # Should reject due to strong directional signal
        assert not should_enter
    
    def test_market_regime_detection(self):
        """Test market regime detection"""
        detector = MarketRegimeDetector()
        
        # Test with trending data
        trending_data = self.test_data.copy()
        trending_data['close'] = trending_data['close'] * np.linspace(1.0, 1.1, len(trending_data))
        
        regime = detector.detect_market_regime(trending_data)
        
        # Verify regime detection
        assert regime.regime_type in ['trending_up', 'trending_down', 'ranging', 'volatile', 'calm']
        assert regime.volatility_level in ['low', 'medium', 'high', 'extreme']
        assert regime.confidence >= 0 and regime.confidence <= 1
        
        # Test bot recommendations
        recommendations = detector.recommend_bots(regime)
        
        assert len(recommendations) > 0
        assert all(rec.score >= 0 and rec.score <= 1 for rec in recommendations)
        assert all(len(rec.reasons) > 0 for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_smart_orchestrator(self):
        """Test smart bot orchestrator"""
        # Orchestrator configuration
        config = {
            'total_capital': 500000,
            'min_capital_per_bot': 50000,
            'max_active_bots': 2,
            'min_bot_score': 0.5,
            'risk_tolerance': 'medium',
            'bot_configs': {
                'momentum_rider': {
                    'name': 'Momentum Test',
                    'symbols': ['NIFTY'],
                    'entry_conditions': {'momentum_threshold': 0.45}
                },
                'short_straddle': {
                    'name': 'Straddle Test',
                    'symbols': ['NIFTY'],
                    'entry_conditions': {'iv_rank_min': 72}
                }
            }
        }
        
        # Create orchestrator
        orchestrator = SmartBotOrchestrator(config, self.db_manager, self.openalgo_client)
        await orchestrator.initialize()
        
        # Test status
        status = orchestrator.get_status()
        
        assert 'is_running' in status
        assert 'current_regime' in status
        assert 'active_bots' in status
        assert 'capital_allocation' in status
        assert status['total_capital'] == 500000
    
    def test_ml_model_stubs(self):
        """Test ML model stub implementations"""
        # Test imports work
        try:
            from src.ml.models.rsi_lstm_model import RSILSTMModel
            from src.ml.models.pattern_cnn_model import PatternCNNModel
            from src.ml.models.adaptive_thresholds_rl import AdaptiveThresholdsRL
            
            # Models should be importable
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import ML models: {e}")
    
    def test_ensemble_signal_combination(self):
        """Test how ensemble combines different signals"""
        config = EnsembleConfig()
        ensemble = IndicatorEnsemble(config)
        
        # Add indicators
        ensemble.add_traditional_indicator('rsi', AdvancedRSI())
        ensemble.add_traditional_indicator('oscillator', OscillatorMatrix())
        
        # Generate signals
        signal = ensemble.generate_ensemble_signal(self.test_data)
        
        # Check metadata
        assert 'metadata' in signal
        assert 'contributing_indicators' in signal['metadata']
        assert 'individual_signals' in signal['metadata']
        
        # Verify consensus calculation
        if signal['signal_type'] != 'hold':
            assert signal['consensus_ratio'] > 0


class TestConfigurationIntegration:
    """Test configuration files integration"""
    
    def test_ml_config_loading(self):
        """Test ML configuration file"""
        import json
        from pathlib import Path
        
        config_path = Path('config/ml_models_config.json')
        assert config_path.exists(), "ML config file should exist"
        
        with open(config_path) as f:
            config = json.load(f)
        
        # Verify structure
        assert 'ensemble_config' in config
        assert 'model_configs' in config
        assert 'bot_specific_settings' in config
        
        # Verify ensemble weights sum to 1
        weights = config['ensemble_config']['weights']
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01, f"Weights should sum to 1, got {total_weight}"
    
    def test_bot_selection_config(self):
        """Test bot selection configuration"""
        import json
        from pathlib import Path
        
        config_path = Path('config/bot_selection_config.json')
        assert config_path.exists(), "Bot selection config should exist"
        
        with open(config_path) as f:
            config = json.load(f)
        
        # Verify structure
        assert 'lookback_periods' in config
        assert 'volatility_thresholds' in config
        assert 'bot_suitability' in config
        
        # Verify all regimes have bot scores
        regimes = ['trending_up', 'trending_down', 'ranging', 'volatile', 'calm']
        for regime in regimes:
            assert regime in config['bot_suitability']
            scores = config['bot_suitability'][regime]
            assert all(0 <= score <= 1 for score in scores.values())


@pytest.mark.integration
class TestEndToEndFlow:
    """Test complete end-to-end flow"""
    
    @pytest.mark.asyncio
    async def test_market_data_to_signal_flow(self):
        """Test flow from market data to final trading signal"""
        # Setup
        db_manager = MockDBManager()
        openalgo_client = MockOpenAlgoClient()
        test_data = generate_test_data()
        
        # 1. Detect market regime
        detector = MarketRegimeDetector()
        regime = detector.detect_market_regime(test_data)
        
        # 2. Get bot recommendations
        recommendations = detector.recommend_bots(regime, available_capital=100000)
        top_bot = recommendations[0]
        
        # 3. Create recommended bot
        if top_bot.bot_name == 'momentum_rider':
            bot = MomentumRiderBot({
                'bot_type': 'momentum_rider',
                'name': 'Test Bot',
                'symbols': ['NIFTY'],
                'available_capital': 100000,
                'entry_conditions': {'momentum_threshold': 0.45}
            }, db_manager, openalgo_client)
        else:
            # Use momentum bot as default for testing
            bot = MomentumRiderBot({
                'bot_type': 'momentum_rider',
                'name': 'Test Bot',
                'symbols': ['NIFTY'],
                'available_capital': 100000,
                'entry_conditions': {'momentum_threshold': 0.45}
            }, db_manager, openalgo_client)
        
        await bot.initialize()
        
        # 4. Generate signal with ML enhancement
        market_update = {
            'symbol': 'NIFTY',
            'ltp': test_data['close'].iloc[-1],
            'volume': test_data['volume'].iloc[-1],
            'data': test_data
        }
        
        signal = await bot.generate_signals('NIFTY', market_update)
        
        # 5. Verify complete flow
        assert regime is not None
        assert len(recommendations) > 0
        # Signal may or may not be generated based on conditions
        if signal:
            assert 'symbol' in signal
            assert 'strength' in signal


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])