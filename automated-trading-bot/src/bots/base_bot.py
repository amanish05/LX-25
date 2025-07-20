"""
Base Bot Class
Abstract base class for all trading bots with ML ensemble integration
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import json
import os
from pathlib import Path
import pandas as pd

from ..core.database import DatabaseManager
from ..integrations.openalgo_client import OpenAlgoClient
from ..utils.logger import TradingLogger

# ML Ensemble System Imports
from ..ml.indicator_ensemble import IndicatorEnsemble, EnsembleConfig
from ..ml.models.rsi_lstm_model import RSILSTMModel
from ..ml.models.pattern_cnn_model import PatternCNNModel  
from ..ml.models.adaptive_thresholds_rl import AdaptiveThresholdsRL
from ..ml.models.confirmation_wrappers import IntegratedConfirmationValidationSystem

# Traditional Indicators
try:
    from ..indicators.rsi_advanced import AdvancedRSI
    from ..indicators.oscillator_matrix import OscillatorMatrix
    from ..indicators.price_action_composite import PriceActionComposite
    from ..indicators.advanced_confirmation import AdvancedConfirmationSystem
    from ..indicators.signal_validator import SignalValidator
except ImportError:
    # Fallback for direct execution
    AdvancedRSI = None
    OscillatorMatrix = None 
    PriceActionComposite = None
    AdvancedConfirmationSystem = None
    SignalValidator = None


class BotState(Enum):
    """Bot operational states"""
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class PositionState(Enum):
    """Position states"""
    PENDING = "pending"
    OPENING = "opening"
    OPEN = "open"
    ADJUSTING = "adjusting"
    CLOSING = "closing"
    CLOSED = "closed"


class BaseBot(ABC):
    """Abstract base class for all trading bots"""
    
    def __init__(self, config: Dict[str, Any], db_manager: DatabaseManager,
                 openalgo_client: OpenAlgoClient, logger: TradingLogger = None):
        self.config = config
        self.db_manager = db_manager
        self.openalgo_client = openalgo_client
        self.logger = logger or TradingLogger(__name__, config.get("name", "BaseBot"))
        
        # Bot identity and configuration
        self.name = config["name"]
        self.bot_type = config.get("bot_type", "unknown")
        self.enabled = config.get("enabled", True)
        
        # Capital management
        self.initial_capital = config.get("capital", 0)
        self.current_capital = self.initial_capital
        self.available_capital = self.initial_capital
        self.locked_capital = 0
        
        # Position management
        self.max_positions = config.get("max_positions", 1)
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.open_orders: Dict[str, Dict[str, Any]] = {}
        
        # Risk parameters
        self.risk_config = config.get("risk", {})
        self.max_position_size = config.get("max_trade_size", self.initial_capital * 0.2)
        
        # Strategy parameters
        self.entry_conditions = config.get("entry", {})
        self.exit_conditions = config.get("exit", {})
        
        # Symbols to monitor
        self.symbols: Set[str] = set()
        self.symbol_data: Dict[str, Dict[str, Any]] = {}
        
        # ML Ensemble System (replaces CompositeIndicators)
        self.indicators = None  # Will be initialized in initialize() method
        self.ml_config = None
        
        # State management
        self.state = BotState.INITIALIZED
        self.last_health_check = datetime.now()
        self.error_count = 0
        self.max_errors = 10
        
        # Performance tracking
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0,
            "max_drawdown": 0,
            "current_drawdown": 0
        }
        
        # Tasks
        self._tasks: List[asyncio.Task] = []
        self._market_data_handler = None
    
    @abstractmethod
    async def initialize(self):
        """Initialize the bot - load models, set parameters, etc."""
        pass
    
    @abstractmethod
    async def generate_signals(self, symbol: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate trading signals based on market data"""
        pass
    
    @abstractmethod
    async def calculate_position_size(self, signal: Dict[str, Any]) -> int:
        """Calculate position size based on signal and risk management"""
        pass
    
    @abstractmethod
    async def should_enter_position(self, signal: Dict[str, Any]) -> bool:
        """Determine if position entry conditions are met"""
        pass
    
    @abstractmethod
    async def should_exit_position(self, position: Dict[str, Any], 
                                  current_data: Dict[str, Any]) -> bool:
        """Determine if position exit conditions are met"""
        pass
    
    async def _initialize_ml_ensemble(self):
        """Initialize ML ensemble system with models and indicators"""
        try:
            # Load ML configuration
            config_path = Path("config/ml_models_config.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.ml_config = json.load(f)
            else:
                self.logger.warning("ML config not found, using default configuration")
                self.ml_config = self._get_default_ml_config()
            
            # Create ensemble configuration
            ensemble_config_data = self.ml_config.get("ensemble_config", {})
            ensemble_config = EnsembleConfig(
                weights=ensemble_config_data.get("weights", {}),
                indicator_weights=ensemble_config_data.get("indicator_weights", {}),
                min_consensus_ratio=ensemble_config_data.get("min_consensus_ratio", 0.6),
                min_confidence=ensemble_config_data.get("min_confidence", 0.5),
                adaptive_weights=ensemble_config_data.get("adaptive_weights", True),
                performance_window=ensemble_config_data.get("performance_window", 100)
            )
            
            # Initialize indicator ensemble
            self.indicators = IndicatorEnsemble(ensemble_config)
            
            # Initialize confirmation and validation system
            confirmation_config = {
                'min_combined_score': self.ml_config.get('min_combined_score', 0.65),
                'require_confirmation': self.ml_config.get('require_confirmation', True),
                'ml_validator_config': self.ml_config.get('price_action_ml_config', {}).get('validator_config')
            }
            self.confirmation_validator = IntegratedConfirmationValidationSystem(confirmation_config)
            
            # Add ML models to ensemble
            await self._add_ml_models_to_ensemble()
            
            # Add traditional indicators to ensemble
            await self._add_traditional_indicators_to_ensemble()
            
            self.logger.info("ML ensemble system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML ensemble: {e}")
            # Fallback to simple ensemble without models
            self.indicators = IndicatorEnsemble()
            raise
    
    async def _add_ml_models_to_ensemble(self):
        """Add ML models to the ensemble"""
        try:
            ml_models_config = self.ml_config.get("ml_models", {})
            
            # Initialize RSI LSTM Model
            if ml_models_config.get("rsi_lstm", {}).get("enabled", False):
                try:
                    rsi_lstm_config = ml_models_config["rsi_lstm"]
                    rsi_lstm_model = RSILSTMModel(
                        sequence_length=rsi_lstm_config.get("sequence_length", 25),
                        hidden_units=rsi_lstm_config.get("hidden_units", 64),
                        num_layers=rsi_lstm_config.get("num_layers", 2),
                        dropout=rsi_lstm_config.get("dropout", 0.2)
                    )
                    
                    # Load trained model if exists
                    model_path = rsi_lstm_config.get("model_path", "models/rsi_lstm_model.pkl")
                    if os.path.exists(model_path):
                        rsi_lstm_model.load_model(model_path)
                        self.logger.info("Loaded RSI LSTM model from disk")
                    
                    self.indicators.add_ml_model("rsi_lstm", rsi_lstm_model)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to add RSI LSTM model: {e}")
            
            # Initialize Pattern CNN Model
            if ml_models_config.get("pattern_cnn", {}).get("enabled", False):
                try:
                    pattern_cnn_config = ml_models_config["pattern_cnn"]
                    pattern_cnn_model = PatternCNNModel(
                        image_size=pattern_cnn_config.get("image_size", [64, 64]),
                        num_channels=pattern_cnn_config.get("num_channels", 1)
                    )
                    
                    # Load trained model if exists
                    model_path = pattern_cnn_config.get("model_path", "models/pattern_cnn_model.pkl")
                    if os.path.exists(model_path):
                        pattern_cnn_model.load_model(model_path)
                        self.logger.info("Loaded Pattern CNN model from disk")
                    
                    self.indicators.add_ml_model("pattern_cnn", pattern_cnn_model)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to add Pattern CNN model: {e}")
            
            # Initialize Adaptive Thresholds RL Model
            if ml_models_config.get("adaptive_thresholds", {}).get("enabled", False):
                try:
                    adaptive_config = ml_models_config["adaptive_thresholds"]
                    adaptive_model = AdaptiveThresholdsRL(
                        lookback_window=adaptive_config.get("environment_params", {}).get("lookback_window", 50)
                    )
                    
                    # Load trained model if exists
                    model_path = adaptive_config.get("model_path", "models/adaptive_thresholds_model.pkl")
                    if os.path.exists(model_path):
                        adaptive_model.load_model(model_path)
                        self.logger.info("Loaded Adaptive Thresholds model from disk")
                    
                    self.indicators.add_ml_model("adaptive_thresholds", adaptive_model)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to add Adaptive Thresholds model: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error adding ML models to ensemble: {e}")
    
    async def _add_traditional_indicators_to_ensemble(self):
        """Add traditional indicators to the ensemble"""
        try:
            traditional_config = self.ml_config.get("traditional_indicators", {})
            
            # Add Advanced RSI
            if traditional_config.get("advanced_rsi", {}).get("enabled", False) and AdvancedRSI:
                try:
                    rsi_config = traditional_config["advanced_rsi"]
                    advanced_rsi = AdvancedRSI(
                        period=rsi_config.get("period", 14),
                        overbought=rsi_config.get("overbought", 70),
                        oversold=rsi_config.get("oversold", 30)
                    )
                    self.indicators.add_traditional_indicator("advanced_rsi", advanced_rsi)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to add Advanced RSI: {e}")
            
            # Add Oscillator Matrix
            if traditional_config.get("oscillator_matrix", {}).get("enabled", False) and OscillatorMatrix:
                try:
                    oscillator_config = traditional_config["oscillator_matrix"]
                    oscillator_matrix = OscillatorMatrix(
                        indicators=oscillator_config.get("indicators", ["rsi", "macd", "stochastic"]),
                        weights=oscillator_config.get("weights", {})
                    )
                    self.indicators.add_traditional_indicator("oscillator_matrix", oscillator_matrix)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to add Oscillator Matrix: {e}")
            
            # Add Price Action Composite or ML-Enhanced Price Action
            if traditional_config.get("price_action_composite", {}).get("enabled", False):
                try:
                    # Check if ML-enhanced price action is enabled
                    price_action_ml_config = self.ml_config.get("price_action_ml_config", {})
                    if price_action_ml_config.get("enabled", False):
                        # Use ML-enhanced price action system
                        from ..ml.models.price_action_ml_wrapper import MLEnhancedPriceActionSystem
                        price_action = MLEnhancedPriceActionSystem(price_action_ml_config)
                        self.indicators.add_traditional_indicator("price_action_composite", price_action)
                        self.logger.info("Added ML-enhanced Price Action System to ensemble")
                    elif PriceActionComposite:
                        # Use traditional price action
                        pa_config = traditional_config["price_action_composite"]
                        price_action = PriceActionComposite(
                            weights=pa_config.get("weights", {}),
                            min_signal_strength=pa_config.get("min_strength", 40)
                        )
                        self.indicators.add_traditional_indicator("price_action_composite", price_action)
                        self.logger.info("Added traditional Price Action Composite to ensemble")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to add Price Action Composite: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error adding traditional indicators to ensemble: {e}")
    
    def _get_default_ml_config(self) -> Dict[str, Any]:
        """Get default ML configuration if config file is missing"""
        return {
            "ensemble_config": {
                "weights": {
                    "ml_models": 0.4,
                    "technical_indicators": 0.6
                },
                "min_consensus_ratio": 0.6,
                "min_confidence": 0.5,
                "adaptive_weights": True
            },
            "ml_models": {},
            "traditional_indicators": {}
        }
    
    def _convert_to_dataframe(self, data: Dict[str, Any]) -> 'pd.DataFrame':
        """Convert market data to pandas DataFrame for ensemble processing"""
        import pandas as pd
        import numpy as np
        from datetime import datetime
        
        # Handle different data formats
        if isinstance(data, dict):
            if 'timestamp' in data or 'time' in data:
                # Single data point - need historical data
                # Try to get historical data from symbol_data cache
                symbol_history = list(self.symbol_data.values())
                if len(symbol_history) < 20:
                    # Not enough data, create minimal DataFrame
                    return pd.DataFrame({
                        'open': [data.get('open', data.get('ltp', 100))],
                        'high': [data.get('high', data.get('ltp', 100))],
                        'low': [data.get('low', data.get('ltp', 100))],
                        'close': [data.get('close', data.get('ltp', 100))],
                        'volume': [data.get('volume', 10000)]
                    }, index=[datetime.now()])
                else:
                    # Build DataFrame from historical data
                    df_data = []
                    for hist_data in symbol_history[-50:]:  # Last 50 data points
                        df_data.append({
                            'open': hist_data.get('open', hist_data.get('ltp', 100)),
                            'high': hist_data.get('high', hist_data.get('ltp', 100)),
                            'low': hist_data.get('low', hist_data.get('ltp', 100)),
                            'close': hist_data.get('close', hist_data.get('ltp', 100)),
                            'volume': hist_data.get('volume', 10000)
                        })
                    
                    timestamps = pd.date_range(end=datetime.now(), periods=len(df_data), freq='5min')
                    return pd.DataFrame(df_data, index=timestamps)
            else:
                # Assume it's already formatted data
                return pd.DataFrame(data)
        else:
            # Fallback: create minimal DataFrame
            return pd.DataFrame({
                'open': [100], 'high': [100], 'low': [100], 'close': [100], 'volume': [10000]
            }, index=[datetime.now()])
    
    def _enhance_signal_with_ensemble(self, bot_signal: Dict[str, Any], 
                                    ensemble_signal) -> Dict[str, Any]:
        """Enhance bot signal with ML ensemble insights"""
        try:
            # Get bot-specific ML configuration
            bot_config = self.ml_config.get("bot_specific_settings", {}).get(self.bot_type, {})
            
            if not bot_config.get("use_ml_ensemble", False):
                return bot_signal
            
            ml_weight = bot_config.get("ml_weight", 0.4)
            traditional_weight = bot_config.get("traditional_weight", 0.6)
            min_ensemble_strength = bot_config.get("min_ensemble_strength", 0.6)
            
            # Check if ensemble meets minimum strength requirement
            if ensemble_signal.strength < min_ensemble_strength:
                self.logger.debug(f"Ensemble strength {ensemble_signal.strength:.2f} below threshold {min_ensemble_strength}")
                return bot_signal
            
            # Check signal alignment
            bot_signal_type = bot_signal.get("type", "NEUTRAL")
            ensemble_signal_type = ensemble_signal.signal_type.upper()
            
            if bot_signal_type == ensemble_signal_type:
                # Signals agree - enhance strength
                original_strength = bot_signal.get("strength", 0.5)
                enhanced_strength = (original_strength * traditional_weight + 
                                   ensemble_signal.strength * ml_weight)
                
                bot_signal["strength"] = min(enhanced_strength, 1.0)
                bot_signal["ml_enhanced"] = True
                bot_signal["ensemble_confidence"] = ensemble_signal.confidence
                bot_signal["consensus_ratio"] = ensemble_signal.consensus_ratio
                
                # Add target and stop from ensemble if available
                if ensemble_signal.target_price:
                    bot_signal["ml_target"] = ensemble_signal.target_price
                if ensemble_signal.stop_loss:
                    bot_signal["ml_stop"] = ensemble_signal.stop_loss
                if ensemble_signal.risk_reward_ratio:
                    bot_signal["ml_risk_reward"] = ensemble_signal.risk_reward_ratio
                
                self.logger.info(f"Signal enhanced with ML: {bot_signal_type} "
                               f"strength {original_strength:.2f} → {enhanced_strength:.2f}")
            
            elif bot_signal_type in ["BUY", "SELL"] and ensemble_signal_type in ["BUY", "SELL"]:
                # Conflicting signals - reduce strength
                if ensemble_signal.confidence > 0.7:
                    bot_signal["strength"] *= 0.5  # Significantly reduce strength
                    bot_signal["ml_conflict"] = True
                    bot_signal["conflict_reason"] = f"Bot: {bot_signal_type}, ML: {ensemble_signal_type}"
                    
                    self.logger.warning(f"ML ensemble conflicts with bot signal: "
                                      f"{bot_signal_type} vs {ensemble_signal_type}")
            
            # Add ensemble metadata
            bot_signal["ensemble_metadata"] = {
                "signal_type": ensemble_signal.signal_type,
                "strength": ensemble_signal.strength,
                "confidence": ensemble_signal.confidence,
                "consensus_ratio": ensemble_signal.consensus_ratio,
                "contributing_indicators": ensemble_signal.contributing_indicators
            }
            
            return bot_signal
            
        except Exception as e:
            self.logger.error(f"Error enhancing signal with ensemble: {e}")
            return bot_signal
    
    async def start(self):
        """Start the bot"""
        try:
            self.state = BotState.STARTING
            self.logger.info(f"Starting bot {self.name}")
            
            # Initialize ML ensemble system first
            await self._initialize_ml_ensemble()
            
            # Initialize bot-specific components
            await self.initialize()
            
            # Initialize capital in database
            await self.db_manager.init_bot_capital(self.name, self.initial_capital)
            
            # Load existing positions
            await self._load_positions()
            
            # Subscribe to market data
            await self._subscribe_to_market_data()
            
            # Start background tasks
            self._tasks.append(asyncio.create_task(self._position_monitor()))
            self._tasks.append(asyncio.create_task(self._health_check()))
            
            self.state = BotState.RUNNING
            self.logger.info(f"Bot {self.name} started successfully")
            
        except Exception as e:
            self.state = BotState.ERROR
            self.logger.error(f"Failed to start bot: {e}")
            raise
    
    async def stop(self):
        """Stop the bot"""
        try:
            self.state = BotState.STOPPING
            self.logger.info(f"Stopping bot {self.name}")
            
            # Cancel all tasks
            for task in self._tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            
            # Unsubscribe from market data
            await self._unsubscribe_from_market_data()
            
            # Close any pending orders
            await self._cancel_pending_orders()
            
            self.state = BotState.STOPPED
            self.logger.info(f"Bot {self.name} stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping bot: {e}")
            raise
    
    async def pause(self):
        """Pause bot operations"""
        self.state = BotState.PAUSED
        self.logger.info(f"Bot {self.name} paused")
    
    async def resume(self):
        """Resume bot operations"""
        if self.state == BotState.PAUSED:
            self.state = BotState.RUNNING
            self.logger.info(f"Bot {self.name} resumed")
    
    async def on_market_data(self, symbol: str, data: Dict[str, Any]):
        """Handle incoming market data"""
        if self.state != BotState.RUNNING:
            return
        
        try:
            # Update symbol data cache
            self.symbol_data[symbol] = data
            
            # Generate ML ensemble signal first (if we have enough data)
            ensemble_signal = None
            if self.indicators and len(self.symbol_data.get(symbol, {})) > 0:
                try:
                    # Convert data to pandas DataFrame for ensemble
                    import pandas as pd
                    df_data = self._convert_to_dataframe(data)
                    
                    # Generate ensemble signal
                    ensemble_signal = self.indicators.generate_ensemble_signal(df_data)
                    
                    if ensemble_signal:
                        self.logger.debug(f"ML ensemble generated signal: {ensemble_signal.signal_type} "
                                        f"(strength: {ensemble_signal.strength:.2f}, "
                                        f"confidence: {ensemble_signal.confidence:.2f})")
                        
                except Exception as e:
                    self.logger.debug(f"ML ensemble signal generation failed: {e}")
            
            # Generate bot-specific signals (traditional logic)
            signal = await self.generate_signals(symbol, data)
            
            # Enhance signal with ML ensemble insights
            if signal and ensemble_signal:
                signal = self._enhance_signal_with_ensemble(signal, ensemble_signal)
            
            # Apply confirmation and validation system
            if signal and hasattr(self, 'confirmation_validator'):
                # Convert to pandas DataFrame for validation
                df_data = self._convert_to_dataframe(data) if not isinstance(data, pd.DataFrame) else data
                
                # Process through confirmation and validation pipeline
                validation_result = self.confirmation_validator.process_ensemble_signal(
                    ensemble_signal.__dict__ if ensemble_signal else signal,
                    df_data,
                    entry_price=signal.get('entry_price')
                )
                
                if not validation_result['is_approved']:
                    self.logger.info(f"Signal rejected by confirmation/validation system: "
                                   f"combined_score={validation_result['combined_score']:.2f}")
                    signal = None
                else:
                    # Enhance signal with validation data
                    signal['validation'] = validation_result
                    signal['ml_confidence'] = validation_result['combined_score']
                    self.logger.info(f"Signal approved with confidence: {signal['ml_confidence']:.2f}")
            
            if signal:
                # Save signal to database
                signal_id = await self.db_manager.save_signal(
                    self.name, symbol, signal.get("exchange", "NSE"),
                    signal["type"], signal.get("strength", 0.5), signal
                )
                
                # Process signal
                await self._process_signal(signal, signal_id)
            
            # Check existing positions
            await self._check_positions(symbol, data)
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error processing market data: {e}")
            
            if self.error_count >= self.max_errors:
                self.state = BotState.ERROR
                self.logger.error(f"Bot {self.name} entering error state")
    
    async def _process_signal(self, signal: Dict[str, Any], signal_id: int):
        """Process trading signal"""
        symbol = signal["symbol"]
        signal_type = signal["type"]
        
        # Check if we should act on this signal
        if signal_type in ["BUY", "SELL"]:
            # Check position limits
            if len(self.positions) >= self.max_positions:
                self.logger.warning(f"Position limit reached for {self.name}")
                return
            
            # Check if we already have a position in this symbol
            if symbol in self.positions:
                self.logger.info(f"Already have position in {symbol}")
                return
            
            # Check entry conditions
            if await self.should_enter_position(signal):
                # Calculate position size
                position_size = await self.calculate_position_size(signal)
                
                if position_size > 0:
                    # Place order
                    success = await self._place_order(signal, position_size)
                    
                    if success:
                        await self.db_manager.mark_signal_executed(signal_id)
                        self.logger.info(f"Executed signal {signal_id} for {symbol}")
    
    async def _place_order(self, signal: Dict[str, Any], quantity: int) -> bool:
        """Place order with OpenAlgo"""
        try:
            symbol = signal["symbol"]
            action = signal["type"]  # BUY or SELL
            
            # Determine order parameters
            order_params = {
                "symbol": symbol,
                "exchange": signal.get("exchange", "NSE"),
                "action": action,
                "quantity": quantity,
                "product": self.config.get("product", "MIS"),
                "price_type": "MARKET",  # Can be made configurable
            }
            
            # Add price for limit orders
            if order_params["price_type"] == "LIMIT":
                order_params["price"] = signal.get("price", 0)
            
            # Place order
            response = await self.openalgo_client.place_smart_order(
                strategy=self.name,
                position_size=quantity,
                **order_params
            )
            
            if response.get("status") == "success":
                order_id = response.get("orderid")
                
                # Track order
                self.open_orders[order_id] = {
                    "symbol": symbol,
                    "action": action,
                    "quantity": quantity,
                    "signal": signal,
                    "placed_at": datetime.now()
                }
                
                # Create position entry
                position_id = await self.db_manager.create_position(
                    self.name, symbol, order_params["exchange"],
                    "LONG" if action == "BUY" else "SHORT",
                    quantity, signal.get("price", 0),
                    {"signal": signal}
                )
                
                # Update capital
                await self._update_capital_allocation(quantity * signal.get("price", 0))
                
                return True
            else:
                self.logger.error(f"Order placement failed: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return False
    
    async def _check_positions(self, symbol: str, data: Dict[str, Any]):
        """Check existing positions for exit conditions"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Check exit conditions
        if await self.should_exit_position(position, data):
            await self._close_position(position)
    
    async def _close_position(self, position: Dict[str, Any]):
        """Close a position"""
        try:
            symbol = position["symbol"]
            quantity = position["quantity"]
            position_type = position["position_type"]
            
            # Determine close action
            action = "SELL" if position_type == "LONG" else "BUY"
            
            # Place closing order
            response = await self.openalgo_client.close_position(
                symbol=symbol,
                exchange=position.get("exchange", "NSE"),
                product=self.config.get("product", "MIS"),
                position_size=quantity
            )
            
            if response.get("status") == "success":
                # Update position in database
                await self.db_manager.update_position(
                    position["id"],
                    status="CLOSED",
                    exit_time=datetime.now(),
                    exit_price=self.symbol_data.get(symbol, {}).get("ltp", 0)
                )
                
                # Remove from active positions
                del self.positions[symbol]
                
                # Update capital
                await self._update_capital_allocation(-position.get("locked_capital", 0))
                
                self.logger.info(f"Closed position in {symbol}")
                
                # Update stats
                await self._update_performance_stats(position)
                
            else:
                self.logger.error(f"Failed to close position: {response}")
                
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    async def _load_positions(self):
        """Load existing positions from database"""
        positions = await self.db_manager.get_open_positions(self.name)
        
        for pos in positions:
            self.positions[pos["symbol"]] = pos
            self.locked_capital += pos.get("locked_capital", 0)
        
        self.available_capital = self.current_capital - self.locked_capital
        self.logger.info(f"Loaded {len(positions)} existing positions")
    
    async def _subscribe_to_market_data(self):
        """Subscribe to market data for relevant symbols"""
        # Get symbols from configuration or strategy
        symbols = list(self.symbols)
        
        if symbols:
            await self.openalgo_client.subscribe_market_data(
                symbols=symbols,
                feed_type="quote",
                callback=self.on_market_data
            )
            self.logger.info(f"Subscribed to market data for {symbols}")
    
    async def _unsubscribe_from_market_data(self):
        """Unsubscribe from market data"""
        symbols = list(self.symbols)
        
        if symbols:
            await self.openalgo_client.unsubscribe_market_data(symbols)
            self.logger.info(f"Unsubscribed from market data")
    
    async def _cancel_pending_orders(self):
        """Cancel all pending orders"""
        for order_id in list(self.open_orders.keys()):
            try:
                await self.openalgo_client.cancel_order(order_id)
                del self.open_orders[order_id]
            except Exception as e:
                self.logger.error(f"Error canceling order {order_id}: {e}")
    
    async def _update_capital_allocation(self, amount: float):
        """Update capital allocation"""
        self.locked_capital += amount
        self.available_capital = self.current_capital - self.locked_capital
        
        await self.db_manager.update_bot_capital(
            self.name,
            locked_capital=self.locked_capital
        )
    
    async def _update_performance_stats(self, closed_position: Dict[str, Any]):
        """Update performance statistics"""
        pnl = closed_position.get("pnl", 0)
        
        self.stats["total_trades"] += 1
        
        if pnl > 0:
            self.stats["winning_trades"] += 1
        else:
            self.stats["losing_trades"] += 1
        
        self.stats["total_pnl"] += pnl
        self.current_capital += pnl
        
        # Update drawdown
        if self.current_capital < self.initial_capital:
            drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
            self.stats["current_drawdown"] = drawdown
            self.stats["max_drawdown"] = max(self.stats["max_drawdown"], drawdown)
        else:
            self.stats["current_drawdown"] = 0
        
        # Update database
        await self.db_manager.update_bot_performance(self.name)
    
    async def _position_monitor(self):
        """Monitor positions periodically"""
        while self.state in [BotState.RUNNING, BotState.PAUSED]:
            try:
                if self.state == BotState.RUNNING:
                    # Update position prices
                    for symbol, position in self.positions.items():
                        if symbol in self.symbol_data:
                            current_price = self.symbol_data[symbol].get("ltp", 0)
                            await self.db_manager.update_position(
                                position["id"],
                                current_price=current_price
                            )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in position monitor: {e}")
                await asyncio.sleep(60)
    
    async def _health_check(self):
        """Periodic health check"""
        while self.state in [BotState.RUNNING, BotState.PAUSED]:
            try:
                # Check connection to OpenAlgo
                funds = await self.openalgo_client.get_funds()
                
                # Update capital from broker
                if funds:
                    available_funds = funds.get("available_balance", 0)
                    self.logger.debug(f"Health check - Available funds: {available_funds}")
                
                # Reset error count on successful health check
                self.error_count = 0
                self.last_health_check = datetime.now()
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Health check failed: {e}")
                await asyncio.sleep(60)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        return {
            "name": self.name,
            "type": self.bot_type,
            "state": self.state.value,
            "enabled": self.enabled,
            "capital": {
                "initial": self.initial_capital,
                "current": self.current_capital,
                "available": self.available_capital,
                "locked": self.locked_capital
            },
            "positions": len(self.positions),
            "open_orders": len(self.open_orders),
            "performance": {
                "total_trades": self.stats["total_trades"],
                "win_rate": (self.stats["winning_trades"] / self.stats["total_trades"] * 100) 
                           if self.stats["total_trades"] > 0 else 0,
                "total_pnl": self.stats["total_pnl"],
                "max_drawdown": self.stats["max_drawdown"]
            },
            "last_health_check": self.last_health_check,
            "error_count": self.error_count
        }