"""
Base Bot Class
Abstract base class for all trading bots
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import json

from ..core.database import DatabaseManager
from ..integrations.openalgo_client import OpenAlgoClient
from ..indicators.composite import CompositeIndicators
from ..utils.logger import TradingLogger


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
        
        # Indicators
        self.indicators = CompositeIndicators(config.get("indicators", {}))
        
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
    
    async def start(self):
        """Start the bot"""
        try:
            self.state = BotState.STARTING
            self.logger.info(f"Starting bot {self.name}")
            
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
            
            # Generate signals
            signal = await self.generate_signals(symbol, data)
            
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