"""
Bot Manager Module
Manages multiple trading bots asynchronously
"""

import asyncio
from typing import Dict, Any, List, Optional, Type
from datetime import datetime
import importlib
import signal

from ..bots.base_bot import BaseBot, BotState
from ..core.database import DatabaseManager
from ..integrations.openalgo_client import OpenAlgoClient
from ..utils.logger import TradingLogger
from ..config import ConfigManager, BOT_CONSTANTS


class BotManager:
    """Manages multiple trading bots and coordinates their activities"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = TradingLogger(__name__)
        
        # Core components
        self.db_manager: Optional[DatabaseManager] = None
        self.openalgo_client: Optional[OpenAlgoClient] = None
        
        # Bot registry
        self.bots: Dict[str, BaseBot] = {}
        self.bot_tasks: Dict[str, asyncio.Task] = {}
        
        # System state
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.start_time = datetime.now()
        self.system_stats = {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_pnl": 0,
            "active_bots": 0
        }
        
        # Bot classes mapping
        self.bot_classes = {
            BOT_CONSTANTS.TYPE_SHORT_STRADDLE: "src.bots.short_straddle_bot.ShortStraddleBot",
            BOT_CONSTANTS.TYPE_IRON_CONDOR: "src.bots.iron_condor_bot.IronCondorBot",
            BOT_CONSTANTS.TYPE_VOLATILITY_EXPANDER: "src.bots.volatility_expander_bot.VolatilityExpanderBot",
            BOT_CONSTANTS.TYPE_MOMENTUM_RIDER: "src.bots.momentum_rider_bot.MomentumRiderBot"
        }
    
    async def initialize(self):
        """Initialize the bot manager and its components"""
        self.logger.info("Initializing Bot Manager")
        
        try:
            # Initialize database
            self.db_manager = DatabaseManager(self.config_manager.app_config.to_dict())
            await self.db_manager.init_database()
            
            # Initialize OpenAlgo client
            self.openalgo_client = OpenAlgoClient(self.config_manager.app_config.to_dict())
            
            # Test connection
            funds = await self.openalgo_client.get_funds()
            self.logger.info(f"Connected to OpenAlgo. Available funds: {funds.get('available_balance', 0)}")
            
            # Initialize bots
            await self._initialize_bots()
            
            self.logger.info("Bot Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Bot Manager: {e}")
            raise
    
    async def _initialize_bots(self):
        """Initialize all enabled bots from configuration"""
        # Get bot configurations from trading parameters
        bot_configs = {
            BOT_CONSTANTS.TYPE_SHORT_STRADDLE: self.config_manager.get_bot_config(BOT_CONSTANTS.TYPE_SHORT_STRADDLE),
            BOT_CONSTANTS.TYPE_IRON_CONDOR: self.config_manager.get_bot_config(BOT_CONSTANTS.TYPE_IRON_CONDOR),
            BOT_CONSTANTS.TYPE_VOLATILITY_EXPANDER: self.config_manager.get_bot_config(BOT_CONSTANTS.TYPE_VOLATILITY_EXPANDER),
            BOT_CONSTANTS.TYPE_MOMENTUM_RIDER: self.config_manager.get_bot_config(BOT_CONSTANTS.TYPE_MOMENTUM_RIDER)
        }
        
        for bot_type, bot_config in bot_configs.items():
            if not bot_config.get("enabled", False):
                self.logger.info(f"Bot {bot_type} is disabled, skipping")
                continue
            
            try:
                # Create bot instance
                bot = await self._create_bot(bot_type, bot_config)
                self.bots[bot.name] = bot
                
                self.logger.info(f"Initialized bot: {bot.name} ({bot_type})")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize bot {bot_type}: {e}")
    
    async def _create_bot(self, bot_type: str, bot_config: Dict[str, Any]) -> BaseBot:
        """Create a bot instance dynamically"""
        # Get bot class path
        class_path = self.bot_classes.get(bot_type)
        if not class_path:
            raise ValueError(f"Unknown bot type: {bot_type}")
        
        # Import bot class
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        bot_class = getattr(module, class_name)
        
        # Add bot type to config
        bot_config["bot_type"] = bot_type
        
        # Create bot instance
        bot = bot_class(
            config=bot_config,
            db_manager=self.db_manager,
            openalgo_client=self.openalgo_client
        )
        
        return bot
    
    async def start(self):
        """Start all bots and the manager"""
        if self.is_running:
            self.logger.warning("Bot Manager is already running")
            return
        
        self.logger.info("Starting Bot Manager")
        self.is_running = True
        
        try:
            # Initialize if not already done
            if not self.db_manager:
                await self.initialize()
            
            # Start WebSocket connection
            asyncio.create_task(self.openalgo_client.connect_websocket())
            
            # Start all bots
            await self._start_all_bots()
            
            # Start monitoring tasks
            asyncio.create_task(self._monitor_system())
            asyncio.create_task(self._cleanup_task())
            
            self.logger.info("Bot Manager started successfully")
            
            # Wait for shutdown
            await self.shutdown_event.wait()
            
        except Exception as e:
            self.logger.error(f"Error in Bot Manager: {e}")
            raise
        finally:
            await self.stop()
    
    async def _start_all_bots(self):
        """Start all initialized bots"""
        for bot_name, bot in self.bots.items():
            if bot.enabled:
                try:
                    # Start the bot
                    await bot.start()
                    
                    # Create monitoring task for the bot
                    task = asyncio.create_task(self._monitor_bot(bot))
                    self.bot_tasks[bot_name] = task
                    
                    self.system_stats["active_bots"] += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to start bot {bot_name}: {e}")
    
    async def stop(self):
        """Stop all bots and cleanup"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping Bot Manager")
        self.is_running = False
        
        try:
            # Stop all bots
            await self._stop_all_bots()
            
            # Close connections
            if self.openalgo_client:
                await self.openalgo_client.close()
            
            if self.db_manager:
                await self.db_manager.close()
            
            self.logger.info("Bot Manager stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping Bot Manager: {e}")
    
    async def _stop_all_bots(self):
        """Stop all running bots"""
        self.logger.info("Stopping all bots...")
        
        # Stop bots
        stop_tasks = []
        for bot_name, bot in self.bots.items():
            if bot.state != BotState.STOPPED:
                stop_tasks.append(bot.stop())
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Cancel monitoring tasks
        for task in self.bot_tasks.values():
            task.cancel()
        
        if self.bot_tasks:
            await asyncio.gather(*self.bot_tasks.values(), return_exceptions=True)
        
        self.bot_tasks.clear()
        self.system_stats["active_bots"] = 0
    
    async def _monitor_bot(self, bot: BaseBot):
        """Monitor individual bot health"""
        while self.is_running and bot.state != BotState.STOPPED:
            try:
                # Check bot state
                if bot.state == BotState.ERROR:
                    self.logger.error(f"Bot {bot.name} is in error state")
                    
                    # Attempt restart if configured
                    if self.config_manager.app_config.system.auto_restart_bots:
                        await self._restart_bot(bot)
                
                # Update system stats
                bot_status = bot.get_status()
                self._update_system_stats(bot_status)
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error monitoring bot {bot.name}: {e}")
                await asyncio.sleep(60)
    
    async def _restart_bot(self, bot: BaseBot):
        """Restart a bot that has failed"""
        try:
            self.logger.info(f"Attempting to restart bot {bot.name}")
            
            # Stop the bot
            await bot.stop()
            
            # Wait a bit
            await asyncio.sleep(5)
            
            # Start again
            await bot.start()
            
            self.logger.info(f"Successfully restarted bot {bot.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to restart bot {bot.name}: {e}")
    
    async def _monitor_system(self):
        """Monitor overall system health"""
        while self.is_running:
            try:
                # Log system status
                uptime = (datetime.now() - self.start_time).total_seconds() / 3600
                
                self.logger.info(
                    f"System Status - Uptime: {uptime:.2f}h, "
                    f"Active Bots: {self.system_stats['active_bots']}, "
                    f"Total Trades: {self.system_stats['total_trades']}, "
                    f"Total PnL: {self.system_stats['total_pnl']:.2f}"
                )
                
                # Check market hours
                if not self._is_market_hours():
                    self.logger.info("Outside market hours")
                
                # Update performance metrics
                for bot in self.bots.values():
                    if bot.state == BotState.RUNNING:
                        await self.db_manager.update_bot_performance(bot.name)
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in system monitor: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_task(self):
        """Periodic cleanup tasks"""
        while self.is_running:
            try:
                # Clean expired cache
                await self.db_manager.cleanup_expired_cache()
                
                # More cleanup tasks can be added here
                
                await asyncio.sleep(3600)  # Every hour
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(3600)
    
    def _update_system_stats(self, bot_status: Dict[str, Any]):
        """Update system statistics from bot status"""
        performance = bot_status.get("performance", {})
        
        # Update aggregated stats (this is simplified, should track deltas)
        self.system_stats["total_trades"] = sum(
            bot.get_status()["performance"]["total_trades"] 
            for bot in self.bots.values()
        )
        self.system_stats["total_pnl"] = sum(
            bot.get_status()["performance"]["total_pnl"] 
            for bot in self.bots.values()
        )
    
    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours"""
        now = datetime.now()
        
        # Skip weekends
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Market hours from config
        market_config = self.config_manager.get_market_hours()
        start_time = datetime.strptime(market_config["start"], "%H:%M").time()
        end_time = datetime.strptime(market_config["end"], "%H:%M").time()
        
        current_time = now.time()
        return start_time <= current_time <= end_time
    
    async def start_bot(self, bot_name: str):
        """Start a specific bot"""
        if bot_name not in self.bots:
            raise ValueError(f"Bot {bot_name} not found")
        
        bot = self.bots[bot_name]
        if bot.state == BotState.RUNNING:
            self.logger.warning(f"Bot {bot_name} is already running")
            return
        
        await bot.start()
        
        # Create monitoring task
        task = asyncio.create_task(self._monitor_bot(bot))
        self.bot_tasks[bot_name] = task
        
        self.system_stats["active_bots"] += 1
        self.logger.info(f"Started bot {bot_name}")
    
    async def stop_bot(self, bot_name: str):
        """Stop a specific bot"""
        if bot_name not in self.bots:
            raise ValueError(f"Bot {bot_name} not found")
        
        bot = self.bots[bot_name]
        await bot.stop()
        
        # Cancel monitoring task
        if bot_name in self.bot_tasks:
            self.bot_tasks[bot_name].cancel()
            del self.bot_tasks[bot_name]
        
        self.system_stats["active_bots"] -= 1
        self.logger.info(f"Stopped bot {bot_name}")
    
    async def pause_bot(self, bot_name: str):
        """Pause a specific bot"""
        if bot_name not in self.bots:
            raise ValueError(f"Bot {bot_name} not found")
        
        await self.bots[bot_name].pause()
        self.logger.info(f"Paused bot {bot_name}")
    
    async def resume_bot(self, bot_name: str):
        """Resume a specific bot"""
        if bot_name not in self.bots:
            raise ValueError(f"Bot {bot_name} not found")
        
        await self.bots[bot_name].resume()
        self.logger.info(f"Resumed bot {bot_name}")
    
    def get_bot_status(self, bot_name: str = None) -> Dict[str, Any]:
        """Get status of specific bot or all bots"""
        if bot_name:
            if bot_name not in self.bots:
                raise ValueError(f"Bot {bot_name} not found")
            return self.bots[bot_name].get_status()
        
        # Return all bot statuses
        return {
            name: bot.get_status() 
            for name, bot in self.bots.items()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            "is_running": self.is_running,
            "market_hours": self._is_market_hours(),
            "stats": self.system_stats,
            "bots": self.get_bot_status(),
            "config": {
                "environment": self.config_manager.app_config.system.environment,
                "total_capital": self.config_manager.app_config.system.total_capital,
                "available_capital": self.config_manager.app_config.system.available_capital
            }
        }
    
    def shutdown(self):
        """Signal shutdown to the bot manager"""
        self.logger.info("Shutdown requested")
        self.shutdown_event.set()