"""
Smart Bot Orchestrator
Manages multiple trading bots based on market regime detection
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import pandas as pd

from ..bots.base_bot import BaseBot
from ..bots.momentum_rider_bot import MomentumRiderBot
from ..bots.short_straddle_bot import ShortStraddleBot
# Note: IronCondorBot and VolatilityExpanderBot are planned but not implemented yet
from .market_regime_detector import MarketRegimeDetector, MarketRegime, BotRecommendation
from ..utils.logger import TradingLogger


class SmartBotOrchestrator:
    """
    Orchestrates multiple trading bots based on market conditions
    
    Features:
    1. Dynamic bot activation based on market regime
    2. Capital allocation across bots
    3. Risk management at portfolio level
    4. Performance monitoring and rebalancing
    5. Smooth transitions between regimes
    """
    
    def __init__(self, config: Dict[str, Any], db_manager, openalgo_client):
        """Initialize bot orchestrator"""
        self.config = config
        self.db_manager = db_manager
        self.openalgo_client = openalgo_client
        self.logger = TradingLogger("SmartBotOrchestrator")
        
        # Market regime detector
        self.regime_detector = MarketRegimeDetector()
        
        # Bot instances (initially inactive)
        self.available_bots = {}
        self.active_bots = {}
        
        # Capital management
        self.total_capital = config.get('total_capital', 1000000)
        self.capital_allocation = {}
        self.min_capital_per_bot = config.get('min_capital_per_bot', 50000)
        
        # Regime tracking
        self.current_regime = None
        self.last_regime_change = datetime.now()
        self.regime_change_cooldown = timedelta(
            minutes=config.get('regime_change_cooldown_minutes', 30)
        )
        
        # Performance tracking
        self.performance_history = []
        self.bot_performance = {}
        
        # Control
        self.is_running = False
        self._tasks = []
        
    async def initialize(self):
        """Initialize all available bots"""
        self.logger.info("Initializing Smart Bot Orchestrator")
        
        # Create bot instances
        bot_configs = self.config.get('bot_configs', {})
        
        # Momentum Rider Bot
        if 'momentum_rider' in bot_configs:
            bot_config = bot_configs['momentum_rider'].copy()
            bot_config['available_capital'] = 0  # Will be allocated dynamically
            self.available_bots['momentum_rider'] = MomentumRiderBot(
                bot_config, self.db_manager, self.openalgo_client, self.logger
            )
        
        # Short Straddle Bot
        if 'short_straddle' in bot_configs:
            bot_config = bot_configs['short_straddle'].copy()
            bot_config['available_capital'] = 0
            self.available_bots['short_straddle'] = ShortStraddleBot(
                bot_config, self.db_manager, self.openalgo_client, self.logger
            )
        
        # Iron Condor Bot (TODO: Implement IronCondorBot)
        # if 'iron_condor' in bot_configs:
        #     bot_config = bot_configs['iron_condor'].copy()
        #     bot_config['available_capital'] = 0
        #     self.available_bots['iron_condor'] = IronCondorBot(
        #         bot_config, self.db_manager, self.openalgo_client, self.logger
        #     )
        
        # Volatility Expander Bot (TODO: Implement VolatilityExpanderBot)
        # if 'volatility_expander' in bot_configs:
        #     bot_config = bot_configs['volatility_expander'].copy()
        #     bot_config['available_capital'] = 0
        #     self.available_bots['volatility_expander'] = VolatilityExpanderBot(
        #         bot_config, self.db_manager, self.openalgo_client, self.logger
        #     )
        
        # Initialize all bots (but don't start them)
        for bot_name, bot in self.available_bots.items():
            await bot.initialize()
            self.logger.info(f"Initialized bot: {bot_name}")
        
        # Load performance history
        self._load_performance_history()
        
        self.logger.info(f"Orchestrator initialized with {len(self.available_bots)} available bots")
    
    async def start(self):
        """Start the orchestrator"""
        self.logger.info("Starting Smart Bot Orchestrator")
        self.is_running = True
        
        # Start monitoring tasks
        self._tasks.append(asyncio.create_task(self._monitor_market_regime()))
        self._tasks.append(asyncio.create_task(self._monitor_performance()))
        self._tasks.append(asyncio.create_task(self._rebalance_capital()))
        
        # Initial regime detection and bot activation
        await self._update_active_bots()
        
        self.logger.info("Smart Bot Orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator and all bots"""
        self.logger.info("Stopping Smart Bot Orchestrator")
        self.is_running = False
        
        # Stop all active bots
        for bot_name, bot in self.active_bots.items():
            await bot.stop()
            self.logger.info(f"Stopped bot: {bot_name}")
        
        # Cancel monitoring tasks
        for task in self._tasks:
            task.cancel()
        
        # Save performance history
        self._save_performance_history()
        
        self.logger.info("Smart Bot Orchestrator stopped")
    
    async def _monitor_market_regime(self):
        """Monitor market regime and adjust bots accordingly"""
        while self.is_running:
            try:
                # Get market data
                market_data = await self._get_market_data()
                iv_data = await self._get_iv_data()
                
                # Detect current regime
                new_regime = self.regime_detector.detect_market_regime(market_data, iv_data)
                
                # Check if regime has changed significantly
                if self._has_regime_changed(new_regime):
                    self.logger.info(f"Market regime changed from {self.current_regime.regime_type if self.current_regime else 'None'} "
                                   f"to {new_regime.regime_type}")
                    
                    self.current_regime = new_regime
                    self.last_regime_change = datetime.now()
                    
                    # Update active bots based on new regime
                    await self._update_active_bots()
                
                # Sleep for monitoring interval
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error monitoring market regime: {e}")
                await asyncio.sleep(60)
    
    async def _update_active_bots(self):
        """Update active bots based on current market regime"""
        if not self.current_regime:
            return
        
        # Get bot recommendations
        recommendations = self.regime_detector.recommend_bots(
            self.current_regime,
            self.total_capital,
            self.config.get('risk_tolerance', 'medium')
        )
        
        # Determine which bots to activate
        target_bots = self._select_target_bots(recommendations)
        
        # Deactivate bots no longer needed
        bots_to_stop = set(self.active_bots.keys()) - set(target_bots.keys())
        for bot_name in bots_to_stop:
            await self._deactivate_bot(bot_name)
        
        # Activate new bots
        bots_to_start = set(target_bots.keys()) - set(self.active_bots.keys())
        for bot_name in bots_to_start:
            await self._activate_bot(bot_name, target_bots[bot_name])
        
        # Update capital allocation for existing bots
        for bot_name in set(self.active_bots.keys()) & set(target_bots.keys()):
            await self._update_bot_capital(bot_name, target_bots[bot_name])
    
    def _select_target_bots(self, recommendations: List[BotRecommendation]) -> Dict[str, float]:
        """Select which bots to activate and their capital allocation"""
        target_bots = {}
        remaining_capital = self.total_capital
        
        # Portfolio composition rules
        max_bots = self.config.get('max_active_bots', 3)
        min_score = self.config.get('min_bot_score', 0.5)
        
        # Select top scoring bots
        selected_count = 0
        for rec in recommendations:
            if selected_count >= max_bots:
                break
            
            if rec.score < min_score:
                continue
            
            if rec.bot_name not in self.available_bots:
                continue
            
            # Calculate capital allocation
            # Higher scores get more capital
            weight = rec.score / sum(r.score for r in recommendations[:max_bots])
            allocated_capital = self.total_capital * weight
            
            # Ensure minimum capital requirement
            if allocated_capital >= self.min_capital_per_bot:
                target_bots[rec.bot_name] = allocated_capital
                remaining_capital -= allocated_capital
                selected_count += 1
                
                self.logger.info(f"Selected {rec.bot_name} with score {rec.score:.2f} "
                               f"and capital ${allocated_capital:,.0f}")
        
        # Redistribute any remaining capital
        if remaining_capital > 0 and target_bots:
            extra_per_bot = remaining_capital / len(target_bots)
            for bot_name in target_bots:
                target_bots[bot_name] += extra_per_bot
        
        return target_bots
    
    async def _activate_bot(self, bot_name: str, capital: float):
        """Activate a bot with allocated capital"""
        if bot_name not in self.available_bots:
            return
        
        bot = self.available_bots[bot_name]
        
        # Update bot configuration
        bot.available_capital = capital
        bot.config['available_capital'] = capital
        
        # Start the bot
        await bot.start()
        
        # Move to active bots
        self.active_bots[bot_name] = bot
        self.capital_allocation[bot_name] = capital
        
        self.logger.info(f"Activated {bot_name} with capital ${capital:,.0f}")
    
    async def _deactivate_bot(self, bot_name: str):
        """Deactivate a bot and close its positions"""
        if bot_name not in self.active_bots:
            return
        
        bot = self.active_bots[bot_name]
        
        # Stop the bot (will close positions)
        await bot.stop()
        
        # Remove from active bots
        del self.active_bots[bot_name]
        if bot_name in self.capital_allocation:
            del self.capital_allocation[bot_name]
        
        self.logger.info(f"Deactivated {bot_name}")
    
    async def _update_bot_capital(self, bot_name: str, new_capital: float):
        """Update capital allocation for an active bot"""
        if bot_name not in self.active_bots:
            return
        
        bot = self.active_bots[bot_name]
        old_capital = self.capital_allocation.get(bot_name, 0)
        
        # Only update if change is significant (>10%)
        if abs(new_capital - old_capital) / old_capital > 0.1:
            bot.available_capital = new_capital
            bot.config['available_capital'] = new_capital
            self.capital_allocation[bot_name] = new_capital
            
            self.logger.info(f"Updated {bot_name} capital from ${old_capital:,.0f} to ${new_capital:,.0f}")
    
    def _has_regime_changed(self, new_regime: MarketRegime) -> bool:
        """Check if regime has changed significantly"""
        # No previous regime
        if not self.current_regime:
            return True
        
        # Check cooldown period
        if datetime.now() - self.last_regime_change < self.regime_change_cooldown:
            return False
        
        # Regime type changed
        if new_regime.regime_type != self.current_regime.regime_type:
            return True
        
        # Significant volatility change
        if new_regime.volatility_level != self.current_regime.volatility_level:
            vol_levels = ['low', 'medium', 'high', 'extreme']
            old_idx = vol_levels.index(self.current_regime.volatility_level)
            new_idx = vol_levels.index(new_regime.volatility_level)
            if abs(new_idx - old_idx) >= 2:
                return True
        
        # Significant IV rank change
        if abs(new_regime.iv_rank - self.current_regime.iv_rank) > 30:
            return True
        
        return False
    
    async def _monitor_performance(self):
        """Monitor bot performance and make adjustments"""
        while self.is_running:
            try:
                # Collect performance metrics
                for bot_name, bot in self.active_bots.items():
                    metrics = await self._get_bot_performance(bot)
                    
                    if bot_name not in self.bot_performance:
                        self.bot_performance[bot_name] = []
                    
                    self.bot_performance[bot_name].append({
                        'timestamp': datetime.now(),
                        'metrics': metrics
                    })
                
                # Check for underperforming bots
                await self._handle_underperformance()
                
                # Log portfolio status
                await self._log_portfolio_status()
                
                # Sleep for monitoring interval
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Error monitoring performance: {e}")
                await asyncio.sleep(300)
    
    async def _get_bot_performance(self, bot: BaseBot) -> Dict[str, Any]:
        """Get performance metrics for a bot"""
        # This would query actual performance from database
        # For now, return mock metrics
        return {
            'total_pnl': bot.total_pnl if hasattr(bot, 'total_pnl') else 0,
            'win_rate': bot.win_rate if hasattr(bot, 'win_rate') else 0.5,
            'active_positions': len(bot.positions) if hasattr(bot, 'positions') else 0,
            'trades_today': bot.trades_today if hasattr(bot, 'trades_today') else 0
        }
    
    async def _handle_underperformance(self):
        """Handle underperforming bots"""
        for bot_name, perf_history in self.bot_performance.items():
            if len(perf_history) < 5:  # Need sufficient history
                continue
            
            # Check recent performance
            recent_metrics = [p['metrics'] for p in perf_history[-5:]]
            avg_pnl = sum(m['total_pnl'] for m in recent_metrics) / len(recent_metrics)
            avg_win_rate = sum(m['win_rate'] for m in recent_metrics) / len(recent_metrics)
            
            # Deactivate if significantly underperforming
            if avg_pnl < -self.capital_allocation.get(bot_name, 0) * 0.05:  # -5% loss
                self.logger.warning(f"{bot_name} underperforming with {avg_pnl:.2f} PnL")
                # Could deactivate or reduce capital here
    
    async def _rebalance_capital(self):
        """Periodically rebalance capital across bots"""
        while self.is_running:
            try:
                # Wait for rebalance interval
                await asyncio.sleep(3600)  # Rebalance every hour
                
                if not self.active_bots:
                    continue
                
                # Calculate current utilization
                total_used = sum(bot.used_capital if hasattr(bot, 'used_capital') else 0 
                               for bot in self.active_bots.values())
                
                # If utilization is low, consider adding more bots
                utilization = total_used / self.total_capital
                if utilization < 0.5:
                    self.logger.info(f"Low capital utilization: {utilization:.1%}")
                    # Could trigger bot update here
                
            except Exception as e:
                self.logger.error(f"Error rebalancing capital: {e}")
    
    async def _log_portfolio_status(self):
        """Log current portfolio status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'regime': self.current_regime.regime_type if self.current_regime else 'Unknown',
            'active_bots': list(self.active_bots.keys()),
            'capital_allocation': self.capital_allocation,
            'total_positions': sum(
                len(bot.positions) if hasattr(bot, 'positions') else 0 
                for bot in self.active_bots.values()
            )
        }
        
        self.logger.info(f"Portfolio Status: {json.dumps(status, indent=2)}")
    
    async def _get_market_data(self) -> pd.DataFrame:
        """Get market data for regime detection"""
        # This would fetch real market data
        # For now, return mock data
        return pd.DataFrame()
    
    async def _get_iv_data(self) -> Dict[str, float]:
        """Get IV data for regime detection"""
        # This would fetch real IV data
        # For now, return mock data
        return {'iv_rank': 50.0, 'iv_percentile': 50.0}
    
    def _load_performance_history(self):
        """Load historical performance data"""
        history_file = Path('data/orchestrator_performance.json')
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.performance_history = json.load(f)
    
    def _save_performance_history(self):
        """Save performance history"""
        history_file = Path('data/orchestrator_performance.json')
        history_file.parent.mkdir(exist_ok=True)
        
        with open(history_file, 'w') as f:
            json.dump(self.performance_history, f, indent=2, default=str)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status"""
        return {
            'is_running': self.is_running,
            'current_regime': {
                'type': self.current_regime.regime_type if self.current_regime else None,
                'volatility': self.current_regime.volatility_level if self.current_regime else None,
                'confidence': self.current_regime.confidence if self.current_regime else 0
            },
            'active_bots': list(self.active_bots.keys()),
            'capital_allocation': self.capital_allocation,
            'total_capital': self.total_capital,
            'performance': {
                bot_name: perf_history[-1]['metrics'] if perf_history else {}
                for bot_name, perf_history in self.bot_performance.items()
            }
        }


# Example configuration
ORCHESTRATOR_CONFIG = {
    'total_capital': 1000000,
    'min_capital_per_bot': 50000,
    'max_active_bots': 3,
    'min_bot_score': 0.5,
    'risk_tolerance': 'medium',
    'regime_change_cooldown_minutes': 30,
    'bot_configs': {
        'momentum_rider': {
            'name': 'Momentum Rider ML',
            'symbols': ['NIFTY', 'BANKNIFTY'],
            'entry_conditions': {'momentum_threshold': 0.45},
            'exit_conditions': {'stop_loss_pct': 2.0, 'take_profit_pct': 3.0},
            'position_sizing': {'base_size_pct': 1.0}
        },
        'short_straddle': {
            'name': 'Short Straddle ML',
            'symbols': ['NIFTY', 'BANKNIFTY'],
            'entry_conditions': {'iv_rank_min': 72},
            'exit_conditions': {'profit_target_pct': 25, 'stop_loss_multiplier': 1.5}
        },
        'iron_condor': {
            'name': 'Iron Condor Safe',
            'symbols': ['NIFTY', 'BANKNIFTY'],
            'entry_conditions': {'iv_percentile_min': 40, 'iv_percentile_max': 70},
            'exit_conditions': {'profit_target_pct': 50}
        },
        'volatility_expander': {
            'name': 'Vol Expander',
            'symbols': ['NIFTY', 'BANKNIFTY'],
            'entry_conditions': {'iv_percentile_max': 30},
            'exit_conditions': {'iv_expansion_target': 1.5}
        }
    }
}