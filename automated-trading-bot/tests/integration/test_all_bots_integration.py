"""
Integration Tests for All Bots Running Together
Tests multiple bots running concurrently with resource sharing
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.core.bot_manager import BotManager
from src.bots.base_bot import BotState, BaseBot
from src.config import BOT_CONSTANTS
from tests.conftest import create_mock_bot


class TestAllBotsIntegration:
    """Integration tests for multiple bots running together"""
    
    @pytest.fixture
    async def multi_bot_manager(self, test_config_manager, mock_db_manager, mock_openalgo_client):
        """Create bot manager with all bot types"""
        bot_manager = BotManager(test_config_manager)
        bot_manager.db_manager = mock_db_manager
        bot_manager.openalgo_client = mock_openalgo_client
        
        # Initialize with mock funds
        mock_openalgo_client.get_funds.return_value = {
            "available_balance": 1000000,
            "used_margin": 0
        }
        
        return bot_manager
    
    @pytest.mark.asyncio
    async def test_all_bots_initialization(self, multi_bot_manager):
        """Test initializing all bot types"""
        # Mock bot creation
        bot_types = [
            BOT_CONSTANTS.TYPE_SHORT_STRADDLE,
            BOT_CONSTANTS.TYPE_IRON_CONDOR,
            BOT_CONSTANTS.TYPE_VOLATILITY_EXPANDER,
            BOT_CONSTANTS.TYPE_MOMENTUM_RIDER
        ]
        
        # Create mock bots
        for bot_type in bot_types:
            bot = create_mock_bot(f"{bot_type}_bot", bot_type)
            multi_bot_manager.bots[bot.name] = bot
        
        # Initialize all bots
        await multi_bot_manager.initialize()
        
        # Verify all bots created
        assert len(multi_bot_manager.bots) == 4
        
        # Verify each bot type
        bot_types_found = [bot.bot_type for bot in multi_bot_manager.bots.values()]
        for bot_type in bot_types:
            assert bot_type in bot_types_found
    
    @pytest.mark.asyncio
    async def test_concurrent_bot_execution(self, multi_bot_manager):
        """Test all bots running concurrently"""
        # Create bots with different states
        bots = {
            "ShortStraddle": create_mock_bot("ShortStraddle", BOT_CONSTANTS.TYPE_SHORT_STRADDLE),
            "IronCondor": create_mock_bot("IronCondor", BOT_CONSTANTS.TYPE_IRON_CONDOR),
            "VolatilityExpander": create_mock_bot("VolatilityExpander", BOT_CONSTANTS.TYPE_VOLATILITY_EXPANDER),
            "MomentumRider": create_mock_bot("MomentumRider", BOT_CONSTANTS.TYPE_MOMENTUM_RIDER)
        }
        
        multi_bot_manager.bots = bots
        
        # Start all bots
        await multi_bot_manager._start_all_bots()
        
        # Verify all bots started
        for bot in bots.values():
            bot.start.assert_called_once()
        
        # Verify monitoring tasks created
        assert len(multi_bot_manager.bot_tasks) == 4
    
    @pytest.mark.asyncio
    async def test_resource_sharing_constraints(self, multi_bot_manager, mock_db_manager):
        """Test resource constraints across multiple bots"""
        # Set total capital
        total_capital = 1000000
        capital_per_bot = 250000
        
        # Mock capital allocation
        mock_db_manager.get_bot_capital.side_effect = lambda bot_name: {
            "initial": capital_per_bot,
            "current": capital_per_bot,
            "available": capital_per_bot * 0.8,  # 80% available
            "locked": capital_per_bot * 0.2
        }
        
        # Create bots
        bots = {}
        for i, bot_type in enumerate([
            BOT_CONSTANTS.TYPE_SHORT_STRADDLE,
            BOT_CONSTANTS.TYPE_IRON_CONDOR,
            BOT_CONSTANTS.TYPE_VOLATILITY_EXPANDER,
            BOT_CONSTANTS.TYPE_MOMENTUM_RIDER
        ]):
            bot = create_mock_bot(f"Bot{i}", bot_type)
            bots[bot.name] = bot
        
        multi_bot_manager.bots = bots
        
        # Get system status
        status = multi_bot_manager.get_system_status()
        
        # Verify total capital equals sum of bot capitals
        assert status["config"]["total_capital"] == total_capital
    
    @pytest.mark.asyncio
    async def test_market_data_distribution(self, multi_bot_manager):
        """Test market data distribution to all bots"""
        # Create bots that subscribe to different symbols
        bot_subscriptions = {
            "Bot1": ["NIFTY", "BANKNIFTY"],
            "Bot2": ["NIFTY"],
            "Bot3": ["BANKNIFTY", "FINNIFTY"],
            "Bot4": ["NIFTY", "BANKNIFTY", "FINNIFTY"]
        }
        
        bots = {}
        for bot_name, symbols in bot_subscriptions.items():
            bot = create_mock_bot(bot_name, BOT_CONSTANTS.TYPE_SHORT_STRADDLE)
            bot.symbols = set(symbols)
            bot.on_market_data = AsyncMock()
            bots[bot_name] = bot
        
        multi_bot_manager.bots = bots
        
        # Simulate market data updates
        market_updates = [
            ("NIFTY", {"ltp": 20000, "volume": 1000000}),
            ("BANKNIFTY", {"ltp": 45000, "volume": 500000}),
            ("FINNIFTY", {"ltp": 21000, "volume": 300000})
        ]
        
        # Distribute market data
        for symbol, data in market_updates:
            # Each bot that subscribes to the symbol should receive data
            for bot_name, bot in bots.items():
                if symbol in bot.symbols:
                    await bot.on_market_data(symbol, data)
        
        # Verify correct distribution
        assert bots["Bot1"].on_market_data.call_count == 2  # NIFTY, BANKNIFTY
        assert bots["Bot2"].on_market_data.call_count == 1  # NIFTY only
        assert bots["Bot3"].on_market_data.call_count == 2  # BANKNIFTY, FINNIFTY
        assert bots["Bot4"].on_market_data.call_count == 3  # All symbols
    
    @pytest.mark.asyncio
    async def test_position_limit_coordination(self, multi_bot_manager, mock_db_manager):
        """Test position limits across all bots"""
        # Set global position limit
        max_total_positions = 10
        
        # Current positions per bot
        bot_positions = {
            "Bot1": 3,
            "Bot2": 2,
            "Bot3": 4,
            "Bot4": 1
        }
        
        # Mock position retrieval
        all_positions = []
        for bot_name, count in bot_positions.items():
            for i in range(count):
                all_positions.append({
                    "bot_name": bot_name,
                    "symbol": f"SYMBOL{i}",
                    "status": "OPEN"
                })
        
        mock_db_manager.get_open_positions.return_value = all_positions
        
        # Check if new position allowed
        total_positions = sum(bot_positions.values())
        assert total_positions == max_total_positions
        
        # New position should be blocked at system level
        can_add_position = total_positions < max_total_positions
        assert can_add_position is False
    
    @pytest.mark.asyncio
    async def test_risk_aggregation_across_bots(self, multi_bot_manager):
        """Test risk metrics aggregation across all bots"""
        # Create bots with different risk profiles
        risk_metrics = {
            "ShortStraddle": {"delta": 50, "gamma": 10, "vega": 500, "theta": -100},
            "IronCondor": {"delta": -20, "gamma": 5, "vega": 300, "theta": -50},
            "VolatilityExpander": {"delta": 0, "gamma": 20, "vega": -200, "theta": -150},
            "MomentumRider": {"delta": 100, "gamma": 0, "vega": 0, "theta": 0}
        }
        
        bots = {}
        for bot_name, metrics in risk_metrics.items():
            bot = create_mock_bot(bot_name, BOT_CONSTANTS.TYPE_SHORT_STRADDLE)
            bot.get_risk_metrics = Mock(return_value=metrics)
            bots[bot_name] = bot
        
        multi_bot_manager.bots = bots
        
        # Calculate aggregate risk
        total_risk = {
            "delta": sum(bot.get_risk_metrics()["delta"] for bot in bots.values()),
            "gamma": sum(bot.get_risk_metrics()["gamma"] for bot in bots.values()),
            "vega": sum(bot.get_risk_metrics()["vega"] for bot in bots.values()),
            "theta": sum(bot.get_risk_metrics()["theta"] for bot in bots.values())
        }
        
        # Verify aggregate risk metrics
        assert total_risk["delta"] == 130  # 50 - 20 + 0 + 100
        assert total_risk["gamma"] == 35   # 10 + 5 + 20 + 0
        assert total_risk["vega"] == 600   # 500 + 300 - 200 + 0
        assert total_risk["theta"] == -300  # -100 - 50 - 150 + 0
    
    @pytest.mark.asyncio
    async def test_performance_comparison(self, multi_bot_manager):
        """Test performance comparison between bots"""
        # Create bots with different performance
        performance_data = {
            "ShortStraddle": {"trades": 50, "win_rate": 70, "pnl": 50000, "sharpe": 1.5},
            "IronCondor": {"trades": 40, "win_rate": 65, "pnl": 30000, "sharpe": 1.2},
            "VolatilityExpander": {"trades": 20, "win_rate": 45, "pnl": -5000, "sharpe": -0.3},
            "MomentumRider": {"trades": 100, "win_rate": 55, "pnl": 80000, "sharpe": 2.1}
        }
        
        bots = {}
        for bot_name, perf in performance_data.items():
            bot = create_mock_bot(bot_name, BOT_CONSTANTS.TYPE_SHORT_STRADDLE)
            bot.get_status = Mock(return_value={
                "name": bot_name,
                "performance": {
                    "total_trades": perf["trades"],
                    "win_rate": perf["win_rate"],
                    "total_pnl": perf["pnl"],
                    "sharpe_ratio": perf["sharpe"]
                }
            })
            bots[bot_name] = bot
        
        multi_bot_manager.bots = bots
        
        # Get comparative analysis
        all_performances = []
        for bot in bots.values():
            status = bot.get_status()
            all_performances.append({
                "bot": status["name"],
                "performance": status["performance"]
            })
        
        # Sort by PnL
        all_performances.sort(key=lambda x: x["performance"]["total_pnl"], reverse=True)
        
        # Verify ranking
        assert all_performances[0]["bot"] == "MomentumRider"  # Highest PnL
        assert all_performances[-1]["bot"] == "VolatilityExpander"  # Lowest PnL
    
    @pytest.mark.asyncio
    async def test_correlation_monitoring(self, multi_bot_manager):
        """Test correlation monitoring between bot positions"""
        # Mock positions with correlation
        positions = {
            "Bot1": [{"symbol": "NIFTY", "type": "SHORT"}],
            "Bot2": [{"symbol": "NIFTY", "type": "SHORT"}],  # High correlation with Bot1
            "Bot3": [{"symbol": "BANKNIFTY", "type": "LONG"}],
            "Bot4": [{"symbol": "GOLD", "type": "LONG"}]  # Low correlation
        }
        
        # Calculate correlation matrix
        correlation_pairs = []
        bot_names = list(positions.keys())
        
        for i in range(len(bot_names)):
            for j in range(i + 1, len(bot_names)):
                bot1, bot2 = bot_names[i], bot_names[j]
                
                # Simple correlation check based on symbol and direction
                pos1 = positions[bot1][0]
                pos2 = positions[bot2][0]
                
                if pos1["symbol"] == pos2["symbol"] and pos1["type"] == pos2["type"]:
                    correlation = 0.9  # High correlation
                elif pos1["symbol"] == pos2["symbol"]:
                    correlation = -0.5  # Negative correlation
                else:
                    correlation = 0.1  # Low correlation
                
                correlation_pairs.append({
                    "bot1": bot1,
                    "bot2": bot2,
                    "correlation": correlation
                })
        
        # Find high correlations
        high_correlations = [cp for cp in correlation_pairs if abs(cp["correlation"]) > 0.7]
        
        # Should identify Bot1 and Bot2 as highly correlated
        assert len(high_correlations) == 1
        assert high_correlations[0]["correlation"] == 0.9
    
    @pytest.mark.asyncio
    async def test_emergency_stop_all_bots(self, multi_bot_manager):
        """Test emergency stop functionality for all bots"""
        # Create running bots
        bots = {}
        for i in range(4):
            bot = create_mock_bot(f"Bot{i}", BOT_CONSTANTS.TYPE_SHORT_STRADDLE, BotState.RUNNING)
            bot.close_all_positions = AsyncMock()
            bots[bot.name] = bot
        
        multi_bot_manager.bots = bots
        multi_bot_manager.is_running = True
        
        # Trigger emergency stop
        await multi_bot_manager.stop()
        
        # Verify all bots stopped
        for bot in bots.values():
            bot.stop.assert_called_once()
        
        # Verify system state
        assert multi_bot_manager.is_running is False
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, multi_bot_manager):
        """Test overall system health monitoring"""
        # Create bots with different health states
        health_states = {
            "Bot1": {"state": BotState.RUNNING, "errors": 0, "last_trade": datetime.now()},
            "Bot2": {"state": BotState.ERROR, "errors": 5, "last_trade": None},
            "Bot3": {"state": BotState.PAUSED, "errors": 1, "last_trade": datetime.now()},
            "Bot4": {"state": BotState.RUNNING, "errors": 0, "last_trade": datetime.now()}
        }
        
        bots = {}
        for bot_name, health in health_states.items():
            bot = create_mock_bot(bot_name, BOT_CONSTANTS.TYPE_SHORT_STRADDLE, health["state"])
            bot.error_count = health["errors"]
            bot.last_trade_time = health["last_trade"]
            bots[bot_name] = bot
        
        multi_bot_manager.bots = bots
        
        # Check system health
        system_health = {
            "healthy_bots": sum(1 for b in bots.values() if b.state == BotState.RUNNING),
            "error_bots": sum(1 for b in bots.values() if b.state == BotState.ERROR),
            "total_errors": sum(b.error_count for b in bots.values())
        }
        
        # Verify health metrics
        assert system_health["healthy_bots"] == 2
        assert system_health["error_bots"] == 1
        assert system_health["total_errors"] == 6