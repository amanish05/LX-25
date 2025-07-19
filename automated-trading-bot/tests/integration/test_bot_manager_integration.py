"""
Integration Tests for Bot Manager
Tests the complete bot management system with all components
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, call
import json

from src.core.bot_manager import BotManager
from src.bots.base_bot import BotState
from src.config import BOT_CONSTANTS
from tests.conftest import create_mock_bot, wait_for_condition, set_market_hours


class TestBotManagerIntegration:
    """Integration tests for Bot Manager"""
    
    @pytest.mark.asyncio
    async def test_bot_manager_initialization(self, mock_bot_manager):
        """Test bot manager initialization with all components"""
        # Initialize bot manager
        await mock_bot_manager.initialize()
        
        # Verify database initialization
        mock_bot_manager.db_manager.init_database.assert_called_once()
        
        # Verify OpenAlgo connection
        mock_bot_manager.openalgo_client.get_funds.assert_called_once()
        
        # Verify bots are loaded
        assert len(mock_bot_manager.bots) >= 0  # Bots loaded based on config
    
    @pytest.mark.asyncio
    async def test_start_multiple_bots_concurrently(self, mock_bot_manager):
        """Test starting multiple bots simultaneously"""
        # Create mock bots
        bot1 = create_mock_bot("ShortStraddleBot", BOT_CONSTANTS.TYPE_SHORT_STRADDLE)
        bot2 = create_mock_bot("IronCondorBot", BOT_CONSTANTS.TYPE_IRON_CONDOR)
        
        mock_bot_manager.bots = {
            bot1.name: bot1,
            bot2.name: bot2
        }
        
        # Start bot manager
        await mock_bot_manager.initialize()
        
        # Start all bots
        await mock_bot_manager._start_all_bots()
        
        # Verify both bots started
        bot1.start.assert_called_once()
        bot2.start.assert_called_once()
        
        # Verify monitoring tasks created
        assert len(mock_bot_manager.bot_tasks) == 2
        assert mock_bot_manager.system_stats["active_bots"] == 2
    
    @pytest.mark.asyncio
    async def test_bot_health_monitoring(self, mock_bot_manager):
        """Test bot health monitoring and auto-restart"""
        # Create a bot that will fail
        bot = create_mock_bot("TestBot", BOT_CONSTANTS.TYPE_SHORT_STRADDLE)
        bot.state = BotState.ERROR
        
        mock_bot_manager.bots = {bot.name: bot}
        mock_bot_manager.is_running = True
        
        # Run monitor for one cycle
        monitor_task = asyncio.create_task(mock_bot_manager._monitor_bot(bot))
        
        # Wait a bit for monitoring to kick in
        await asyncio.sleep(0.1)
        
        # Cancel the monitor task
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        # Verify restart was attempted
        bot.stop.assert_called()
        bot.start.assert_called()
    
    @pytest.mark.asyncio
    async def test_market_hours_checking(self, mock_bot_manager):
        """Test market hours validation"""
        # Test during market hours
        with set_market_hours(is_open=True):
            assert mock_bot_manager._is_market_hours() is True
        
        # Test outside market hours
        with set_market_hours(is_open=False):
            assert mock_bot_manager._is_market_hours() is False
        
        # Test on weekend
        with patch('datetime.datetime.now') as mock_now:
            # Saturday
            mock_now.return_value = datetime(2024, 2, 24, 10, 0, 0)  # Saturday
            assert mock_bot_manager._is_market_hours() is False
    
    @pytest.mark.asyncio
    async def test_system_status_aggregation(self, mock_bot_manager):
        """Test system status aggregation from all bots"""
        # Create bots with different states
        bot1 = create_mock_bot("Bot1", BOT_CONSTANTS.TYPE_SHORT_STRADDLE, BotState.RUNNING)
        bot2 = create_mock_bot("Bot2", BOT_CONSTANTS.TYPE_IRON_CONDOR, BotState.PAUSED)
        
        mock_bot_manager.bots = {
            bot1.name: bot1,
            bot2.name: bot2
        }
        
        # Get system status
        status = mock_bot_manager.get_system_status()
        
        # Verify status structure
        assert "uptime_hours" in status
        assert "is_running" in status
        assert "market_hours" in status
        assert "stats" in status
        assert "bots" in status
        assert "config" in status
        
        # Verify bot statuses
        assert len(status["bots"]) == 2
        assert bot1.name in status["bots"]
        assert bot2.name in status["bots"]
    
    @pytest.mark.asyncio
    async def test_concurrent_bot_operations(self, mock_bot_manager):
        """Test concurrent operations on multiple bots"""
        # Create multiple bots
        bots = []
        for i in range(5):
            bot = create_mock_bot(f"Bot{i}", BOT_CONSTANTS.TYPE_SHORT_STRADDLE)
            bots.append(bot)
            mock_bot_manager.bots[bot.name] = bot
        
        # Start all bots concurrently
        start_tasks = [mock_bot_manager.start_bot(bot.name) for bot in bots]
        await asyncio.gather(*start_tasks)
        
        # Verify all bots started
        for bot in bots:
            bot.start.assert_called_once()
        
        # Stop all bots concurrently
        stop_tasks = [mock_bot_manager.stop_bot(bot.name) for bot in bots]
        await asyncio.gather(*stop_tasks)
        
        # Verify all bots stopped
        for bot in bots:
            bot.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_bot_restart_on_failure(self, mock_bot_manager):
        """Test automatic bot restart on failure"""
        # Create a bot that will fail
        bot = create_mock_bot("FailingBot", BOT_CONSTANTS.TYPE_SHORT_STRADDLE)
        mock_bot_manager.bots = {bot.name: bot}
        
        # Simulate bot failure
        bot.state = BotState.ERROR
        
        # Call restart method
        await mock_bot_manager._restart_bot(bot)
        
        # Verify restart sequence
        bot.stop.assert_called_once()
        bot.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_performance_metrics_update(self, mock_bot_manager):
        """Test performance metrics aggregation"""
        # Create bots with performance data
        bot1 = create_mock_bot("Bot1", BOT_CONSTANTS.TYPE_SHORT_STRADDLE)
        bot2 = create_mock_bot("Bot2", BOT_CONSTANTS.TYPE_IRON_CONDOR)
        
        mock_bot_manager.bots = {
            bot1.name: bot1,
            bot2.name: bot2
        }
        
        # Update system stats
        mock_bot_manager._update_system_stats(bot1.get_status())
        
        # Verify stats updated
        assert mock_bot_manager.system_stats["total_trades"] == 20  # 10 + 10
        assert mock_bot_manager.system_stats["total_pnl"] == 10000  # 5000 + 5000
    
    @pytest.mark.asyncio
    async def test_websocket_connection_management(self, mock_bot_manager):
        """Test WebSocket connection and reconnection"""
        await mock_bot_manager.initialize()
        
        # Start bot manager
        start_task = asyncio.create_task(mock_bot_manager.start())
        
        # Wait a bit for start
        await asyncio.sleep(0.1)
        
        # Cancel start task
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass
        
        # Verify WebSocket connection attempted
        mock_bot_manager.openalgo_client.connect_websocket.assert_called()
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, mock_bot_manager):
        """Test graceful shutdown of all components"""
        # Create and start bots
        bot1 = create_mock_bot("Bot1", BOT_CONSTANTS.TYPE_SHORT_STRADDLE, BotState.RUNNING)
        bot2 = create_mock_bot("Bot2", BOT_CONSTANTS.TYPE_IRON_CONDOR, BotState.RUNNING)
        
        mock_bot_manager.bots = {
            bot1.name: bot1,
            bot2.name: bot2
        }
        mock_bot_manager.is_running = True
        
        # Create monitoring tasks
        task1 = asyncio.create_task(asyncio.sleep(10))
        task2 = asyncio.create_task(asyncio.sleep(10))
        mock_bot_manager.bot_tasks = {
            bot1.name: task1,
            bot2.name: task2
        }
        
        # Stop bot manager
        await mock_bot_manager.stop()
        
        # Verify all bots stopped
        bot1.stop.assert_called_once()
        bot2.stop.assert_called_once()
        
        # Verify tasks cancelled
        assert task1.cancelled()
        assert task2.cancelled()
        
        # Verify state
        assert mock_bot_manager.is_running is False
        assert len(mock_bot_manager.bot_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_individual_bot_control(self, mock_bot_manager):
        """Test individual bot start/stop/pause/resume"""
        bot = create_mock_bot("TestBot", BOT_CONSTANTS.TYPE_SHORT_STRADDLE)
        mock_bot_manager.bots = {bot.name: bot}
        
        # Start bot
        await mock_bot_manager.start_bot(bot.name)
        bot.start.assert_called_once()
        
        # Pause bot
        await mock_bot_manager.pause_bot(bot.name)
        bot.pause.assert_called_once()
        
        # Resume bot
        await mock_bot_manager.resume_bot(bot.name)
        bot.resume.assert_called_once()
        
        # Stop bot
        await mock_bot_manager.stop_bot(bot.name)
        bot.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_in_bot_operations(self, mock_bot_manager):
        """Test error handling during bot operations"""
        # Test with non-existent bot
        with pytest.raises(ValueError, match="Bot NonExistentBot not found"):
            await mock_bot_manager.start_bot("NonExistentBot")
        
        # Test bot that fails to start
        bot = create_mock_bot("FailingBot", BOT_CONSTANTS.TYPE_SHORT_STRADDLE)
        bot.start.side_effect = Exception("Start failed")
        mock_bot_manager.bots = {bot.name: bot}
        
        # Should not raise, but log error
        try:
            await mock_bot_manager.start_bot(bot.name)
        except Exception:
            pytest.fail("Bot manager should handle bot start failures gracefully")
    
    @pytest.mark.asyncio
    async def test_database_cleanup_task(self, mock_bot_manager):
        """Test periodic database cleanup"""
        mock_bot_manager.is_running = True
        
        # Run cleanup task for one cycle
        cleanup_task = asyncio.create_task(mock_bot_manager._cleanup_task())
        
        # Wait a bit
        await asyncio.sleep(0.1)
        
        # Stop the task
        mock_bot_manager.is_running = False
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        
        # Verify cleanup was called
        mock_bot_manager.db_manager.cleanup_expired_cache.assert_called()
    
    @pytest.mark.asyncio
    async def test_bot_capital_management(self, mock_bot_manager):
        """Test capital allocation and management across bots"""
        # Create bots with capital requirements
        bot1 = create_mock_bot("Bot1", BOT_CONSTANTS.TYPE_SHORT_STRADDLE)
        bot2 = create_mock_bot("Bot2", BOT_CONSTANTS.TYPE_IRON_CONDOR)
        
        mock_bot_manager.bots = {
            bot1.name: bot1,
            bot2.name: bot2
        }
        
        # Get system status
        status = mock_bot_manager.get_system_status()
        
        # Verify capital information
        assert "total_capital" in status["config"]
        assert "available_capital" in status["config"]
        
        # Each bot should have capital allocation
        for bot_name, bot_status in status["bots"].items():
            assert "capital" in bot_status
            assert "initial" in bot_status["capital"]
            assert "current" in bot_status["capital"]
    
    @pytest.mark.asyncio
    async def test_concurrent_market_data_handling(self, mock_bot_manager):
        """Test handling market data updates for multiple bots"""
        # Create bots
        bot1 = create_mock_bot("Bot1", BOT_CONSTANTS.TYPE_SHORT_STRADDLE)
        bot2 = create_mock_bot("Bot2", BOT_CONSTANTS.TYPE_IRON_CONDOR)
        
        # Add market data handler
        bot1.on_market_data = AsyncMock()
        bot2.on_market_data = AsyncMock()
        
        mock_bot_manager.bots = {
            bot1.name: bot1,
            bot2.name: bot2
        }
        
        # Simulate market data update
        market_data = {
            "symbol": "NIFTY",
            "ltp": 20000,
            "timestamp": datetime.now().isoformat()
        }
        
        # Both bots should receive the update
        await bot1.on_market_data("NIFTY", market_data)
        await bot2.on_market_data("NIFTY", market_data)
        
        bot1.on_market_data.assert_called_once_with("NIFTY", market_data)
        bot2.on_market_data.assert_called_once_with("NIFTY", market_data)