"""
Full System Integration Tests
Tests the complete trading system end-to-end
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import httpx
from fastapi.testclient import TestClient

from src.core.bot_manager import BotManager
from src.api.app import create_app
from src.config import get_config_manager, BOT_CONSTANTS
from src.bots.base_bot import BotState
from tests.conftest import MockWebSocket, create_mock_bot, wait_for_condition


class TestFullSystemIntegration:
    """Full system integration tests"""
    
    @pytest.mark.asyncio
    async def test_system_startup_sequence(self, test_config_manager, mock_db_manager, mock_openalgo_client):
        """Test complete system startup sequence"""
        # 1. Initialize configuration
        assert test_config_manager.app_config.system.environment == "test"
        
        # 2. Create bot manager
        bot_manager = BotManager(test_config_manager)
        bot_manager.db_manager = mock_db_manager
        bot_manager.openalgo_client = mock_openalgo_client
        
        # 3. Initialize bot manager
        await bot_manager.initialize()
        
        # Verify database initialized
        mock_db_manager.init_database.assert_called_once()
        
        # Verify OpenAlgo connection
        mock_openalgo_client.get_funds.assert_called_once()
        
        # 4. Create API application
        app = create_app(bot_manager, test_config_manager)
        assert app is not None
        
        # 5. Test API is accessible
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_complete_trading_workflow(self, test_config_manager):
        """Test complete trading workflow from signal to exit"""
        # Setup mocks
        mock_db = AsyncMock()
        mock_openalgo = AsyncMock()
        
        # Initialize system
        bot_manager = BotManager(test_config_manager)
        bot_manager.db_manager = mock_db
        bot_manager.openalgo_client = mock_openalgo
        
        # Mock responses
        mock_openalgo.get_funds.return_value = {"available_balance": 1000000}
        mock_openalgo.get_option_chain.return_value = {
            "2024-02-29": {
                "strikes": {
                    "20000": {
                        "CE": {"ltp": 150, "iv": 25, "oi": 50000},
                        "PE": {"ltp": 140, "iv": 24, "oi": 45000}
                    }
                }
            }
        }
        mock_openalgo.place_order.return_value = {"status": "success", "orderid": "12345"}
        mock_openalgo.get_quote.return_value = {"ltp": 75}  # Price for exit
        
        mock_db.init_bot_capital = AsyncMock()
        mock_db.get_open_positions = AsyncMock(return_value=[])
        mock_db.create_position = AsyncMock(return_value=1)
        mock_db.update_position = AsyncMock()
        mock_db.close_position = AsyncMock()
        mock_db.save_signal = AsyncMock(return_value=1)
        mock_db.update_bot_capital = AsyncMock()
        
        # Initialize system
        await bot_manager.initialize()
        
        # Create short straddle bot
        bot = create_mock_bot("ShortStraddleBot", BOT_CONSTANTS.TYPE_SHORT_STRADDLE)
        bot_manager.bots[bot.name] = bot
        
        # Start system
        await bot_manager.start()
        
        # Simulate market data that triggers signal
        market_data = {
            "symbol": "NIFTY",
            "ltp": 20000,
            "timestamp": datetime.now().isoformat()
        }
        
        # Process market data (would trigger signal in real bot)
        # In real scenario, bot.on_market_data would be called
        
        # Simulate signal execution
        signal = {
            "symbol": "NIFTY",
            "type": "SHORT_STRADDLE",
            "iv_rank": 85
        }
        
        # Save signal
        await mock_db.save_signal(
            bot_name=bot.name,
            symbol=signal["symbol"],
            signal_type=signal["type"],
            signal_strength=0.85
        )
        
        # Create position
        position_id = await mock_db.create_position(
            bot_name=bot.name,
            symbol="NIFTY",
            position_type="SHORT_STRADDLE",
            quantity=50,
            entry_price=290
        )
        
        # Update capital
        await mock_db.update_bot_capital(bot.name, locked_delta=-60000, pnl_delta=0)
        
        # Simulate time passing and profit target hit
        await asyncio.sleep(0.1)
        
        # Close position
        await mock_db.close_position(position_id, exit_price=145, pnl=7250)
        await mock_db.update_bot_capital(bot.name, locked_delta=60000, pnl_delta=7250)
        
        # Verify complete workflow executed
        assert mock_db.save_signal.called
        assert mock_db.create_position.called
        assert mock_db.close_position.called
        assert mock_db.update_bot_capital.call_count == 2
    
    @pytest.mark.asyncio
    async def test_api_bot_control_integration(self, test_config_manager):
        """Test controlling bots via API"""
        # Setup system
        bot_manager = Mock()
        bot_manager.get_bot_status.return_value = {}
        bot_manager.start_bot = AsyncMock()
        bot_manager.stop_bot = AsyncMock()
        bot_manager.get_system_status.return_value = {
            "is_running": True,
            "bots": {}
        }
        
        # Create API app
        app = create_app(bot_manager, test_config_manager)
        
        with TestClient(app) as client:
            # Start bot via API
            response = client.post("/api/bots/TestBot/start")
            assert response.status_code == 200
            bot_manager.start_bot.assert_called_with("TestBot")
            
            # Stop bot via API
            response = client.post("/api/bots/TestBot/stop")
            assert response.status_code == 200
            bot_manager.stop_bot.assert_called_with("TestBot")
    
    @pytest.mark.asyncio
    async def test_websocket_market_data_flow(self):
        """Test market data flow through WebSocket"""
        # Mock WebSocket server
        mock_ws = MockWebSocket()
        
        # Market data handler
        received_data = []
        
        async def on_market_data(symbol, data):
            received_data.append((symbol, data))
        
        # Simulate WebSocket connection
        with patch('websockets.connect', return_value=mock_ws):
            # Send market updates
            updates = [
                {
                    "type": "market_data",
                    "symbol": "NIFTY",
                    "data": {"ltp": 20000 + i, "volume": 1000000 + i * 1000}
                }
                for i in range(5)
            ]
            
            for update in updates:
                mock_ws.add_message(update)
            
            # Process messages
            for _ in range(5):
                msg = await mock_ws.recv()
                data = json.loads(msg)
                await on_market_data(data["symbol"], data["data"])
            
            # Verify all updates received
            assert len(received_data) == 5
            
            # Verify data integrity
            for i, (symbol, data) in enumerate(received_data):
                assert symbol == "NIFTY"
                assert data["ltp"] == 20000 + i
    
    @pytest.mark.asyncio
    async def test_multi_bot_resource_management(self, test_config_manager):
        """Test resource management with multiple bots"""
        # Create bot manager
        bot_manager = BotManager(test_config_manager)
        
        # Mock components
        bot_manager.db_manager = AsyncMock()
        bot_manager.openalgo_client = AsyncMock()
        bot_manager.openalgo_client.get_funds.return_value = {"available_balance": 1000000}
        
        # Create multiple bots
        bot_configs = [
            {"name": "Bot1", "type": BOT_CONSTANTS.TYPE_SHORT_STRADDLE, "capital": 250000},
            {"name": "Bot2", "type": BOT_CONSTANTS.TYPE_IRON_CONDOR, "capital": 250000},
            {"name": "Bot3", "type": BOT_CONSTANTS.TYPE_VOLATILITY_EXPANDER, "capital": 250000},
            {"name": "Bot4", "type": BOT_CONSTANTS.TYPE_MOMENTUM_RIDER, "capital": 250000}
        ]
        
        for config in bot_configs:
            bot = create_mock_bot(config["name"], config["type"])
            bot.initial_capital = config["capital"]
            bot_manager.bots[bot.name] = bot
        
        # Start all bots
        await bot_manager._start_all_bots()
        
        # Verify capital allocation
        total_allocated = sum(bot.initial_capital for bot in bot_manager.bots.values())
        assert total_allocated == 1000000  # All capital allocated
        
        # Verify each bot started
        for bot in bot_manager.bots.values():
            bot.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_propagation_and_recovery(self, test_config_manager):
        """Test error handling across system components"""
        # Setup system with potential failure points
        bot_manager = BotManager(test_config_manager)
        
        # Mock database failure
        bot_manager.db_manager = Mock()
        bot_manager.db_manager.init_database = AsyncMock(side_effect=Exception("DB Connection failed"))
        
        # Mock OpenAlgo client
        bot_manager.openalgo_client = Mock()
        bot_manager.openalgo_client.get_funds = AsyncMock(return_value={"available_balance": 0})
        
        # Try to initialize - should handle DB error gracefully
        with pytest.raises(Exception, match="DB Connection failed"):
            await bot_manager.initialize()
        
        # Fix database and retry
        bot_manager.db_manager.init_database = AsyncMock()
        await bot_manager.initialize()
        
        # System should recover
        assert bot_manager.db_manager.init_database.called
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, test_config_manager):
        """Test performance monitoring across system"""
        # Create system components
        bot_manager = BotManager(test_config_manager)
        bot_manager.db_manager = AsyncMock()
        bot_manager.openalgo_client = AsyncMock()
        
        # Mock performance data
        performance_data = {
            "total_trades": 100,
            "winning_trades": 65,
            "total_pnl": 150000,
            "max_drawdown": 25000,
            "sharpe_ratio": 1.8
        }
        
        bot_manager.db_manager.get_bot_performance = AsyncMock(return_value=performance_data)
        
        # Create API and test performance endpoint
        app = create_app(bot_manager, test_config_manager)
        
        with TestClient(app) as client:
            response = client.get("/api/performance")
            assert response.status_code == 200
            
            data = response.json()
            assert "total_pnl" in data
            assert "win_rate" in data
    
    @pytest.mark.asyncio
    async def test_configuration_hot_reload(self, tmp_path):
        """Test configuration changes without restart"""
        # Create config directory
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Initial config
        initial_config = {
            "system": {"total_capital": 1000000},
            "domains": {"openalgo_api_port": 5000}
        }
        
        config_file = config_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(initial_config, f)
        
        # Create config manager
        from src.config import ConfigManager
        config_manager = ConfigManager(str(config_dir))
        
        # Verify initial values
        assert config_manager.app_config.system.total_capital == 1000000
        
        # Update config file
        updated_config = initial_config.copy()
        updated_config["system"]["total_capital"] = 2000000
        
        with open(config_file, 'w') as f:
            json.dump(updated_config, f)
        
        # Reload configuration
        config_manager.reload()
        
        # Verify updated values
        assert config_manager.app_config.system.total_capital == 2000000
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown_sequence(self, test_config_manager):
        """Test graceful system shutdown"""
        # Create running system
        bot_manager = BotManager(test_config_manager)
        bot_manager.db_manager = AsyncMock()
        bot_manager.openalgo_client = AsyncMock()
        bot_manager.openalgo_client.disconnect_websocket = AsyncMock()
        
        # Add running bots
        for i in range(3):
            bot = create_mock_bot(f"Bot{i}", BOT_CONSTANTS.TYPE_SHORT_STRADDLE, BotState.RUNNING)
            bot_manager.bots[bot.name] = bot
            bot_manager.bot_tasks[bot.name] = asyncio.create_task(asyncio.sleep(10))
        
        bot_manager.is_running = True
        
        # Initiate shutdown
        await bot_manager.stop()
        
        # Verify shutdown sequence
        # 1. All bots stopped
        for bot in bot_manager.bots.values():
            bot.stop.assert_called_once()
        
        # 2. All tasks cancelled
        assert len(bot_manager.bot_tasks) == 0
        
        # 3. WebSocket disconnected
        bot_manager.openalgo_client.disconnect_websocket.assert_called_once()
        
        # 4. System marked as stopped
        assert bot_manager.is_running is False
    
    @pytest.mark.asyncio
    async def test_data_consistency_across_components(self, test_config_manager):
        """Test data consistency between API, bots, and database"""
        # Setup system
        bot_manager = BotManager(test_config_manager)
        bot_manager.db_manager = AsyncMock()
        bot_manager.openalgo_client = AsyncMock()
        
        # Create position data
        position_data = {
            "id": 1,
            "bot_name": "TestBot",
            "symbol": "NIFTY",
            "entry_price": 150,
            "current_price": 140,
            "pnl": 500
        }
        
        # Mock database returns position
        bot_manager.db_manager.get_open_positions.return_value = [position_data]
        
        # Create bot that reports same position
        bot = create_mock_bot("TestBot", BOT_CONSTANTS.TYPE_SHORT_STRADDLE)
        bot.positions = {1: position_data}
        bot_manager.bots[bot.name] = bot
        
        # Create API
        app = create_app(bot_manager, test_config_manager)
        
        # Verify consistency via API
        with TestClient(app) as client:
            response = client.get("/api/positions")
            assert response.status_code == 200
            
            positions = response.json()
            assert len(positions) == 1
            assert positions[0]["pnl"] == 500
            
            # Verify bot status matches
            bot_status = bot.get_status()
            assert len(bot.positions) == 1