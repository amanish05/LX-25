"""
Integration Tests for Short Straddle Bot
Tests the complete short straddle trading workflow
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from src.bots.short_straddle_bot import ShortStraddleBot
from src.bots.base_bot import BotState
from tests.conftest import sample_market_data, sample_option_chain


class TestShortStraddleIntegration:
    """Integration tests for Short Straddle Bot"""
    
    @pytest.fixture
    async def short_straddle_bot(self, sample_bot_config, mock_db_manager, mock_openalgo_client):
        """Create short straddle bot instance"""
        bot = ShortStraddleBot(
            config=sample_bot_config,
            db_manager=mock_db_manager,
            openalgo_client=mock_openalgo_client
        )
        return bot
    
    @pytest.mark.asyncio
    async def test_end_to_end_trade_flow(self, short_straddle_bot, mock_openalgo_client, mock_db_manager):
        """Test complete trade flow from signal to exit"""
        # Initialize bot
        await short_straddle_bot.start()
        assert short_straddle_bot.state == BotState.RUNNING
        
        # Setup high IV scenario
        option_chain = sample_option_chain()["NIFTY"]
        mock_openalgo_client.get_option_chain.return_value = option_chain
        
        # Mock IV rank calculation to trigger signal
        with patch.object(short_straddle_bot, '_calculate_iv_rank', return_value=85):
            # Generate signal
            market_data = sample_market_data()["NIFTY"]
            signal = await short_straddle_bot.generate_signals("NIFTY", market_data)
            
            assert signal is not None
            assert signal["type"] == "SHORT_STRADDLE"
            assert signal["iv_rank"] == 85
        
        # Execute trade
        position_size = await short_straddle_bot.calculate_position_size(signal)
        assert position_size == 50  # 1 lot NIFTY
        
        # Mock order placement success
        mock_openalgo_client.place_order.return_value = {
            "status": "success",
            "orderid": "12345"
        }
        
        # Place orders
        await short_straddle_bot.execute_signal(signal, position_size)
        
        # Verify both legs placed
        assert mock_openalgo_client.place_order.call_count == 2
        
        # Verify position created in database
        mock_db_manager.create_position.assert_called()
        
        # Simulate profit scenario
        mock_openalgo_client.get_quote.side_effect = [
            {"ltp": 75},  # Call price dropped 50%
            {"ltp": 75}   # Put price dropped 50%
        ]
        
        # Check exit conditions
        position = {"id": 1, "symbol": "NIFTY"}
        should_exit = await short_straddle_bot.should_exit_position(position, market_data)
        assert should_exit is True
        
        # Execute exit
        await short_straddle_bot._close_position(position)
        
        # Verify exit orders placed
        assert mock_openalgo_client.place_order.call_count == 4  # 2 entry + 2 exit
        
        # Verify position closed in database
        mock_db_manager.close_position.assert_called()
    
    @pytest.mark.asyncio
    async def test_risk_management_integration(self, short_straddle_bot, mock_db_manager):
        """Test risk management across multiple positions"""
        # Set capital and risk limits
        short_straddle_bot.available_capital = 500000
        short_straddle_bot.max_positions = 2
        
        # Create first position
        position1_data = {
            "bot_name": short_straddle_bot.name,
            "symbol": "NIFTY",
            "position_type": "SHORT_STRADDLE",
            "quantity": 50,
            "entry_price": 300
        }
        
        mock_db_manager.get_open_positions.return_value = [position1_data]
        
        # Try to create second position
        signal = {
            "symbol": "BANKNIFTY",
            "total_premium": 400,
            "metadata": {"spot_price": 45000}
        }
        
        # Should allow second position
        can_trade = await short_straddle_bot._check_risk_limits(signal)
        assert can_trade is True
        
        # Add second position
        position2_data = position1_data.copy()
        position2_data["symbol"] = "BANKNIFTY"
        mock_db_manager.get_open_positions.return_value = [position1_data, position2_data]
        
        # Try to create third position - should fail
        signal3 = signal.copy()
        signal3["symbol"] = "FINNIFTY"
        
        can_trade = await short_straddle_bot._check_risk_limits(signal3)
        assert can_trade is False  # Max positions reached
    
    @pytest.mark.asyncio
    async def test_market_data_processing(self, short_straddle_bot):
        """Test processing of live market data"""
        received_signals = []
        
        # Mock signal execution
        async def mock_execute(signal, size):
            received_signals.append(signal)
        
        short_straddle_bot.execute_signal = mock_execute
        
        # Start bot
        await short_straddle_bot.start()
        
        # Simulate market data updates
        for i in range(5):
            market_data = {
                "ltp": 20000 + i * 10,
                "volume": 1000000 + i * 10000,
                "timestamp": datetime.now().isoformat()
            }
            
            # Process market data
            await short_straddle_bot.on_market_data("NIFTY", market_data)
        
        # Bot should process data and generate signals if conditions met
        # (Actual signal generation depends on IV conditions)
    
    @pytest.mark.asyncio
    async def test_position_adjustment_workflow(self, short_straddle_bot, mock_openalgo_client):
        """Test position adjustment when delta becomes skewed"""
        # Create existing position
        short_straddle_bot.straddle_positions["NIFTY"] = {
            "call_symbol": "NIFTY24FEB20000CE",
            "put_symbol": "NIFTY24FEB20000PE",
            "strike": 20000,
            "quantity": 50,
            "entry_premium": 300
        }
        
        # Simulate market move causing delta skew
        # Call premium increases significantly
        mock_openalgo_client.get_quote.side_effect = [
            {"ltp": 250},  # Call price increased
            {"ltp": 50}    # Put price decreased
        ]
        
        # Check if adjustment needed
        position = {"symbol": "NIFTY", "id": 1}
        needs_adjustment = await short_straddle_bot._check_adjustment_needed(position)
        
        # Should trigger adjustment due to delta imbalance
        assert needs_adjustment is True
    
    @pytest.mark.asyncio
    async def test_capital_allocation_integration(self, short_straddle_bot, mock_db_manager):
        """Test capital allocation and updates"""
        # Initial capital
        short_straddle_bot.initial_capital = 500000
        short_straddle_bot.available_capital = 500000
        
        # Execute a trade
        signal = {
            "symbol": "NIFTY",
            "total_premium": 300,
            "metadata": {"spot_price": 20000}
        }
        
        position_size = 50
        margin_required = 60000  # Approximate margin for short straddle
        
        # Mock margin calculation
        with patch.object(short_straddle_bot, '_calculate_margin_requirement', return_value=margin_required):
            # Execute signal
            await short_straddle_bot.execute_signal(signal, position_size)
        
        # Verify capital updated
        mock_db_manager.update_bot_capital.assert_called_with(
            short_straddle_bot.name,
            locked_delta=-margin_required,
            pnl_delta=0
        )
    
    @pytest.mark.asyncio
    async def test_performance_tracking_integration(self, short_straddle_bot, mock_db_manager):
        """Test performance metrics tracking"""
        # Simulate completed trades
        trades = [
            {"pnl": 5000, "success": True},
            {"pnl": -2000, "success": False},
            {"pnl": 3000, "success": True},
            {"pnl": 4000, "success": True}
        ]
        
        # Process trades
        for trade in trades:
            short_straddle_bot.stats["total_trades"] += 1
            if trade["success"]:
                short_straddle_bot.stats["winning_trades"] += 1
            else:
                short_straddle_bot.stats["losing_trades"] += 1
            short_straddle_bot.stats["total_pnl"] += trade["pnl"]
        
        # Get bot status
        status = short_straddle_bot.get_status()
        
        # Verify performance metrics
        assert status["performance"]["total_trades"] == 4
        assert status["performance"]["win_rate"] == 75.0
        assert status["performance"]["total_pnl"] == 10000
    
    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, short_straddle_bot, mock_openalgo_client):
        """Test error handling and recovery"""
        # Simulate order placement failure
        mock_openalgo_client.place_order.side_effect = [
            {"status": "success", "orderid": "12345"},  # First leg succeeds
            Exception("Network error")  # Second leg fails
        ]
        
        signal = {
            "symbol": "NIFTY",
            "call_symbol": "NIFTY24FEB20000CE",
            "put_symbol": "NIFTY24FEB20000PE",
            "call_price": 150,
            "put_price": 140,
            "total_premium": 290
        }
        
        # Try to place order
        success = await short_straddle_bot._place_order(signal, 50)
        
        # Should handle partial execution gracefully
        assert success is False
        
        # Should attempt to cancel first leg
        mock_openalgo_client.cancel_order.assert_called_with("12345")
    
    @pytest.mark.asyncio
    async def test_multi_expiry_handling(self, short_straddle_bot, mock_openalgo_client):
        """Test handling multiple expiry dates"""
        # Mock option chain with multiple expiries
        expiry1 = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        expiry2 = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        
        option_chain = {
            expiry1: {
                "strikes": {
                    "20000": {
                        "CE": {"ltp": 100, "iv": 25},
                        "PE": {"ltp": 95, "iv": 24}
                    }
                }
            },
            expiry2: {
                "strikes": {
                    "20000": {
                        "CE": {"ltp": 150, "iv": 20},
                        "PE": {"ltp": 140, "iv": 19}
                    }
                }
            }
        }
        
        mock_openalgo_client.get_option_chain.return_value = option_chain
        
        # Bot should select appropriate expiry based on DTE criteria
        selected_expiry = await short_straddle_bot._select_expiry("NIFTY", option_chain)
        
        # Should select monthly expiry (30 days)
        assert selected_expiry == expiry2
    
    @pytest.mark.asyncio
    async def test_concurrent_position_management(self, short_straddle_bot):
        """Test managing multiple positions concurrently"""
        # Create multiple positions
        positions = []
        for i in range(3):
            pos = {
                "id": i + 1,
                "symbol": f"STOCK{i}",
                "entry_price": 100 + i * 10,
                "current_price": 95 + i * 10,
                "pnl": 250 * (i + 1)
            }
            positions.append(pos)
        
        # Process positions concurrently
        tasks = []
        for position in positions:
            task = short_straddle_bot.should_exit_position(position, {})
            tasks.append(task)
        
        # All tasks should complete without blocking each other
        results = await asyncio.gather(*tasks)
        
        # Each position evaluated independently
        assert len(results) == 3