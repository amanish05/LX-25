"""
Tests for Short Straddle Bot
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from src.bots.short_straddle_bot import ShortStraddleBot
from src.bots.base_bot import BotState


@pytest.fixture
def bot_config():
    """Test configuration for short straddle bot"""
    return {
        "name": "TestShortStraddleBot",
        "bot_type": "short_straddle",
        "enabled": True,
        "capital": 200000,
        "max_positions": 2,
        "lots_per_trade": 1,
        "product": "NRML",
        "entry": {
            "iv_rank_min": 75,
            "dte_min": 30,
            "dte_max": 45
        },
        "exit": {
            "profit_target_pct": 50,
            "stop_loss_multiplier": 2,
            "time_exit_dte": 21
        }
    }


@pytest.fixture
def mock_db_manager():
    """Mock database manager"""
    db_manager = Mock()
    db_manager.init_bot_capital = AsyncMock()
    db_manager.get_open_positions = AsyncMock(return_value=[])
    db_manager.create_position = AsyncMock(return_value=1)
    db_manager.update_position = AsyncMock()
    db_manager.save_signal = AsyncMock(return_value=1)
    db_manager.mark_signal_executed = AsyncMock()
    db_manager.update_bot_capital = AsyncMock()
    db_manager.update_bot_performance = AsyncMock()
    return db_manager


@pytest.fixture
def mock_openalgo_client():
    """Mock OpenAlgo client"""
    client = Mock()
    client.subscribe_market_data = AsyncMock()
    client.unsubscribe_market_data = AsyncMock()
    client.get_funds = AsyncMock(return_value={"available_balance": 200000})
    client.place_order = AsyncMock(return_value={"status": "success", "orderid": "12345"})
    client.cancel_order = AsyncMock()
    client.get_quote = AsyncMock(return_value={"ltp": 150})
    client.get_option_chain = AsyncMock(return_value={
        "2024-02-29": {
            "strikes": {
                "20000": {
                    "CE": {"ltp": 150, "iv": 20, "oi": 50000, "volume": 5000},
                    "PE": {"ltp": 140, "iv": 19, "oi": 45000, "volume": 4500}
                }
            }
        }
    })
    return client


@pytest.fixture
async def bot(bot_config, mock_db_manager, mock_openalgo_client):
    """Create bot instance"""
    bot = ShortStraddleBot(
        config=bot_config,
        db_manager=mock_db_manager,
        openalgo_client=mock_openalgo_client
    )
    return bot


class TestShortStraddleBot:
    """Test cases for Short Straddle Bot"""
    
    @pytest.mark.asyncio
    async def test_bot_initialization(self, bot):
        """Test bot initialization"""
        assert bot.name == "TestShortStraddleBot"
        assert bot.bot_type == "short_straddle"
        assert bot.initial_capital == 200000
        assert bot.iv_rank_threshold == 75
        assert bot.dte_min == 30
        assert bot.dte_max == 45
        assert bot.state == BotState.INITIALIZED
    
    @pytest.mark.asyncio
    async def test_bot_start(self, bot, mock_db_manager):
        """Test bot start process"""
        await bot.start()
        
        # Check state
        assert bot.state == BotState.RUNNING
        
        # Check if capital was initialized
        mock_db_manager.init_bot_capital.assert_called_once_with(
            "TestShortStraddleBot", 200000
        )
        
        # Check if positions were loaded
        mock_db_manager.get_open_positions.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_signal_generation_high_iv(self, bot):
        """Test signal generation when IV is high"""
        # Mock IV rank calculation
        with patch.object(bot, '_calculate_iv_rank', return_value=80):
            with patch.object(bot, '_get_option_data') as mock_get_option:
                mock_get_option.side_effect = [
                    {"symbol": "NIFTY24FEB20000CE", "ltp": 150, "iv": 20, "oi": 50000, "volume": 5000},
                    {"symbol": "NIFTY24FEB20000PE", "ltp": 140, "iv": 19, "oi": 45000, "volume": 4500}
                ]
                
                signal = await bot.generate_signals("NIFTY", {"ltp": 20000})
                
                assert signal is not None
                assert signal["type"] == "SHORT_STRADDLE"
                assert signal["strike"] == 20000
                assert signal["total_premium"] == 290
                assert signal["iv_rank"] == 80
    
    @pytest.mark.asyncio
    async def test_signal_generation_low_iv(self, bot):
        """Test no signal when IV is low"""
        with patch.object(bot, '_calculate_iv_rank', return_value=60):
            signal = await bot.generate_signals("NIFTY", {"ltp": 20000})
            assert signal is None
    
    @pytest.mark.asyncio
    async def test_position_size_calculation(self, bot):
        """Test position size calculation"""
        signal = {
            "symbol": "NIFTY",
            "total_premium": 300,
            "metadata": {"spot_price": 20000}
        }
        
        with patch.object(bot, '_calculate_margin_requirement', return_value=50000):
            size = await bot.calculate_position_size(signal)
            
            # 1 lot = 50 for NIFTY
            assert size == 50
    
    @pytest.mark.asyncio
    async def test_position_size_insufficient_capital(self, bot):
        """Test position size when capital is insufficient"""
        signal = {
            "symbol": "NIFTY",
            "total_premium": 300,
            "metadata": {"spot_price": 20000}
        }
        
        bot.available_capital = 10000  # Less than required
        
        with patch.object(bot, '_calculate_margin_requirement', return_value=50000):
            size = await bot.calculate_position_size(signal)
            assert size == 0
    
    @pytest.mark.asyncio
    async def test_should_enter_position(self, bot):
        """Test entry conditions check"""
        signal = {"symbol": "NIFTY", "metadata": {"call_iv": 20}}
        
        with patch.object(bot, '_check_market_conditions', return_value=True):
            with patch.object(bot, '_has_major_event_soon', return_value=False):
                with patch.object(bot, '_calculate_realized_volatility', return_value=15):
                    should_enter = await bot.should_enter_position(signal)
                    assert should_enter is True
    
    @pytest.mark.asyncio
    async def test_should_exit_profit_target(self, bot):
        """Test exit on profit target"""
        bot.straddle_positions["NIFTY"] = {
            "call_symbol": "NIFTY24FEB20000CE",
            "put_symbol": "NIFTY24FEB20000PE",
            "entry_premium": 300,
            "expiry": "2024-02-29",
            "entry_iv_rank": 80
        }
        
        position = {"symbol": "NIFTY"}
        
        with patch.object(bot, '_get_current_option_price') as mock_price:
            # Current premium = 150 (50% of entry)
            mock_price.side_effect = [75, 75]
            
            should_exit = await bot.should_exit_position(position, {})
            assert should_exit is True
    
    @pytest.mark.asyncio
    async def test_should_exit_stop_loss(self, bot):
        """Test exit on stop loss"""
        bot.straddle_positions["NIFTY"] = {
            "call_symbol": "NIFTY24FEB20000CE",
            "put_symbol": "NIFTY24FEB20000PE",
            "entry_premium": 300,
            "expiry": "2024-02-29",
            "entry_iv_rank": 80
        }
        
        position = {"symbol": "NIFTY"}
        
        with patch.object(bot, '_get_current_option_price') as mock_price:
            # Current premium = 900 (3x of entry - stop loss)
            mock_price.side_effect = [450, 450]
            
            should_exit = await bot.should_exit_position(position, {})
            assert should_exit is True
    
    @pytest.mark.asyncio
    async def test_place_straddle_order(self, bot, mock_openalgo_client):
        """Test placing both legs of straddle"""
        signal = {
            "symbol": "NIFTY",
            "call_symbol": "NIFTY24FEB20000CE",
            "put_symbol": "NIFTY24FEB20000PE",
            "call_price": 150,
            "put_price": 140,
            "total_premium": 290,
            "strike": 20000,
            "expiry": "2024-02-29",
            "iv_rank": 80
        }
        
        success = await bot._place_order(signal, 50)
        
        assert success is True
        assert "NIFTY" in bot.straddle_positions
        assert bot.straddle_positions["NIFTY"]["entry_premium"] == 290
        
        # Check both orders were placed
        assert mock_openalgo_client.place_order.call_count == 2
    
    @pytest.mark.asyncio
    async def test_close_straddle_position(self, bot, mock_openalgo_client):
        """Test closing both legs of straddle"""
        bot.straddle_positions["NIFTY"] = {
            "call_symbol": "NIFTY24FEB20000CE",
            "put_symbol": "NIFTY24FEB20000PE",
            "quantity": 50
        }
        
        position = {"id": 1, "symbol": "NIFTY"}
        
        await bot._close_position(position)
        
        # Check both closing orders were placed
        assert mock_openalgo_client.place_order.call_count == 2
        assert "NIFTY" not in bot.straddle_positions
    
    @pytest.mark.asyncio
    async def test_margin_calculation(self, bot):
        """Test margin requirement calculation"""
        signal = {
            "metadata": {"spot_price": 20000}
        }
        
        margin = bot._calculate_margin_requirement(signal)
        
        # 15% of spot * 2 legs
        expected_margin = 20000 * 0.15 * 2
        assert margin == expected_margin
    
    @pytest.mark.asyncio
    async def test_get_atm_strike(self, bot):
        """Test ATM strike calculation"""
        # Test NIFTY (50 point strikes)
        atm = bot._get_atm_strike(19973, {"NIFTY": {}}, "2024-02-29")
        assert atm == 20000
        
        # Test BANKNIFTY (100 point strikes)
        atm = bot._get_atm_strike(45650, {"BANKNIFTY": {}}, "2024-02-29")
        assert atm == 45700
    
    @pytest.mark.asyncio
    async def test_check_liquidity(self, bot):
        """Test liquidity check"""
        call_data = {"oi": 5000, "volume": 500}
        put_data = {"oi": 4500, "volume": 450}
        
        assert bot._check_liquidity(call_data, put_data) is True
        
        # Test with low liquidity
        call_data = {"oi": 500, "volume": 50}
        put_data = {"oi": 450, "volume": 45}
        
        assert bot._check_liquidity(call_data, put_data) is False
    
    @pytest.mark.asyncio
    async def test_bot_status(self, bot):
        """Test bot status reporting"""
        bot.stats["total_trades"] = 10
        bot.stats["winning_trades"] = 7
        bot.stats["total_pnl"] = 15000
        
        status = bot.get_status()
        
        assert status["name"] == "TestShortStraddleBot"
        assert status["type"] == "short_straddle"
        assert status["performance"]["total_trades"] == 10
        assert status["performance"]["win_rate"] == 70.0
        assert status["performance"]["total_pnl"] == 15000


if __name__ == "__main__":
    pytest.main(["-v", __file__])