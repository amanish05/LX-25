"""
Integration Tests for Database Module
Tests database operations with real SQLite database
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import select
import json

from src.core.database import (
    DatabaseManager, Base, BotPosition, BotTrade, 
    BotPerformance, BotSignal, BotCapital, MarketDataCache
)


class TestDatabaseIntegration:
    """Integration tests for Database operations"""
    
    @pytest.mark.asyncio
    async def test_database_initialization(self, test_config):
        """Test database initialization and table creation"""
        db_manager = DatabaseManager(test_config)
        
        # Initialize database
        await db_manager.init_database()
        
        # Verify tables exist by trying to query them
        async with db_manager.get_session() as session:
            # Try to select from each table
            await session.execute(select(BotPosition).limit(1))
            await session.execute(select(BotTrade).limit(1))
            await session.execute(select(BotPerformance).limit(1))
            await session.execute(select(BotSignal).limit(1))
            await session.execute(select(BotCapital).limit(1))
            await session.execute(select(MarketDataCache).limit(1))
        
        await db_manager.close()
    
    @pytest.mark.asyncio
    async def test_concurrent_database_access(self, test_config):
        """Test concurrent database operations"""
        db_manager = DatabaseManager(test_config)
        await db_manager.init_database()
        
        # Initialize bot capital for multiple bots
        bot_names = [f"Bot{i}" for i in range(5)]
        capital = 200000
        
        # Concurrent initialization
        tasks = [
            db_manager.init_bot_capital(bot_name, capital)
            for bot_name in bot_names
        ]
        await asyncio.gather(*tasks)
        
        # Verify all capitals created
        for bot_name in bot_names:
            capital_info = await db_manager.get_bot_capital(bot_name)
            assert capital_info["initial"] == capital
            assert capital_info["current"] == capital
            assert capital_info["available"] == capital
            assert capital_info["locked"] == 0
        
        await db_manager.close()
    
    @pytest.mark.asyncio
    async def test_position_lifecycle(self, test_config):
        """Test complete position lifecycle: create, update, close"""
        db_manager = DatabaseManager(test_config)
        await db_manager.init_database()
        
        bot_name = "TestBot"
        await db_manager.init_bot_capital(bot_name, 500000)
        
        # Create position
        position_data = {
            "bot_name": bot_name,
            "symbol": "NIFTY",
            "exchange": "NFO",
            "position_type": "SHORT",
            "quantity": 50,
            "entry_price": 150.0,
            "strategy_params": {"strike": 20000, "expiry": "2024-02-29"}
        }
        
        position_id = await db_manager.create_position(**position_data)
        assert position_id is not None
        
        # Update position
        update_data = {
            "current_price": 140.0,
            "pnl": 500.0
        }
        await db_manager.update_position(position_id, update_data)
        
        # Get open positions
        positions = await db_manager.get_open_positions(bot_name)
        assert len(positions) == 1
        assert positions[0]["id"] == position_id
        assert positions[0]["current_price"] == 140.0
        assert positions[0]["pnl"] == 500.0
        
        # Close position
        await db_manager.close_position(position_id, exit_price=140.0, pnl=500.0)
        
        # Verify position closed
        positions = await db_manager.get_open_positions(bot_name)
        assert len(positions) == 0
        
        # Get position by ID
        position = await db_manager.get_position_by_id(position_id)
        assert position["status"] == "CLOSED"
        assert position["exit_price"] == 140.0
        
        await db_manager.close()
    
    @pytest.mark.asyncio
    async def test_trade_recording(self, test_config):
        """Test trade recording and retrieval"""
        db_manager = DatabaseManager(test_config)
        await db_manager.init_database()
        
        bot_name = "TestBot"
        
        # Create multiple trades
        trades_data = [
            {
                "bot_name": bot_name,
                "symbol": "NIFTY24FEB20000CE",
                "exchange": "NFO",
                "action": "SELL",
                "quantity": 50,
                "price": 150.0,
                "order_id": f"ORDER{i}",
                "trade_time": datetime.now() - timedelta(hours=i)
            }
            for i in range(5)
        ]
        
        # Record trades
        trade_ids = []
        for trade_data in trades_data:
            trade_id = await db_manager.create_trade(**trade_data)
            trade_ids.append(trade_id)
        
        # Get trades
        trades = await db_manager.get_trades(
            bot_name=bot_name,
            limit=3
        )
        
        assert len(trades) == 3
        # Should be ordered by trade_time desc
        assert trades[0]["order_id"] == "ORDER0"
        
        await db_manager.close()
    
    @pytest.mark.asyncio
    async def test_signal_management(self, test_config):
        """Test signal creation and execution tracking"""
        db_manager = DatabaseManager(test_config)
        await db_manager.init_database()
        
        bot_name = "TestBot"
        
        # Save signal
        signal_data = {
            "bot_name": bot_name,
            "symbol": "NIFTY",
            "signal_type": "SHORT_STRADDLE",
            "signal_strength": 0.85,
            "metadata": {
                "iv_rank": 82,
                "strike": 20000,
                "expiry": "2024-02-29"
            }
        }
        
        signal_id = await db_manager.save_signal(**signal_data)
        assert signal_id is not None
        
        # Get unexecuted signals
        signals = await db_manager.get_signals(bot_name=bot_name, executed=False)
        assert len(signals) == 1
        assert signals[0]["id"] == signal_id
        
        # Mark signal as executed
        await db_manager.mark_signal_executed(signal_id, order_ids=["ORDER123", "ORDER456"])
        
        # Verify signal marked as executed
        signals = await db_manager.get_signals(bot_name=bot_name, executed=True)
        assert len(signals) == 1
        assert signals[0]["executed"] is True
        
        await db_manager.close()
    
    @pytest.mark.asyncio
    async def test_capital_management(self, test_config):
        """Test capital allocation and updates"""
        db_manager = DatabaseManager(test_config)
        await db_manager.init_database()
        
        bot_name = "TestBot"
        initial_capital = 500000
        
        # Initialize capital
        await db_manager.init_bot_capital(bot_name, initial_capital)
        
        # Update capital after trade
        position_capital = 50000
        pnl = 2500
        
        await db_manager.update_bot_capital(
            bot_name,
            locked_delta=-position_capital,  # Lock capital for position
            pnl_delta=0
        )
        
        # Verify capital update
        capital = await db_manager.get_bot_capital(bot_name)
        assert capital["available"] == initial_capital - position_capital
        assert capital["locked"] == position_capital
        
        # Close position and update capital
        await db_manager.update_bot_capital(
            bot_name,
            locked_delta=position_capital,  # Release locked capital
            pnl_delta=pnl
        )
        
        # Verify final capital
        capital = await db_manager.get_bot_capital(bot_name)
        assert capital["current"] == initial_capital + pnl
        assert capital["available"] == initial_capital + pnl
        assert capital["locked"] == 0
        
        await db_manager.close()
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, test_config):
        """Test performance metrics calculation and storage"""
        db_manager = DatabaseManager(test_config)
        await db_manager.init_database()
        
        bot_name = "TestBot"
        
        # Create some completed trades for performance calculation
        trades = [
            {"pnl": 1000, "win": True},
            {"pnl": -500, "win": False},
            {"pnl": 1500, "win": True},
            {"pnl": -200, "win": False},
            {"pnl": 800, "win": True},
        ]
        
        # Create positions with trades
        for i, trade in enumerate(trades):
            position_id = await db_manager.create_position(
                bot_name=bot_name,
                symbol="NIFTY",
                exchange="NFO",
                position_type="SHORT",
                quantity=50,
                entry_price=150.0
            )
            
            # Close position with PnL
            await db_manager.close_position(
                position_id,
                exit_price=150.0 - (trade["pnl"] / 50),
                pnl=trade["pnl"]
            )
        
        # Update performance metrics
        await db_manager.update_bot_performance(bot_name)
        
        # Get performance
        performance = await db_manager.get_bot_performance(bot_name)
        
        assert performance["total_trades"] == 5
        assert performance["winning_trades"] == 3
        assert performance["losing_trades"] == 2
        assert performance["total_pnl"] == 2600  # Sum of all PnLs
        assert performance["win_rate"] == 60.0  # 3/5 * 100
        
        await db_manager.close()
    
    @pytest.mark.asyncio
    async def test_market_data_caching(self, test_config):
        """Test market data caching functionality"""
        db_manager = DatabaseManager(test_config)
        await db_manager.init_database()
        
        # Cache market data
        symbol = "NIFTY"
        data_type = "quote"
        data = {
            "ltp": 20000,
            "open": 19950,
            "high": 20100,
            "low": 19900,
            "volume": 1000000
        }
        
        await db_manager.cache_market_data(symbol, data_type, data)
        
        # Retrieve cached data
        cached_data = await db_manager.get_cached_market_data(symbol, data_type)
        assert cached_data is not None
        assert cached_data["ltp"] == 20000
        
        # Test cache expiry (immediate for testing)
        await db_manager.cleanup_expired_cache(max_age_seconds=0)
        
        # Verify cache cleared
        cached_data = await db_manager.get_cached_market_data(symbol, data_type)
        assert cached_data is None
        
        await db_manager.close()
    
    @pytest.mark.asyncio
    async def test_transaction_rollback(self, test_config):
        """Test transaction rollback on error"""
        db_manager = DatabaseManager(test_config)
        await db_manager.init_database()
        
        bot_name = "TestBot"
        await db_manager.init_bot_capital(bot_name, 500000)
        
        # Try to create position with invalid data
        try:
            async with db_manager.get_session() as session:
                # Start transaction
                position = BotPosition(
                    bot_name=bot_name,
                    symbol="NIFTY",
                    exchange="NFO",
                    position_type="INVALID_TYPE",  # This should fail
                    quantity=50,
                    entry_price=150.0
                )
                session.add(position)
                await session.commit()
        except Exception:
            # Transaction should rollback
            pass
        
        # Verify no position was created
        positions = await db_manager.get_open_positions(bot_name)
        assert len(positions) == 0
        
        await db_manager.close()
    
    @pytest.mark.asyncio
    async def test_bulk_operations(self, test_config):
        """Test bulk insert and update operations"""
        db_manager = DatabaseManager(test_config)
        await db_manager.init_database()
        
        bot_name = "TestBot"
        
        # Bulk create positions
        positions_data = [
            {
                "bot_name": bot_name,
                "symbol": f"STOCK{i}",
                "exchange": "NSE",
                "position_type": "LONG",
                "quantity": 100,
                "entry_price": 100.0 + i
            }
            for i in range(100)
        ]
        
        # Time bulk insert
        start_time = datetime.now()
        
        async with db_manager.get_session() as session:
            positions = [BotPosition(**data) for data in positions_data]
            session.add_all(positions)
            await session.commit()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should be fast (less than 1 second for 100 records)
        assert duration < 1.0
        
        # Verify all positions created
        positions = await db_manager.get_open_positions(bot_name)
        assert len(positions) == 100
        
        await db_manager.close()
    
    @pytest.mark.asyncio
    async def test_complex_queries(self, test_config):
        """Test complex query scenarios"""
        db_manager = DatabaseManager(test_config)
        await db_manager.init_database()
        
        # Create positions for multiple bots
        bots = ["Bot1", "Bot2", "Bot3"]
        for bot_name in bots:
            await db_manager.init_bot_capital(bot_name, 300000)
            
            # Create positions with different symbols
            for symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
                await db_manager.create_position(
                    bot_name=bot_name,
                    symbol=symbol,
                    exchange="NFO",
                    position_type="SHORT",
                    quantity=50,
                    entry_price=150.0
                )
        
        # Query positions by symbol across all bots
        async with db_manager.get_session() as session:
            result = await session.execute(
                select(BotPosition).where(
                    BotPosition.symbol == "NIFTY",
                    BotPosition.status == "OPEN"
                )
            )
            nifty_positions = result.scalars().all()
            assert len(nifty_positions) == 3
        
        await db_manager.close()
    
    @pytest.mark.asyncio
    async def test_database_connection_pool(self, test_config):
        """Test connection pooling under load"""
        db_manager = DatabaseManager(test_config)
        await db_manager.init_database()
        
        # Create many concurrent operations
        async def db_operation(i):
            # Each operation gets a session from pool
            position_id = await db_manager.create_position(
                bot_name=f"Bot{i % 5}",
                symbol="NIFTY",
                exchange="NFO",
                position_type="LONG",
                quantity=50,
                entry_price=20000.0
            )
            
            # Update position
            await db_manager.update_position(
                position_id,
                {"current_price": 20100.0, "pnl": 5000.0}
            )
            
            return position_id
        
        # Run 50 concurrent operations
        tasks = [db_operation(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        
        # All operations should succeed
        assert len(results) == 50
        assert all(r is not None for r in results)
        
        await db_manager.close()