"""
Test database connectivity and basic operations
"""

import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, text
import os

from src.core.database import Base, BotCapital


class TestDatabaseConnectivity:
    """Test database connectivity and basic operations"""
    
    @pytest.mark.asyncio
    async def test_database_connection(self, test_db):
        """Test that we can connect to the test database"""
        engine = create_async_engine(test_db, echo=False)
        
        try:
            async with engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                assert result.scalar() == 1
                
            print("✅ Database connection successful")
            
        finally:
            await engine.dispose()
    
    @pytest.mark.asyncio
    async def test_basic_crud_operations(self, test_db):
        """Test basic CRUD operations"""
        engine = create_async_engine(test_db, echo=False)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        try:
            async with async_session() as session:
                # Create
                bot_capital = BotCapital(
                    bot_name="TestBot",
                    initial_capital=100000.0,
                    current_capital=100000.0,
                    available_capital=100000.0,
                    locked_capital=0.0
                )
                
                session.add(bot_capital)
                await session.commit()
                
                # Read
                result = await session.execute(
                    select(BotCapital).where(BotCapital.bot_name == "TestBot")
                )
                saved_capital = result.scalar_one()
                
                assert saved_capital.initial_capital == 100000.0
                assert saved_capital.bot_name == "TestBot"
                
                print("✅ CRUD operations successful")
                
        finally:
            await engine.dispose()