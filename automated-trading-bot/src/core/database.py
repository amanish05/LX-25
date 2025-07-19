"""
Database Module for Automated Trading Bot
Handles all database operations with async support
"""

import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import json
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, 
    Text, ForeignKey, UniqueConstraint, Index, select, and_, or_
)
from sqlalchemy.dialects.postgresql import UUID
import uuid

from ..utils.logger import TradingLogger


Base = declarative_base()


class BotPosition(Base):
    """Bot positions table"""
    __tablename__ = 'bot_positions'
    
    id = Column(Integer, primary_key=True)
    bot_name = Column(String(100), nullable=False, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    exchange = Column(String(20), nullable=False)
    position_type = Column(String(10), nullable=False)  # LONG, SHORT
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float)
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime)
    exit_price = Column(Float)
    pnl = Column(Float)
    status = Column(String(20), default='OPEN', index=True)
    strategy_params = Column(Text)  # JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    trades = relationship("BotTrade", back_populates="position")
    
    __table_args__ = (
        Index('idx_bot_positions_bot_status', 'bot_name', 'status'),
        Index('idx_bot_positions_symbol_status', 'symbol', 'status'),
    )


class BotTrade(Base):
    """Bot trades table"""
    __tablename__ = 'bot_trades'
    
    id = Column(Integer, primary_key=True)
    bot_name = Column(String(100), nullable=False, index=True)
    position_id = Column(Integer, ForeignKey('bot_positions.id'))
    order_id = Column(String(100))
    symbol = Column(String(50), nullable=False)
    exchange = Column(String(20), nullable=False)
    action = Column(String(10), nullable=False)  # BUY, SELL
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    order_type = Column(String(20))  # MARKET, LIMIT
    status = Column(String(20))  # PENDING, EXECUTED, CANCELLED
    executed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    position = relationship("BotPosition", back_populates="trades")


class BotPerformance(Base):
    """Bot performance metrics table"""
    __tablename__ = 'bot_performance'
    
    id = Column(Integer, primary_key=True)
    bot_name = Column(String(100), nullable=False)
    date = Column(DateTime, nullable=False)
    capital_allocated = Column(Float, nullable=False)
    capital_used = Column(Float, nullable=False)
    realized_pnl = Column(Float, default=0)
    unrealized_pnl = Column(Float, default=0)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    max_drawdown = Column(Float, default=0)
    sharpe_ratio = Column(Float)
    win_rate = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('bot_name', 'date', name='uq_bot_performance_bot_date'),
        Index('idx_bot_performance_bot_date', 'bot_name', 'date'),
    )


class BotSignal(Base):
    """Bot signals table"""
    __tablename__ = 'bot_signals'
    
    id = Column(Integer, primary_key=True)
    bot_name = Column(String(100), nullable=False, index=True)
    symbol = Column(String(50), nullable=False)
    exchange = Column(String(20), nullable=False)
    signal_type = Column(String(20), nullable=False)  # BUY, SELL, HOLD
    signal_strength = Column(Float)
    signal_data = Column(Text)  # JSON
    executed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_bot_signals_bot_created', 'bot_name', 'created_at'),
    )


class MarketDataCache(Base):
    """Market data cache table"""
    __tablename__ = 'market_data_cache'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False)
    exchange = Column(String(20), nullable=False)
    data_type = Column(String(20), nullable=False)  # QUOTE, DEPTH, HISTORICAL
    timeframe = Column(String(10))
    data = Column(Text, nullable=False)  # JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'exchange', 'data_type', 'timeframe', 
                        name='uq_market_data_cache'),
        Index('idx_market_data_cache_expires', 'expires_at'),
    )


class BotCapital(Base):
    """Bot capital tracking table"""
    __tablename__ = 'bot_capital'
    
    id = Column(Integer, primary_key=True)
    bot_name = Column(String(100), nullable=False, unique=True)
    initial_capital = Column(Float, nullable=False)
    current_capital = Column(Float, nullable=False)
    locked_capital = Column(Float, default=0)
    available_capital = Column(Float, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Async database manager for trading bot system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Use environment variable for database URL or fail with clear message
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError(
                "DATABASE_URL environment variable is not set. "
                "Please check your .env file and ensure DATABASE_URL is properly configured."
            )
        self.logger = TradingLogger(__name__)
        
        # Create async engine
        db_config = config.get("database", config.get("domains", {}).get("database", {}))
        self.engine = create_async_engine(
            self.db_url,
            echo=db_config.get("echo", False),
            pool_size=db_config.get("pool_size", 5),
            max_overflow=db_config.get("max_overflow", 10)
        )
        
        # Create session factory
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def init_database(self):
        """Initialize database tables"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self.logger.info("Database tables initialized")
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session"""
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    # Position Management
    async def create_position(self, bot_name: str, symbol: str, exchange: str,
                            position_type: str, quantity: int, entry_price: float,
                            strategy_params: Dict[str, Any] = None) -> int:
        """Create a new position"""
        async with self.get_session() as session:
            position = BotPosition(
                bot_name=bot_name,
                symbol=symbol,
                exchange=exchange,
                position_type=position_type,
                quantity=quantity,
                entry_price=entry_price,
                current_price=entry_price,
                strategy_params=json.dumps(strategy_params) if strategy_params else None
            )
            session.add(position)
            await session.flush()
            
            self.logger.position_opened(
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                bot_name=bot_name
            )
            
            return position.id
    
    async def update_position(self, position_id: int, **kwargs):
        """Update an existing position"""
        async with self.get_session() as session:
            result = await session.execute(
                select(BotPosition).where(BotPosition.id == position_id)
            )
            position = result.scalar_one_or_none()
            
            if position:
                for key, value in kwargs.items():
                    if hasattr(position, key):
                        setattr(position, key, value)
                
                # Calculate PnL if closing
                if kwargs.get('status') == 'CLOSED' and kwargs.get('exit_price'):
                    exit_price = kwargs['exit_price']
                    if position.position_type == 'LONG':
                        position.pnl = (exit_price - position.entry_price) * position.quantity
                    else:
                        position.pnl = (position.entry_price - exit_price) * position.quantity
                    
                    self.logger.position_closed(
                        symbol=position.symbol,
                        quantity=position.quantity,
                        exit_price=exit_price,
                        pnl=position.pnl,
                        bot_name=position.bot_name
                    )
    
    async def get_open_positions(self, bot_name: str = None) -> List[Dict[str, Any]]:
        """Get all open positions"""
        async with self.get_session() as session:
            query = select(BotPosition).where(BotPosition.status == 'OPEN')
            
            if bot_name:
                query = query.where(BotPosition.bot_name == bot_name)
            
            result = await session.execute(query.order_by(BotPosition.entry_time.desc()))
            positions = result.scalars().all()
            
            return [self._position_to_dict(pos) for pos in positions]
    
    # Trade Management
    async def record_trade(self, bot_name: str, position_id: int, order_id: str,
                         symbol: str, exchange: str, action: str, quantity: int,
                         price: float, order_type: str = "MARKET",
                         status: str = "EXECUTED") -> int:
        """Record a trade execution"""
        async with self.get_session() as session:
            trade = BotTrade(
                bot_name=bot_name,
                position_id=position_id,
                order_id=order_id,
                symbol=symbol,
                exchange=exchange,
                action=action,
                quantity=quantity,
                price=price,
                order_type=order_type,
                status=status,
                executed_at=datetime.utcnow() if status == "EXECUTED" else None
            )
            session.add(trade)
            await session.flush()
            
            self.logger.trade_executed(
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=price,
                order_id=order_id,
                bot_name=bot_name
            )
            
            return trade.id
    
    # Performance Tracking
    async def update_bot_performance(self, bot_name: str, date: datetime = None):
        """Update bot performance metrics"""
        if date is None:
            date = datetime.utcnow().date()
        
        async with self.get_session() as session:
            # Get capital info
            capital_result = await session.execute(
                select(BotCapital).where(BotCapital.bot_name == bot_name)
            )
            capital_info = capital_result.scalar_one_or_none()
            
            if not capital_info:
                self.logger.warning(f"No capital info found for bot {bot_name}")
                return
            
            # Calculate metrics for the day
            # Get closed positions
            closed_positions = await session.execute(
                select(BotPosition).where(
                    and_(
                        BotPosition.bot_name == bot_name,
                        BotPosition.status == 'CLOSED',
                        BotPosition.exit_time >= date,
                        BotPosition.exit_time < date + timedelta(days=1)
                    )
                )
            )
            closed_pos_list = closed_positions.scalars().all()
            
            # Calculate realized PnL
            realized_pnl = sum(pos.pnl or 0 for pos in closed_pos_list)
            winning_trades = sum(1 for pos in closed_pos_list if (pos.pnl or 0) > 0)
            losing_trades = sum(1 for pos in closed_pos_list if (pos.pnl or 0) < 0)
            total_trades = len(closed_pos_list)
            
            # Get open positions for unrealized PnL
            open_positions = await session.execute(
                select(BotPosition).where(
                    and_(
                        BotPosition.bot_name == bot_name,
                        BotPosition.status == 'OPEN'
                    )
                )
            )
            open_pos_list = open_positions.scalars().all()
            
            # Calculate unrealized PnL (simplified - would need current prices)
            unrealized_pnl = sum(
                (pos.current_price - pos.entry_price) * pos.quantity 
                if pos.position_type == 'LONG' else
                (pos.entry_price - pos.current_price) * pos.quantity
                for pos in open_pos_list
            )
            
            # Check if performance record exists
            existing = await session.execute(
                select(BotPerformance).where(
                    and_(
                        BotPerformance.bot_name == bot_name,
                        BotPerformance.date == date
                    )
                )
            )
            performance = existing.scalar_one_or_none()
            
            if performance:
                # Update existing
                performance.capital_allocated = capital_info.initial_capital
                performance.capital_used = capital_info.locked_capital
                performance.realized_pnl = realized_pnl
                performance.unrealized_pnl = unrealized_pnl
                performance.total_trades = total_trades
                performance.winning_trades = winning_trades
                performance.losing_trades = losing_trades
                performance.win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            else:
                # Create new
                performance = BotPerformance(
                    bot_name=bot_name,
                    date=date,
                    capital_allocated=capital_info.initial_capital,
                    capital_used=capital_info.locked_capital,
                    realized_pnl=realized_pnl,
                    unrealized_pnl=unrealized_pnl,
                    total_trades=total_trades,
                    winning_trades=winning_trades,
                    losing_trades=losing_trades,
                    win_rate=(winning_trades / total_trades * 100) if total_trades > 0 else 0
                )
                session.add(performance)
    
    # Signal Management
    async def save_signal(self, bot_name: str, symbol: str, exchange: str,
                        signal_type: str, signal_strength: float = None,
                        signal_data: Dict[str, Any] = None) -> int:
        """Save a trading signal"""
        async with self.get_session() as session:
            signal = BotSignal(
                bot_name=bot_name,
                symbol=symbol,
                exchange=exchange,
                signal_type=signal_type,
                signal_strength=signal_strength,
                signal_data=json.dumps(signal_data) if signal_data else None
            )
            session.add(signal)
            await session.flush()
            
            self.logger.signal_generated(
                symbol=symbol,
                signal_type=signal_type,
                strength=signal_strength or 0,
                bot_name=bot_name
            )
            
            return signal.id
    
    async def mark_signal_executed(self, signal_id: int):
        """Mark a signal as executed"""
        async with self.get_session() as session:
            result = await session.execute(
                select(BotSignal).where(BotSignal.id == signal_id)
            )
            signal = result.scalar_one_or_none()
            
            if signal:
                signal.executed = True
    
    # Capital Management
    async def init_bot_capital(self, bot_name: str, initial_capital: float):
        """Initialize capital for a bot"""
        async with self.get_session() as session:
            # Check if exists
            existing = await session.execute(
                select(BotCapital).where(BotCapital.bot_name == bot_name)
            )
            capital = existing.scalar_one_or_none()
            
            if capital:
                capital.initial_capital = initial_capital
                capital.current_capital = initial_capital
                capital.available_capital = initial_capital
                capital.locked_capital = 0
            else:
                capital = BotCapital(
                    bot_name=bot_name,
                    initial_capital=initial_capital,
                    current_capital=initial_capital,
                    available_capital=initial_capital,
                    locked_capital=0
                )
                session.add(capital)
    
    async def update_bot_capital(self, bot_name: str, locked_capital: float = None,
                               capital_change: float = None):
        """Update bot capital allocation"""
        async with self.get_session() as session:
            result = await session.execute(
                select(BotCapital).where(BotCapital.bot_name == bot_name)
            )
            capital = result.scalar_one_or_none()
            
            if capital:
                if locked_capital is not None:
                    capital.locked_capital = locked_capital
                    capital.available_capital = capital.current_capital - locked_capital
                
                if capital_change is not None:
                    capital.current_capital += capital_change
                    capital.available_capital += capital_change
                
                capital.last_updated = datetime.utcnow()
    
    async def get_bot_capital(self, bot_name: str) -> Optional[Dict[str, Any]]:
        """Get capital information for a bot"""
        async with self.get_session() as session:
            result = await session.execute(
                select(BotCapital).where(BotCapital.bot_name == bot_name)
            )
            capital = result.scalar_one_or_none()
            
            if capital:
                return {
                    'initial_capital': capital.initial_capital,
                    'current_capital': capital.current_capital,
                    'locked_capital': capital.locked_capital,
                    'available_capital': capital.available_capital,
                    'last_updated': capital.last_updated
                }
            return None
    
    # Market Data Cache
    async def cache_market_data(self, symbol: str, exchange: str, data_type: str,
                              data: Dict[str, Any], timeframe: str = None,
                              ttl_seconds: int = 300):
        """Cache market data with TTL"""
        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        
        async with self.get_session() as session:
            # Check if exists
            query = select(MarketDataCache).where(
                and_(
                    MarketDataCache.symbol == symbol,
                    MarketDataCache.exchange == exchange,
                    MarketDataCache.data_type == data_type,
                    MarketDataCache.timeframe == timeframe
                )
            )
            existing = await session.execute(query)
            cache_entry = existing.scalar_one_or_none()
            
            if cache_entry:
                cache_entry.data = json.dumps(data)
                cache_entry.created_at = datetime.utcnow()
                cache_entry.expires_at = expires_at
            else:
                cache_entry = MarketDataCache(
                    symbol=symbol,
                    exchange=exchange,
                    data_type=data_type,
                    timeframe=timeframe,
                    data=json.dumps(data),
                    expires_at=expires_at
                )
                session.add(cache_entry)
    
    async def get_cached_market_data(self, symbol: str, exchange: str,
                                   data_type: str, timeframe: str = None) -> Optional[Dict[str, Any]]:
        """Get cached market data if not expired"""
        async with self.get_session() as session:
            query = select(MarketDataCache).where(
                and_(
                    MarketDataCache.symbol == symbol,
                    MarketDataCache.exchange == exchange,
                    MarketDataCache.data_type == data_type,
                    MarketDataCache.timeframe == timeframe,
                    MarketDataCache.expires_at > datetime.utcnow()
                )
            ).order_by(MarketDataCache.created_at.desc())
            
            result = await session.execute(query.limit(1))
            cache_entry = result.scalar_one_or_none()
            
            if cache_entry:
                return json.loads(cache_entry.data)
            return None
    
    async def cleanup_expired_cache(self):
        """Remove expired cache entries"""
        async with self.get_session() as session:
            await session.execute(
                MarketDataCache.__table__.delete().where(
                    MarketDataCache.expires_at < datetime.utcnow()
                )
            )
            self.logger.debug("Cleaned up expired cache entries")
    
    # Helper methods
    def _position_to_dict(self, position: BotPosition) -> Dict[str, Any]:
        """Convert position object to dictionary"""
        return {
            'id': position.id,
            'bot_name': position.bot_name,
            'symbol': position.symbol,
            'exchange': position.exchange,
            'position_type': position.position_type,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'current_price': position.current_price,
            'entry_time': position.entry_time,
            'exit_time': position.exit_time,
            'exit_price': position.exit_price,
            'pnl': position.pnl,
            'status': position.status,
            'strategy_params': json.loads(position.strategy_params) if position.strategy_params else None
        }
    
    async def close(self):
        """Close database connections"""
        await self.engine.dispose()


async def init_database(config_manager):
    """Initialize database with tables"""
    db_manager = DatabaseManager(config_manager.app_config.to_dict())
    await db_manager.init_database()
    await db_manager.close()