"""
TimescaleDB Manager for High-Performance Time Series Data
Optimized for tick data, OHLCV, and real-time analytics
"""

import asyncio
import asyncpg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from contextlib import asynccontextmanager

from src.data.historical_data_collector import TickData


@dataclass
class TimescaleConfig:
    """Configuration for TimescaleDB connection"""
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_bot"
    username: str = "postgres"
    password: str = ""
    pool_min_size: int = 5
    pool_max_size: int = 20
    command_timeout: int = 60


class TimescaleManager:
    """
    High-performance TimescaleDB manager for trading data
    
    Features:
    - Automatic hypertable creation and management
    - Optimized tick data storage with compression
    - Real-time data streaming capabilities
    - Continuous aggregates for OHLCV generation
    - Retention policies for data lifecycle management
    """
    
    def __init__(self, config: Optional[TimescaleConfig] = None):
        """Initialize TimescaleDB manager"""
        self.config = config or TimescaleConfig()
        self.logger = logging.getLogger(__name__)
        self.pool: Optional[asyncpg.Pool] = None
        
        # Table schemas
        self.schemas = {
            'tick_data': {
                'table': 'tick_data',
                'time_column': 'timestamp',
                'chunk_interval': '1 hour',
                'compression_after': '7 days',
                'retention_period': '2 years'
            },
            'ohlcv_data': {
                'table': 'ohlcv_data',
                'time_column': 'timestamp',
                'chunk_interval': '1 day',
                'compression_after': '30 days',
                'retention_period': '5 years'
            },
            'order_book': {
                'table': 'order_book',
                'time_column': 'timestamp',
                'chunk_interval': '1 hour',
                'compression_after': '3 days',
                'retention_period': '30 days'
            },
            'trades': {
                'table': 'trades',
                'time_column': 'timestamp',
                'chunk_interval': '1 hour',
                'compression_after': '7 days',
                'retention_period': '1 year'
            },
            'market_regime': {
                'table': 'market_regime',
                'time_column': 'timestamp',
                'chunk_interval': '1 day',
                'compression_after': '90 days',
                'retention_period': '3 years'
            }
        }
    
    async def initialize(self):
        """Initialize TimescaleDB connection and create tables"""
        self.logger.info("Initializing TimescaleDB connection...")
        
        # Create connection pool
        self.pool = await asyncpg.create_pool(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.username,
            password=self.config.password,
            min_size=self.config.pool_min_size,
            max_size=self.config.pool_max_size,
            command_timeout=self.config.command_timeout
        )
        
        # Initialize database schema
        await self._create_tables()
        await self._create_hypertables()
        await self._create_indexes()
        await self._setup_compression()
        await self._setup_retention_policies()
        await self._create_continuous_aggregates()
        
        self.logger.info("TimescaleDB initialization completed")
    
    async def _create_tables(self):
        """Create all necessary tables"""
        queries = [
            # Tick data table
            """
            CREATE TABLE IF NOT EXISTS tick_data (
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                price DECIMAL(12,4) NOT NULL,
                volume DECIMAL(12,4) NOT NULL,
                bid_price DECIMAL(12,4),
                ask_price DECIMAL(12,4),
                bid_size DECIMAL(12,4),
                ask_size DECIMAL(12,4),
                trade_type VARCHAR(10),
                exchange VARCHAR(20),
                trade_id VARCHAR(50),
                sequence_number BIGINT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """,
            
            # OHLCV data table
            """
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                open DECIMAL(12,4) NOT NULL,
                high DECIMAL(12,4) NOT NULL,
                low DECIMAL(12,4) NOT NULL,
                close DECIMAL(12,4) NOT NULL,
                volume DECIMAL(12,4) NOT NULL,
                vwap DECIMAL(12,4),
                trades_count INTEGER,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """,
            
            # Order book snapshots
            """
            CREATE TABLE IF NOT EXISTS order_book (
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                level INTEGER NOT NULL,
                bid_price DECIMAL(12,4),
                bid_size DECIMAL(12,4),
                ask_price DECIMAL(12,4),
                ask_size DECIMAL(12,4),
                total_bid_volume DECIMAL(12,4),
                total_ask_volume DECIMAL(12,4),
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """,
            
            # Individual trades
            """
            CREATE TABLE IF NOT EXISTS trades (
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                trade_id VARCHAR(50) NOT NULL,
                price DECIMAL(12,4) NOT NULL,
                volume DECIMAL(12,4) NOT NULL,
                side VARCHAR(10),
                buyer_maker BOOLEAN,
                trade_time TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """,
            
            # Market regime data
            """
            CREATE TABLE IF NOT EXISTS market_regime (
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                regime VARCHAR(20) NOT NULL,
                probability DECIMAL(5,4) NOT NULL,
                duration INTEGER,
                volatility DECIMAL(8,6),
                trend_strength DECIMAL(5,4),
                characteristics JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        ]
        
        async with self.pool.acquire() as conn:
            for query in queries:
                await conn.execute(query)
                
        self.logger.info("Created all tables")
    
    async def _create_hypertables(self):
        """Convert tables to hypertables for time-series optimization"""
        async with self.pool.acquire() as conn:
            for schema_name, schema_info in self.schemas.items():
                table = schema_info['table']
                time_column = schema_info['time_column']
                chunk_interval = schema_info['chunk_interval']
                
                try:
                    # Check if already a hypertable
                    result = await conn.fetchval(
                        "SELECT hypertable_name FROM timescaledb_information.hypertables WHERE hypertable_name = $1",
                        table
                    )
                    
                    if not result:
                        # Create hypertable
                        await conn.execute(f"""
                            SELECT create_hypertable('{table}', '{time_column}', 
                                                   chunk_time_interval => INTERVAL '{chunk_interval}',
                                                   if_not_exists => TRUE);
                        """)
                        self.logger.info(f"Created hypertable: {table}")
                    else:
                        self.logger.info(f"Hypertable already exists: {table}")
                        
                except Exception as e:
                    self.logger.error(f"Error creating hypertable {table}: {e}")
    
    async def _create_indexes(self):
        """Create optimized indexes for queries"""
        indexes = [
            # Tick data indexes
            "CREATE INDEX IF NOT EXISTS idx_tick_data_symbol_time ON tick_data (symbol, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_tick_data_price ON tick_data (price);",
            "CREATE INDEX IF NOT EXISTS idx_tick_data_volume ON tick_data (volume);",
            
            # OHLCV indexes
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe_time ON ohlcv_data (symbol, timeframe, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_close ON ohlcv_data (close);",
            
            # Order book indexes
            "CREATE INDEX IF NOT EXISTS idx_order_book_symbol_time ON order_book (symbol, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_order_book_level ON order_book (level);",
            
            # Trades indexes
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades (symbol, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_trades_trade_id ON trades (trade_id);",
            
            # Market regime indexes
            "CREATE INDEX IF NOT EXISTS idx_market_regime_symbol_time ON market_regime (symbol, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_market_regime_regime ON market_regime (regime);"
        ]
        
        async with self.pool.acquire() as conn:
            for index_query in indexes:
                try:
                    await conn.execute(index_query)
                except Exception as e:
                    self.logger.warning(f"Index creation warning: {e}")
                    
        self.logger.info("Created database indexes")
    
    async def _setup_compression(self):
        """Setup compression policies for data efficiency"""
        async with self.pool.acquire() as conn:
            for schema_name, schema_info in self.schemas.items():
                table = schema_info['table']
                compression_after = schema_info['compression_after']
                
                try:
                    # Add compression policy
                    await conn.execute(f"""
                        SELECT add_compression_policy('{table}', INTERVAL '{compression_after}');
                    """)
                    self.logger.info(f"Added compression policy for {table}")
                except Exception as e:
                    self.logger.warning(f"Compression policy warning for {table}: {e}")
    
    async def _setup_retention_policies(self):
        """Setup data retention policies"""
        async with self.pool.acquire() as conn:
            for schema_name, schema_info in self.schemas.items():
                table = schema_info['table']
                retention_period = schema_info['retention_period']
                
                try:
                    # Add retention policy
                    await conn.execute(f"""
                        SELECT add_retention_policy('{table}', INTERVAL '{retention_period}');
                    """)
                    self.logger.info(f"Added retention policy for {table}")
                except Exception as e:
                    self.logger.warning(f"Retention policy warning for {table}: {e}")
    
    async def _create_continuous_aggregates(self):
        """Create continuous aggregates for real-time analytics"""
        aggregates = [
            # 1-minute OHLCV from tick data
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1min
            WITH (timescaledb.continuous) AS
            SELECT time_bucket('1 minute', timestamp) AS timestamp,
                   symbol,
                   first(price, timestamp) AS open,
                   max(price) AS high,
                   min(price) AS low,
                   last(price, timestamp) AS close,
                   sum(volume) AS volume,
                   avg(price) AS vwap,
                   count(*) AS trades_count
            FROM tick_data
            GROUP BY time_bucket('1 minute', timestamp), symbol;
            """,
            
            # 5-minute OHLCV
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_5min
            WITH (timescaledb.continuous) AS
            SELECT time_bucket('5 minutes', timestamp) AS timestamp,
                   symbol,
                   first(open, timestamp) AS open,
                   max(high) AS high,
                   min(low) AS low,
                   last(close, timestamp) AS close,
                   sum(volume) AS volume,
                   avg(vwap) AS vwap,
                   sum(trades_count) AS trades_count
            FROM ohlcv_1min
            GROUP BY time_bucket('5 minutes', timestamp), symbol;
            """,
            
            # Real-time market statistics
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS market_stats_1min
            WITH (timescaledb.continuous) AS
            SELECT time_bucket('1 minute', timestamp) AS timestamp,
                   symbol,
                   avg(price) AS avg_price,
                   stddev(price) AS price_volatility,
                   sum(volume) AS total_volume,
                   max(price) - min(price) AS price_range,
                   count(*) AS tick_count
            FROM tick_data
            GROUP BY time_bucket('1 minute', timestamp), symbol;
            """
        ]
        
        async with self.pool.acquire() as conn:
            for aggregate_query in aggregates:
                try:
                    await conn.execute(aggregate_query)
                    self.logger.info("Created continuous aggregate")
                except Exception as e:
                    self.logger.warning(f"Continuous aggregate warning: {e}")
        
        # Add refresh policies
        refresh_policies = [
            "SELECT add_continuous_aggregate_policy('ohlcv_1min', start_offset => INTERVAL '1 hour', end_offset => INTERVAL '1 minute', schedule_interval => INTERVAL '1 minute');",
            "SELECT add_continuous_aggregate_policy('ohlcv_5min', start_offset => INTERVAL '1 hour', end_offset => INTERVAL '5 minutes', schedule_interval => INTERVAL '5 minutes');",
            "SELECT add_continuous_aggregate_policy('market_stats_1min', start_offset => INTERVAL '1 hour', end_offset => INTERVAL '1 minute', schedule_interval => INTERVAL '1 minute');"
        ]
        
        async with self.pool.acquire() as conn:
            for policy in refresh_policies:
                try:
                    await conn.execute(policy)
                except Exception as e:
                    self.logger.warning(f"Refresh policy warning: {e}")
    
    async def insert_tick_data(self, tick_data: List[TickData]) -> int:
        """Insert tick data efficiently using batch operations"""
        if not tick_data:
            return 0
            
        # Prepare data for insertion
        records = []
        for tick in tick_data:
            records.append((
                tick.timestamp,
                tick.symbol,
                float(tick.price),
                float(tick.volume),
                float(tick.bid_price) if tick.bid_price else None,
                float(tick.ask_price) if tick.ask_price else None,
                float(tick.bid_size) if tick.bid_size else None,
                float(tick.ask_size) if tick.ask_size else None,
                tick.trade_type,
                tick.exchange,
                tick.trade_id,
                tick.sequence_number
            ))
        
        query = """
            INSERT INTO tick_data (timestamp, symbol, price, volume, bid_price, ask_price, 
                                 bid_size, ask_size, trade_type, exchange, trade_id, sequence_number)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """
        
        async with self.pool.acquire() as conn:
            await conn.executemany(query, records)
            
        self.logger.debug(f"Inserted {len(records)} tick records")
        return len(records)
    
    async def insert_ohlcv_data(self, ohlcv_data: pd.DataFrame) -> int:
        """Insert OHLCV data efficiently"""
        if ohlcv_data.empty:
            return 0
            
        records = []
        for _, row in ohlcv_data.iterrows():
            records.append((
                row.name if isinstance(row.name, datetime) else datetime.fromisoformat(str(row.name)),
                row.get('symbol', 'UNKNOWN'),
                row.get('timeframe', '5min'),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume']),
                float(row.get('vwap', 0)),
                int(row.get('trades_count', 0))
            ))
        
        query = """
            INSERT INTO ohlcv_data (timestamp, symbol, timeframe, open, high, low, close, 
                                  volume, vwap, trades_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (timestamp, symbol, timeframe) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                vwap = EXCLUDED.vwap,
                trades_count = EXCLUDED.trades_count
        """
        
        async with self.pool.acquire() as conn:
            await conn.executemany(query, records)
            
        self.logger.debug(f"Inserted {len(records)} OHLCV records")
        return len(records)
    
    async def get_tick_data(self, 
                          symbol: str, 
                          start_time: datetime, 
                          end_time: datetime,
                          limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve tick data for analysis"""
        query = """
            SELECT timestamp, price, volume, bid_price, ask_price, bid_size, ask_size,
                   trade_type, exchange, trade_id, sequence_number
            FROM tick_data
            WHERE symbol = $1 AND timestamp >= $2 AND timestamp <= $3
            ORDER BY timestamp DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, start_time, end_time)
            
        if not rows:
            return pd.DataFrame()
            
        df = pd.DataFrame([dict(row) for row in rows])
        df.set_index('timestamp', inplace=True)
        return df
    
    async def get_ohlcv_data(self,
                           symbol: str,
                           timeframe: str,
                           start_time: datetime,
                           end_time: datetime,
                           source: str = 'ohlcv_data') -> pd.DataFrame:
        """Retrieve OHLCV data from main table or continuous aggregates"""
        
        # Choose appropriate source
        if source == 'continuous_1min':
            table = 'ohlcv_1min'
        elif source == 'continuous_5min':
            table = 'ohlcv_5min'
        else:
            table = 'ohlcv_data'
        
        if table == 'ohlcv_data':
            query = f"""
                SELECT timestamp, open, high, low, close, volume, vwap, trades_count
                FROM {table}
                WHERE symbol = $1 AND timeframe = $2 AND timestamp >= $3 AND timestamp <= $4
                ORDER BY timestamp
            """
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, symbol, timeframe, start_time, end_time)
        else:
            query = f"""
                SELECT timestamp, open, high, low, close, volume, vwap, trades_count
                FROM {table}
                WHERE symbol = $1 AND timestamp >= $2 AND timestamp <= $3
                ORDER BY timestamp
            """
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, symbol, start_time, end_time)
        
        if not rows:
            return pd.DataFrame()
            
        df = pd.DataFrame([dict(row) for row in rows])
        df.set_index('timestamp', inplace=True)
        return df
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol"""
        query = """
            SELECT price FROM tick_data
            WHERE symbol = $1
            ORDER BY timestamp DESC
            LIMIT 1
        """
        
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(query, symbol)
            
        return float(result) if result else None
    
    async def get_market_stats(self, 
                             symbol: str, 
                             minutes: int = 60) -> Dict[str, Any]:
        """Get real-time market statistics"""
        since = datetime.now() - timedelta(minutes=minutes)
        
        query = """
            SELECT avg_price, price_volatility, total_volume, price_range, tick_count
            FROM market_stats_1min
            WHERE symbol = $1 AND timestamp >= $2
            ORDER BY timestamp DESC
            LIMIT $3
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, since, minutes)
            
        if not rows:
            return {}
            
        # Aggregate statistics
        df = pd.DataFrame([dict(row) for row in rows])
        
        return {
            'avg_price': float(df['avg_price'].mean()),
            'volatility': float(df['price_volatility'].mean()),
            'total_volume': float(df['total_volume'].sum()),
            'price_range': float(df['price_range'].max()),
            'tick_count': int(df['tick_count'].sum()),
            'minutes_covered': len(df)
        }
    
    async def cleanup_old_data(self):
        """Manual cleanup of old data (beyond retention policies)"""
        queries = [
            "SELECT drop_chunks('tick_data', INTERVAL '3 years');",
            "SELECT drop_chunks('order_book', INTERVAL '60 days');",
            "SELECT drop_chunks('trades', INTERVAL '2 years');"
        ]
        
        async with self.pool.acquire() as conn:
            for query in queries:
                try:
                    result = await conn.execute(query)
                    self.logger.info(f"Cleanup result: {result}")
                except Exception as e:
                    self.logger.warning(f"Cleanup warning: {e}")
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions"""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn
    
    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
            self.logger.info("Closed TimescaleDB connections")
    
    def __del__(self):
        """Cleanup on deletion"""
        if self.pool:
            try:
                asyncio.create_task(self.close())
            except:
                pass