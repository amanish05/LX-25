"""
Common Database Configuration Module
Supports both SQLite and PostgreSQL with environment-based configuration
"""

import os
from typing import Dict, Any, Optional
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """
    Common database configuration for both OpenAlgo and Automated Trading Bot
    Supports SQLite and PostgreSQL with optimized settings
    """
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv('DATABASE_URL', 'sqlite:///db/openalgo.db')
        self.parsed_url = urlparse(self.database_url)
        self.db_type = self._detect_db_type()
        
    def _detect_db_type(self) -> str:
        """Detect database type from URL"""
        if 'sqlite' in self.database_url:
            return 'sqlite'
        elif 'postgresql' in self.database_url or 'postgres' in self.database_url:
            return 'postgresql'
        else:
            raise ValueError(f"Unsupported database type in URL: {self.database_url}")
    
    def get_engine_kwargs(self, async_mode: bool = False) -> Dict[str, Any]:
        """
        Get SQLAlchemy engine kwargs optimized for the database type
        
        Args:
            async_mode: Whether to use async drivers
            
        Returns:
            Dictionary of engine configuration parameters
        """
        if self.db_type == 'sqlite':
            return self._get_sqlite_config(async_mode)
        elif self.db_type == 'postgresql':
            return self._get_postgresql_config(async_mode)
    
    def _get_sqlite_config(self, async_mode: bool) -> Dict[str, Any]:
        """SQLite-specific configuration"""
        config = {
            'pool_pre_ping': True,
            'pool_recycle': 3600,
        }
        
        if async_mode:
            # For automated-trading-bot async operations
            config.update({
                'pool_size': 5,
                'max_overflow': 10,
            })
        else:
            # For OpenAlgo sync operations
            config.update({
                'pool_size': 50,
                'max_overflow': 100,
                'connect_args': {
                    'check_same_thread': False
                }
            })
        
        return config
    
    def _get_postgresql_config(self, async_mode: bool) -> Dict[str, Any]:
        """PostgreSQL-specific configuration with optimizations"""
        config = {
            'pool_pre_ping': True,
            'pool_recycle': 3600,
            'echo_pool': os.getenv('DEBUG', 'False').lower() == 'true',
        }
        
        if async_mode:
            # For automated-trading-bot async operations
            config.update({
                'pool_size': 20,  # Higher for PostgreSQL
                'max_overflow': 40,
                'pool_timeout': 30,
                'connect_args': {
                    'server_settings': {
                        'application_name': 'automated_trading_bot',
                        'jit': 'off'
                    },
                    'timeout': 10,
                    'command_timeout': 10,
                }
            })
        else:
            # For OpenAlgo sync operations
            config.update({
                'pool_size': 50,
                'max_overflow': 100,
                'pool_timeout': 30,
                'connect_args': {
                    'application_name': 'openalgo',
                    'options': '-c statement_timeout=30000',  # 30 seconds
                    'keepalives': 1,
                    'keepalives_idle': 30,
                    'keepalives_interval': 10,
                    'keepalives_count': 5,
                }
            })
        
        return config
    
    def get_async_url(self) -> str:
        """Convert sync database URL to async driver URL"""
        if self.db_type == 'sqlite':
            # Convert sqlite:/// to sqlite+aiosqlite:///
            return self.database_url.replace('sqlite:///', 'sqlite+aiosqlite:///')
        elif self.db_type == 'postgresql':
            # Convert postgresql:// to postgresql+asyncpg://
            url = self.database_url
            if url.startswith('postgresql://'):
                return url.replace('postgresql://', 'postgresql+asyncpg://')
            elif url.startswith('postgres://'):
                return url.replace('postgres://', 'postgresql+asyncpg://')
            return url
        return self.database_url
    
    def get_sync_url(self) -> str:
        """Get synchronous database URL"""
        if self.db_type == 'postgresql' and 'asyncpg' in self.database_url:
            # Remove async driver
            return self.database_url.replace('+asyncpg', '')
        return self.database_url
    
    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite"""
        return self.db_type == 'sqlite'
    
    @property
    def is_postgresql(self) -> bool:
        """Check if using PostgreSQL"""
        return self.db_type == 'postgresql'
    
    def get_database_specific_types(self) -> Dict[str, Any]:
        """
        Get database-specific column types for migrations
        
        Returns:
            Dictionary mapping generic types to database-specific types
        """
        if self.is_postgresql:
            from sqlalchemy.dialects.postgresql import JSONB, UUID
            return {
                'json': JSONB,
                'uuid': UUID(as_uuid=True),
            }
        else:
            from sqlalchemy import JSON, String
            return {
                'json': JSON,
                'uuid': String(36),
            }
    
    def validate_connection(self) -> bool:
        """Validate database connection parameters"""
        try:
            if self.is_postgresql:
                # Check if we can parse PostgreSQL URL
                if '@' not in self.database_url:
                    logger.error("PostgreSQL URL missing credentials")
                    return False
                if '/' not in self.database_url.split('@')[1]:
                    logger.error("PostgreSQL URL missing database name")
                    return False
            return True
        except Exception as e:
            logger.error(f"Invalid database URL: {e}")
            return False


def get_database_config(database_url: Optional[str] = None) -> DatabaseConfig:
    """Factory function to get database configuration"""
    return DatabaseConfig(database_url)


# Migration helpers
def get_postgresql_migration_sql() -> Dict[str, str]:
    """Get PostgreSQL-specific SQL for migrations"""
    return {
        'create_extensions': """
            CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
            CREATE EXTENSION IF NOT EXISTS "pgcrypto";
        """,
        'create_indexes': """
            -- Optimize for time-series queries
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON bot_trades(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_positions_updated ON bot_positions(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON bot_performance(timestamp DESC);
            
            -- Optimize for bot queries
            CREATE INDEX IF NOT EXISTS idx_trades_bot_symbol ON bot_trades(bot_name, symbol);
            CREATE INDEX IF NOT EXISTS idx_positions_bot_symbol ON bot_positions(bot_name, symbol);
            
            -- Optimize for JSONB queries (PostgreSQL specific)
            CREATE INDEX IF NOT EXISTS idx_positions_metadata ON bot_positions USING GIN (metadata);
            CREATE INDEX IF NOT EXISTS idx_trades_metadata ON bot_trades USING GIN (metadata);
        """,
        'optimize_settings': """
            -- Optimize for financial data workloads
            ALTER TABLE bot_trades SET (autovacuum_vacuum_scale_factor = 0.01);
            ALTER TABLE bot_positions SET (autovacuum_vacuum_scale_factor = 0.01);
            ALTER TABLE market_data_cache SET (autovacuum_vacuum_scale_factor = 0.05);
        """
    }