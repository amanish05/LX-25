"""
Pytest Configuration and Shared Fixtures
Provides common test fixtures and utilities for all tests
"""

import os
import sys
import asyncio
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime, timedelta
import tempfile
import json
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.database import DatabaseManager, Base
from src.core.bot_manager import BotManager
from src.integrations.openalgo_client import OpenAlgoClient
from src.config import ConfigManager, get_config_manager
from src.bots.base_bot import BotState
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession


# Configure pytest for asyncio
pytest_plugins = ('pytest_asyncio',)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def test_db():
    """Create a test database for each test"""
    # Create temporary database
    db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db_path = db_file.name
    db_file.close()
    
    # Create engine and tables
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield db_path
    
    # Cleanup
    await engine.dispose()
    os.unlink(db_path)


@pytest.fixture
def test_config(test_db):
    """Create test configuration"""
    config = {
        "system": {
            "name": "Test Trading Bot",
            "version": "1.0.0",
            "environment": "test",
            "total_capital": 1000000,
            "emergency_reserve": 100000,
            "thread_pool_size": 2,
            "auto_restart_bots": True
        },
        "domains": {
            "openalgo_api_host": "http://127.0.0.1",
            "openalgo_api_port": 5000,
            "openalgo_api_version": "v1",
            "websocket_host": "ws://127.0.0.1",
            "websocket_port": 8765,
            "database_type": "sqlite",
            "database_connection": {"path": test_db}
        },
        "api": {
            "host": "127.0.0.1",
            "port": 8081,  # Different port for testing
            "cors_origins": ["http://localhost:3000"],
            "docs_enabled": True
        },
        "logging": {
            "level": "DEBUG",
            "file": "logs/test_trading_bot.log",
            "console_output": False
        },
        "monitoring": {
            "health_check_interval": 5,
            "position_sync_interval": 5,
            "metrics_update_interval": 2
        },
        "execution": {
            "order_timeout_seconds": 5,
            "retry_attempts": 2,
            "slippage_tolerance": 0.5,
            "partial_fill_wait": 5,
            "market_order_protection": True,
            "max_concurrent_orders": 5
        },
        "market_hours": {
            "start": "09:15",
            "end": "15:30",
            "timezone": "Asia/Kolkata",
            "pre_market_start": "09:00",
            "post_market_end": "15:30"
        }
    }
    return config


@pytest.fixture
def test_config_manager(test_config, tmp_path):
    """Create test configuration manager"""
    # Create config directory
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Write test config
    config_file = config_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(test_config, f)
    
    # Create and return config manager
    config_manager = ConfigManager(str(config_dir))
    return config_manager


@pytest.fixture
def mock_openalgo_client():
    """Mock OpenAlgo client with common responses"""
    client = Mock(spec=OpenAlgoClient)
    
    # Async methods
    client.connect_websocket = AsyncMock()
    client.disconnect_websocket = AsyncMock()
    client.subscribe_market_data = AsyncMock()
    client.unsubscribe_market_data = AsyncMock()
    
    # API methods
    client.get_funds = AsyncMock(return_value={
        "available_balance": 500000,
        "used_margin": 100000,
        "total_balance": 600000
    })
    
    client.get_positions = AsyncMock(return_value=[])
    
    client.place_order = AsyncMock(return_value={
        "status": "success",
        "orderid": "TEST123456",
        "message": "Order placed successfully"
    })
    
    client.cancel_order = AsyncMock(return_value={
        "status": "success",
        "message": "Order cancelled"
    })
    
    client.get_quote = AsyncMock(return_value={
        "ltp": 20000,
        "open": 19950,
        "high": 20100,
        "low": 19900,
        "close": 20000,
        "volume": 1000000
    })
    
    client.get_option_chain = AsyncMock(return_value={
        "2024-02-29": {
            "strikes": {
                "20000": {
                    "CE": {"ltp": 150, "iv": 20, "oi": 50000, "volume": 5000},
                    "PE": {"ltp": 140, "iv": 19, "oi": 45000, "volume": 4500}
                },
                "19900": {
                    "CE": {"ltp": 200, "iv": 19, "oi": 40000, "volume": 4000},
                    "PE": {"ltp": 100, "iv": 18, "oi": 35000, "volume": 3500}
                },
                "20100": {
                    "CE": {"ltp": 100, "iv": 21, "oi": 45000, "volume": 4500},
                    "PE": {"ltp": 190, "iv": 20, "oi": 40000, "volume": 4000}
                }
            }
        }
    })
    
    # WebSocket data handler
    client.on_market_data = None
    
    return client


@pytest.fixture
async def mock_db_manager(test_config):
    """Mock database manager"""
    db_manager = Mock(spec=DatabaseManager)
    
    # Async methods
    db_manager.init_database = AsyncMock()
    db_manager.close = AsyncMock()
    db_manager.get_session = AsyncMock()
    
    # Bot capital methods
    db_manager.init_bot_capital = AsyncMock()
    db_manager.update_bot_capital = AsyncMock()
    db_manager.get_bot_capital = AsyncMock(return_value={
        "initial": 200000,
        "current": 195000,
        "available": 150000,
        "locked": 45000
    })
    
    # Position methods
    db_manager.create_position = AsyncMock(return_value=1)
    db_manager.update_position = AsyncMock()
    db_manager.close_position = AsyncMock()
    db_manager.get_open_positions = AsyncMock(return_value=[])
    db_manager.get_position_by_id = AsyncMock(return_value=None)
    
    # Trade methods
    db_manager.create_trade = AsyncMock(return_value=1)
    db_manager.get_trades = AsyncMock(return_value=[])
    
    # Signal methods
    db_manager.save_signal = AsyncMock(return_value=1)
    db_manager.mark_signal_executed = AsyncMock()
    db_manager.get_signals = AsyncMock(return_value=[])
    
    # Performance methods
    db_manager.update_bot_performance = AsyncMock()
    db_manager.get_bot_performance = AsyncMock(return_value={
        "total_trades": 0,
        "winning_trades": 0,
        "total_pnl": 0,
        "max_drawdown": 0
    })
    
    # Cache methods
    db_manager.cache_market_data = AsyncMock()
    db_manager.get_cached_market_data = AsyncMock(return_value=None)
    db_manager.cleanup_expired_cache = AsyncMock()
    
    return db_manager


@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return {
        "NIFTY": {
            "ltp": 20000,
            "open": 19950,
            "high": 20100,
            "low": 19900,
            "close": 20000,
            "volume": 1000000,
            "oi": 5000000,
            "timestamp": datetime.now().isoformat()
        },
        "BANKNIFTY": {
            "ltp": 45000,
            "open": 44900,
            "high": 45200,
            "low": 44800,
            "close": 45000,
            "volume": 500000,
            "oi": 2500000,
            "timestamp": datetime.now().isoformat()
        }
    }


@pytest.fixture
def sample_option_chain():
    """Sample option chain data for testing"""
    expiry = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    
    return {
        "NIFTY": {
            expiry: {
                "strikes": {
                    "20000": {
                        "CE": {"ltp": 150, "iv": 20, "oi": 50000, "volume": 5000, "bid": 148, "ask": 152},
                        "PE": {"ltp": 140, "iv": 19, "oi": 45000, "volume": 4500, "bid": 138, "ask": 142}
                    },
                    "19900": {
                        "CE": {"ltp": 200, "iv": 19, "oi": 40000, "volume": 4000, "bid": 198, "ask": 202},
                        "PE": {"ltp": 100, "iv": 18, "oi": 35000, "volume": 3500, "bid": 98, "ask": 102}
                    },
                    "20100": {
                        "CE": {"ltp": 100, "iv": 21, "oi": 45000, "volume": 4500, "bid": 98, "ask": 102},
                        "PE": {"ltp": 190, "iv": 20, "oi": 40000, "volume": 4000, "bid": 188, "ask": 192}
                    }
                }
            }
        }
    }


@pytest.fixture
def sample_position():
    """Sample position data for testing"""
    return {
        "id": 1,
        "bot_name": "TestBot",
        "symbol": "NIFTY",
        "exchange": "NFO",
        "position_type": "SHORT",
        "quantity": 50,
        "entry_price": 150,
        "current_price": 140,
        "entry_time": datetime.now() - timedelta(hours=1),
        "pnl": 500,
        "status": "OPEN"
    }


@pytest.fixture
def sample_bot_config():
    """Sample bot configuration for testing"""
    return {
        "name": "TestShortStraddleBot",
        "bot_type": "short_straddle",
        "enabled": True,
        "capital": 200000,
        "max_positions": 2,
        "entry": {
            "iv_rank_min": 70,
            "dte_min": 20,
            "dte_max": 45
        },
        "exit": {
            "profit_target_pct": 50,
            "stop_loss_multiplier": 2,
            "time_exit_dte": 21
        },
        "risk": {
            "max_position_size": 40000,
            "max_daily_loss": 5000
        }
    }


@pytest.fixture
async def mock_bot_manager(test_config_manager, mock_db_manager, mock_openalgo_client):
    """Mock bot manager"""
    bot_manager = BotManager(test_config_manager)
    
    # Replace real components with mocks
    bot_manager.db_manager = mock_db_manager
    bot_manager.openalgo_client = mock_openalgo_client
    
    return bot_manager


# Utility functions for tests

def create_mock_bot(name: str, bot_type: str, state: BotState = BotState.INITIALIZED):
    """Create a mock bot instance"""
    bot = Mock()
    bot.name = name
    bot.bot_type = bot_type
    bot.state = state
    bot.start = AsyncMock()
    bot.stop = AsyncMock()
    bot.pause = AsyncMock()
    bot.resume = AsyncMock()
    bot.get_status = Mock(return_value={
        "name": name,
        "type": bot_type,
        "state": state.value,
        "capital": {
            "initial": 200000,
            "current": 195000,
            "available": 150000,
            "locked": 45000
        },
        "positions": 1,
        "performance": {
            "total_trades": 10,
            "winning_trades": 7,
            "total_pnl": 5000,
            "win_rate": 70.0
        }
    })
    return bot


async def wait_for_condition(condition_func, timeout: float = 5.0, interval: float = 0.1):
    """Wait for a condition to become true"""
    start_time = asyncio.get_event_loop().time()
    
    while not condition_func():
        if asyncio.get_event_loop().time() - start_time > timeout:
            raise TimeoutError(f"Condition not met within {timeout} seconds")
        await asyncio.sleep(interval)


def assert_api_response(response, expected_status: int = 200):
    """Assert API response is valid"""
    assert response.status_code == expected_status
    if expected_status == 200:
        data = response.json()
        assert data is not None
        return data
    return None


# Market hours helpers

def set_market_hours(is_open: bool = True):
    """Set current time to be within or outside market hours"""
    from unittest.mock import patch
    from datetime import time
    
    if is_open:
        # Set to 10:00 AM (market open)
        mock_time = datetime.now().replace(hour=10, minute=0, second=0)
    else:
        # Set to 4:00 PM (market closed)
        mock_time = datetime.now().replace(hour=16, minute=0, second=0)
    
    return patch('datetime.datetime.now', return_value=mock_time)


# WebSocket test helpers

class MockWebSocket:
    """Mock WebSocket for testing"""
    
    def __init__(self):
        self.messages = []
        self.is_connected = False
        self.send = AsyncMock()
        self.recv = AsyncMock()
        self.close = AsyncMock()
        
    async def connect(self):
        self.is_connected = True
        
    async def disconnect(self):
        self.is_connected = False
        await self.close()
        
    def add_message(self, message: Dict[str, Any]):
        """Add a message to be received"""
        self.messages.append(json.dumps(message))
        self.recv.side_effect = [self.messages.pop(0) for _ in range(len(self.messages))]


# Performance test helpers

def measure_async_performance(func):
    """Decorator to measure async function performance"""
    async def wrapper(*args, **kwargs):
        start_time = asyncio.get_event_loop().time()
        result = await func(*args, **kwargs)
        end_time = asyncio.get_event_loop().time()
        
        print(f"{func.__name__} took {end_time - start_time:.3f} seconds")
        return result
    
    return wrapper