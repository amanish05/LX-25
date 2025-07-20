"""
Application Configuration
Central configuration management for the entire application
"""

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from .constants import TIME_CONSTANTS, SYSTEM_CONSTANTS, RISK_CONSTANTS, DATABASE_CONSTANTS, ORDER_CONSTANTS, TRADING_CONSTANTS, INDICATOR_CONSTANTS


@dataclass
class DomainConfig:
    """Domain configuration"""
    openalgo_api_host: str = "http://127.0.0.1"
    openalgo_api_port: int = 5000
    openalgo_api_version: str = "v1"
    websocket_host: str = "ws://127.0.0.1"
    websocket_port: int = 8765
    database_type: str = "postgresql"  # PostgreSQL for all environments
    database_connection: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.database_connection is None:
            # Default empty - must be provided via DATABASE_URL env var
            self.database_connection = {}
    
    @property
    def openalgo_api_url(self) -> str:
        return f"{self.openalgo_api_host}:{self.openalgo_api_port}/api/{self.openalgo_api_version}"
    
    @property
    def websocket_url(self) -> str:
        return f"{self.websocket_host}:{self.websocket_port}"
    
    @property
    def database_url(self) -> str:
        if self.database_type == "postgresql":
            conn = self.database_connection
            # If we have a direct URL from env var, use it
            if "url" in conn:
                return conn["url"]
            # Otherwise DATABASE_URL env var is required
            raise ValueError("DATABASE_URL environment variable must be set for PostgreSQL")
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")


@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: List[str] = None
    docs_enabled: bool = True
    title: str = "Automated Trading Bot API"
    version: str = "1.0.0"
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:3000"]


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = SYSTEM_CONSTANTS.LOG_LEVEL_INFO
    file: str = "logs/trading_bot.log"
    max_size_mb: int = 10
    backup_count: int = 5
    format: str = "json"  # json or text
    console_output: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    health_check_interval: int = TIME_CONSTANTS.HEALTH_CHECK_INTERVAL
    position_sync_interval: int = TIME_CONSTANTS.POSITION_SYNC_INTERVAL
    metrics_update_interval: int = TIME_CONSTANTS.METRICS_UPDATE_INTERVAL
    performance_calculation_interval: int = 300
    alert_channels: List[str] = None
    
    def __post_init__(self):
        if self.alert_channels is None:
            self.alert_channels = ["console", "log"]


@dataclass
class ExecutionConfig:
    """Trade execution configuration"""
    order_timeout_seconds: int = TIME_CONSTANTS.ORDER_TIMEOUT_SECONDS
    retry_attempts: int = RISK_CONSTANTS.MAX_ORDER_RETRY_ATTEMPTS
    slippage_tolerance: float = RISK_CONSTANTS.SLIPPAGE_TOLERANCE_PERCENT
    partial_fill_wait: int = RISK_CONSTANTS.PARTIAL_FILL_WAIT_SECONDS
    market_order_protection: bool = True
    max_concurrent_orders: int = SYSTEM_CONSTANTS.MAX_CONCURRENT_ORDERS


@dataclass
class MarketHoursConfig:
    """Market hours configuration"""
    start: str = TIME_CONSTANTS.MARKET_OPEN_TIME
    end: str = TIME_CONSTANTS.MARKET_CLOSE_TIME
    timezone: str = TIME_CONSTANTS.TIMEZONE
    pre_market_start: str = TIME_CONSTANTS.PRE_MARKET_OPEN
    post_market_end: str = TIME_CONSTANTS.POST_MARKET_CLOSE
    
    def is_market_open(self, current_time) -> bool:
        """Check if market is currently open"""
        # Implementation would check current time against market hours
        pass


@dataclass
class SystemConfig:
    """System-level configuration"""
    name: str = "Automated Trading Bot"
    version: str = "1.0.0"
    environment: str = SYSTEM_CONSTANTS.ENV_DEVELOPMENT
    total_capital: float = 1000000
    emergency_reserve: float = 100000
    thread_pool_size: int = SYSTEM_CONSTANTS.THREAD_POOL_SIZE
    auto_restart_bots: bool = True
    
    @property
    def available_capital(self) -> float:
        return self.total_capital - self.emergency_reserve
    
    @property
    def is_production(self) -> bool:
        return self.environment == SYSTEM_CONSTANTS.ENV_PRODUCTION
    
    @property
    def is_development(self) -> bool:
        return self.environment == SYSTEM_CONSTANTS.ENV_DEVELOPMENT


class AppConfig:
    """Main application configuration manager"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or SYSTEM_CONSTANTS.DEFAULT_CONFIG_FILE
        
        # Initialize sub-configurations
        self.system = SystemConfig()
        self.domains = DomainConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        self.monitoring = MonitoringConfig()
        self.execution = ExecutionConfig()
        self.market_hours = MarketHoursConfig()
        
        # Load from file if exists
        if os.path.exists(self.config_file):
            self.load_from_file()
        
        # Apply environment overrides
        self._apply_env_overrides()
    
    def load_from_file(self):
        """Load configuration from JSON file"""
        with open(self.config_file, 'r') as f:
            data = json.load(f)
        
        # Update configurations
        if 'system' in data:
            self.system = SystemConfig(**data['system'])
        
        if 'domains' in data:
            self.domains = DomainConfig(**data['domains'])
        
        if 'api' in data:
            self.api = APIConfig(**data['api'])
        
        if 'logging' in data:
            self.logging = LoggingConfig(**data['logging'])
        
        if 'monitoring' in data:
            self.monitoring = MonitoringConfig(**data['monitoring'])
        
        if 'execution' in data:
            self.execution = ExecutionConfig(**data['execution'])
        
        if 'market_hours' in data:
            self.market_hours = MarketHoursConfig(**data['market_hours'])
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # System overrides
        if env := os.getenv("ENVIRONMENT"):
            self.system.environment = env
        
        if capital := os.getenv("TOTAL_CAPITAL"):
            self.system.total_capital = float(capital)
        
        # Domain overrides
        if api_host := os.getenv("OPENALGO_API_HOST"):
            self.domains.openalgo_api_host = api_host
        
        if api_port := os.getenv("OPENALGO_API_PORT"):
            self.domains.openalgo_api_port = int(api_port)
        
        if ws_host := os.getenv("WEBSOCKET_HOST"):
            self.domains.websocket_host = ws_host
        
        if ws_port := os.getenv("WEBSOCKET_PORT"):
            self.domains.websocket_port = int(ws_port)
        
        # API overrides
        if api_port := os.getenv("API_PORT"):
            self.api.port = int(api_port)
        
        # Database override
        if db_url := os.getenv("DATABASE_URL"):
            # Parse database URL and update configuration
            if 'postgresql' in db_url or 'postgres' in db_url:
                self.domains.database_type = "postgresql"
                # For PostgreSQL, we'll use the URL directly
                self.domains.database_connection = {"url": db_url}
        
        # Logging overrides
        if log_level := os.getenv("LOG_LEVEL"):
            self.logging.level = log_level
    
    def save_to_file(self, file_path: str = None):
        """Save configuration to JSON file"""
        file_path = file_path or self.config_file
        
        config_dict = {
            'system': asdict(self.system),
            'domains': asdict(self.domains),
            'api': asdict(self.api),
            'logging': asdict(self.logging),
            'monitoring': asdict(self.monitoring),
            'execution': asdict(self.execution),
            'market_hours': asdict(self.market_hours)
        }
        
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Validate system
        if self.system.total_capital <= 0:
            errors.append("Total capital must be positive")
        
        if self.system.emergency_reserve >= self.system.total_capital:
            errors.append("Emergency reserve must be less than total capital")
        
        # Validate domains
        if not self.domains.openalgo_api_host:
            errors.append("OpenAlgo API host is required")
        
        # Validate API
        if self.api.port <= 0 or self.api.port > 65535:
            errors.append("API port must be between 1 and 65535")
        
        # Validate logging
        valid_log_levels = [SYSTEM_CONSTANTS.LOG_LEVEL_DEBUG, SYSTEM_CONSTANTS.LOG_LEVEL_INFO, 
                           SYSTEM_CONSTANTS.LOG_LEVEL_WARNING, SYSTEM_CONSTANTS.LOG_LEVEL_ERROR, 
                           SYSTEM_CONSTANTS.LOG_LEVEL_CRITICAL]
        if self.logging.level not in valid_log_levels:
            errors.append(f"Invalid log level: {self.logging.level}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True
    
    def get_bot_config_path(self) -> str:
        """Get path to bot-specific configurations"""
        return os.path.join(os.path.dirname(self.config_file), "bots")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'system': asdict(self.system),
            'domains': {
                'openalgo_api': {
                    'host': self.domains.openalgo_api_host,
                    'port': self.domains.openalgo_api_port,
                    'version': self.domains.openalgo_api_version,
                    'url': self.domains.openalgo_api_url
                },
                'websocket': {
                    'host': self.domains.websocket_host,
                    'port': self.domains.websocket_port,
                    'url': self.domains.websocket_url
                },
                'database': {
                    'type': self.domains.database_type,
                    'connection': self.domains.database_connection,
                    'url': self.domains.database_url
                }
            },
            'api': asdict(self.api),
            'logging': asdict(self.logging),
            'monitoring': asdict(self.monitoring),
            'execution': asdict(self.execution),
            'market_hours': asdict(self.market_hours),
            'constants': {
                'time': asdict(TIME_CONSTANTS),
                'trading': asdict(TRADING_CONSTANTS),
                'risk': asdict(RISK_CONSTANTS),
                'indicators': asdict(INDICATOR_CONSTANTS)
            }
        }
    
    def get_api_key(self) -> Optional[str]:
        """Get API key from environment or config"""
        return os.getenv("OPENALGO_API_KEY", "")