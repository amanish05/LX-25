"""
Settings Module
User-modifiable settings and preferences
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path


@dataclass
class NotificationSettings:
    """Notification preferences"""
    enabled: bool = True
    
    # Channels
    console: bool = True
    log_file: bool = True
    email: bool = False
    telegram: bool = False
    webhook: bool = False
    
    # Email settings
    email_recipients: List[str] = field(default_factory=list)
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_smtp_user: str = ""
    email_smtp_password: str = ""  # Should be stored securely
    
    # Telegram settings
    telegram_bot_token: str = ""  # Should be stored securely
    telegram_chat_ids: List[str] = field(default_factory=list)
    
    # Webhook settings
    webhook_urls: List[str] = field(default_factory=list)
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    
    # Notification levels
    notify_on_trade_entry: bool = True
    notify_on_trade_exit: bool = True
    notify_on_stop_loss: bool = True
    notify_on_error: bool = True
    notify_on_daily_summary: bool = True
    notify_on_position_adjustment: bool = True
    notify_on_risk_limit: bool = True
    
    # Throttling
    min_notification_interval: int = 60  # seconds between similar notifications
    max_notifications_per_hour: int = 50


@dataclass
class UISettings:
    """User interface preferences"""
    theme: str = "dark"  # dark, light, auto
    
    # Dashboard settings
    refresh_interval: int = 5  # seconds
    show_greeks: bool = True
    show_charts: bool = True
    chart_timeframe: str = "5m"  # 1m, 5m, 15m, 1h, 1d
    
    # Display preferences
    decimal_places: int = 2
    currency_symbol: str = "₹"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    timezone_display: str = "local"  # local, market, utc
    
    # Table settings
    default_page_size: int = 50
    show_closed_positions: bool = False
    position_grouping: str = "strategy"  # strategy, symbol, none
    
    # Performance display
    performance_period: str = "daily"  # daily, weekly, monthly
    show_benchmark: bool = True
    benchmark_symbol: str = "NIFTY"


@dataclass
class BacktestSettings:
    """Backtesting preferences"""
    # Data settings
    data_source: str = "historical"  # historical, simulated
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    initial_capital: float = 1000000
    
    # Execution simulation
    commission_per_lot: float = 20.0
    slippage_ticks: int = 1
    fill_probability: float = 0.95
    partial_fill_enabled: bool = True
    
    # Market simulation
    simulate_gaps: bool = True
    simulate_circuit_limits: bool = True
    simulate_liquidity: bool = True
    
    # Speed settings
    playback_speed: float = 1.0  # 1.0 = realtime, 0 = as fast as possible
    skip_non_trading_hours: bool = True
    
    # Analysis
    calculate_drawdown: bool = True
    calculate_sharpe: bool = True
    calculate_calmar: bool = True
    monte_carlo_runs: int = 1000
    confidence_level: float = 0.95


@dataclass
class DataSettings:
    """Data management preferences"""
    # Storage
    cache_enabled: bool = True
    cache_ttl_minutes: int = 5
    max_cache_size_mb: int = 100
    
    # Historical data
    auto_download_missing: bool = True
    data_retention_days: int = 365
    tick_data_retention_days: int = 30
    
    # Real-time data
    reconnect_on_disconnect: bool = True
    max_reconnect_attempts: int = 10
    reconnect_delay_seconds: int = 5
    
    # Data quality
    validate_data: bool = True
    fill_missing_data: bool = True
    outlier_detection: bool = True
    outlier_threshold_std: float = 3.0


@dataclass
class SecuritySettings:
    """Security preferences"""
    # API security
    require_api_key: bool = True
    api_key_rotation_days: int = 90
    allowed_ips: List[str] = field(default_factory=lambda: ["127.0.0.1"])
    
    # Session management
    session_timeout_minutes: int = 1440  # 24 hours
    require_2fa: bool = False
    
    # Audit
    audit_logging: bool = True
    audit_retention_days: int = 90
    
    # Encryption
    encrypt_sensitive_data: bool = True
    encryption_algorithm: str = "AES-256"


@dataclass
class PerformanceSettings:
    """Performance tuning preferences"""
    # Threading
    use_multiprocessing: bool = False
    worker_processes: int = 4
    thread_pool_size: int = 10
    
    # Database
    db_connection_pool_size: int = 5
    db_query_timeout_seconds: int = 30
    use_db_cache: bool = True
    
    # Memory
    max_memory_usage_mb: int = 2048
    garbage_collection_interval: int = 300  # seconds
    
    # Optimization
    use_numba: bool = True
    use_cython: bool = False
    vectorize_calculations: bool = True


@dataclass
class Settings:
    """Main settings container"""
    # Sub-settings
    notifications: NotificationSettings = field(default_factory=NotificationSettings)
    ui: UISettings = field(default_factory=UISettings)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)
    data: DataSettings = field(default_factory=DataSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    
    # User preferences
    user_name: str = "Trader"
    preferred_broker: str = "default"
    paper_trading_default: bool = True
    
    # Auto-save
    auto_save_enabled: bool = True
    auto_save_interval_minutes: int = 5
    
    # Advanced
    debug_mode: bool = False
    experimental_features: bool = False
    telemetry_enabled: bool = False
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'Settings':
        """Load settings from JSON file"""
        if not os.path.exists(file_path):
            return cls()
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        settings = cls()
        
        # Load sub-settings
        if 'notifications' in data:
            settings.notifications = NotificationSettings(**data['notifications'])
        
        if 'ui' in data:
            settings.ui = UISettings(**data['ui'])
        
        if 'backtest' in data:
            settings.backtest = BacktestSettings(**data['backtest'])
            
        if 'data' in data:
            settings.data = DataSettings(**data['data'])
            
        if 'security' in data:
            settings.security = SecuritySettings(**data['security'])
            
        if 'performance' in data:
            settings.performance = PerformanceSettings(**data['performance'])
        
        # Load user preferences
        settings.user_name = data.get('user_name', settings.user_name)
        settings.preferred_broker = data.get('preferred_broker', settings.preferred_broker)
        settings.paper_trading_default = data.get('paper_trading_default', settings.paper_trading_default)
        
        # Load auto-save
        settings.auto_save_enabled = data.get('auto_save_enabled', settings.auto_save_enabled)
        settings.auto_save_interval_minutes = data.get('auto_save_interval_minutes', settings.auto_save_interval_minutes)
        
        # Load advanced
        settings.debug_mode = data.get('debug_mode', settings.debug_mode)
        settings.experimental_features = data.get('experimental_features', settings.experimental_features)
        settings.telemetry_enabled = data.get('telemetry_enabled', settings.telemetry_enabled)
        
        return settings
    
    def save_to_file(self, file_path: str):
        """Save settings to JSON file"""
        # Create directory if needed
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'notifications': asdict(self.notifications),
            'ui': asdict(self.ui),
            'backtest': asdict(self.backtest),
            'data': asdict(self.data),
            'security': asdict(self.security),
            'performance': asdict(self.performance),
            'user_name': self.user_name,
            'preferred_broker': self.preferred_broker,
            'paper_trading_default': self.paper_trading_default,
            'auto_save_enabled': self.auto_save_enabled,
            'auto_save_interval_minutes': self.auto_save_interval_minutes,
            'debug_mode': self.debug_mode,
            'experimental_features': self.experimental_features,
            'telemetry_enabled': self.telemetry_enabled
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    
    def apply_environment_overrides(self):
        """Apply environment variable overrides"""
        # Debug mode
        if debug := os.getenv("DEBUG_MODE"):
            self.debug_mode = debug.lower() == "true"
        
        # Paper trading
        if paper := os.getenv("PAPER_TRADING"):
            self.paper_trading_default = paper.lower() == "true"
        
        # Performance
        if workers := os.getenv("WORKER_PROCESSES"):
            self.performance.worker_processes = int(workers)
        
        # Security
        if require_key := os.getenv("REQUIRE_API_KEY"):
            self.security.require_api_key = require_key.lower() == "true"
    
    def get_notification_channels(self) -> List[str]:
        """Get active notification channels"""
        channels = []
        if self.notifications.console:
            channels.append("console")
        if self.notifications.log_file:
            channels.append("log")
        if self.notifications.email:
            channels.append("email")
        if self.notifications.telegram:
            channels.append("telegram")
        if self.notifications.webhook:
            channels.append("webhook")
        return channels
    
    def validate(self) -> List[str]:
        """Validate settings and return list of warnings"""
        warnings = []
        
        # Check email settings
        if self.notifications.email and not self.notifications.email_smtp_host:
            warnings.append("Email notifications enabled but SMTP host not configured")
        
        # Check telegram settings
        if self.notifications.telegram and not self.notifications.telegram_bot_token:
            warnings.append("Telegram notifications enabled but bot token not configured")
        
        # Check security
        if not self.security.require_api_key and self.security.allowed_ips == ["127.0.0.1"]:
            warnings.append("API key not required and only localhost allowed - consider security implications")
        
        # Check performance
        if self.performance.max_memory_usage_mb < 512:
            warnings.append("Low memory limit may cause performance issues")
        
        return warnings