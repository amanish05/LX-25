"""
Configuration Manager
Unified configuration management that combines AppConfig, TradingParameters, and Settings
"""

import os
import json
from typing import Optional, Dict, Any
from pathlib import Path
from .app_config import AppConfig
from .trading_params import TradingParameters
from .settings import Settings


class ConfigManager:
    """
    Central configuration manager that handles all configuration aspects
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        self.app_config = self._load_app_config()
        self.trading_params = self._load_trading_params()
        self.settings = self._load_settings()
        
        # Apply any global overrides
        self.trading_params.apply_global_overrides()
        self.settings.apply_environment_overrides()
        
        # Validate configurations
        self._validate_all()
    
    def _load_app_config(self) -> AppConfig:
        """Load application configuration"""
        config_file = self.config_dir / "config.json"
        if config_file.exists():
            return AppConfig(str(config_file))
        else:
            # Use example if main config doesn't exist
            example_file = self.config_dir / "config.example.json"
            if example_file.exists():
                return AppConfig(str(example_file))
            return AppConfig()
    
    def _load_trading_params(self) -> TradingParameters:
        """Load trading parameters"""
        params_file = self.config_dir / "trading_params.json"
        if params_file.exists():
            with open(params_file, 'r') as f:
                data = json.load(f)
            return TradingParameters.from_dict(data)
        else:
            # Use example if main params don't exist
            example_file = self.config_dir / "trading_params.example.json"
            if example_file.exists():
                with open(example_file, 'r') as f:
                    data = json.load(f)
                return TradingParameters.from_dict(data)
            return TradingParameters()
    
    def _load_settings(self) -> Settings:
        """Load user settings"""
        settings_file = self.config_dir / "settings.json"
        if settings_file.exists():
            return Settings.load_from_file(str(settings_file))
        else:
            # Use example if main settings don't exist
            example_file = self.config_dir / "settings.example.json"
            if example_file.exists():
                return Settings.load_from_file(str(example_file))
            return Settings()
    
    def _validate_all(self):
        """Validate all configurations"""
        # Validate app config
        self.app_config.validate()
        
        # Validate settings and show warnings
        warnings = self.settings.validate()
        if warnings:
            for warning in warnings:
                print(f"Configuration Warning: {warning}")
    
    def save_all(self):
        """Save all configurations"""
        # Save app config
        self.app_config.save_to_file(str(self.config_dir / "config.json"))
        
        # Save trading params
        with open(self.config_dir / "trading_params.json", 'w') as f:
            json.dump(self.trading_params.to_dict(), f, indent=4)
        
        # Save settings
        self.settings.save_to_file(str(self.config_dir / "settings.json"))
    
    def get_bot_config(self, bot_name: str) -> Dict[str, Any]:
        """Get complete configuration for a specific bot"""
        # Get strategy parameters
        strategy_params = self.trading_params.get_strategy_params(bot_name)
        
        return {
            'bot_name': bot_name,
            'system': {
                'environment': self.app_config.system.environment,
                'is_production': self.app_config.system.is_production,
                'available_capital': self.app_config.system.available_capital,
            },
            'domains': {
                'openalgo_api_url': self.app_config.domains.openalgo_api_url,
                'websocket_url': self.app_config.domains.websocket_url,
                'database_url': self.app_config.domains.database_url,
            },
            'execution': {
                'order_timeout': self.app_config.execution.order_timeout_seconds,
                'retry_attempts': self.app_config.execution.retry_attempts,
                'slippage_tolerance': self.app_config.execution.slippage_tolerance,
                'market_order_protection': self.app_config.execution.market_order_protection,
            },
            'monitoring': {
                'health_check_interval': self.app_config.monitoring.health_check_interval,
                'position_sync_interval': self.app_config.monitoring.position_sync_interval,
                'alert_channels': self.app_config.monitoring.alert_channels,
            },
            'strategy_params': strategy_params,
            'notifications': {
                'enabled': self.settings.notifications.enabled,
                'channels': self.settings.get_notification_channels(),
            },
            'paper_trading': self.trading_params.paper_trading_mode or self.settings.paper_trading_default,
        }
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API server configuration"""
        return {
            'host': self.app_config.api.host,
            'port': self.app_config.api.port,
            'cors_origins': self.app_config.api.cors_origins,
            'docs_enabled': self.app_config.api.docs_enabled,
            'title': self.app_config.api.title,
            'version': self.app_config.api.version,
            'require_api_key': self.settings.security.require_api_key,
            'allowed_ips': self.settings.security.allowed_ips,
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            'level': self.app_config.logging.level,
            'file': self.app_config.logging.file,
            'max_size_mb': self.app_config.logging.max_size_mb,
            'backup_count': self.app_config.logging.backup_count,
            'format': self.app_config.logging.format,
            'console_output': self.app_config.logging.console_output,
            'debug_mode': self.settings.debug_mode,
        }
    
    def get_market_hours(self) -> Dict[str, Any]:
        """Get market hours configuration"""
        return {
            'start': self.app_config.market_hours.start,
            'end': self.app_config.market_hours.end,
            'timezone': self.app_config.market_hours.timezone,
            'pre_market_start': self.app_config.market_hours.pre_market_start,
            'post_market_end': self.app_config.market_hours.post_market_end,
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return {
            'use_multiprocessing': self.settings.performance.use_multiprocessing,
            'worker_processes': self.settings.performance.worker_processes,
            'thread_pool_size': self.app_config.system.thread_pool_size,
            'db_connection_pool_size': self.settings.performance.db_connection_pool_size,
            'max_memory_usage_mb': self.settings.performance.max_memory_usage_mb,
            'use_numba': self.settings.performance.use_numba,
            'vectorize_calculations': self.settings.performance.vectorize_calculations,
        }
    
    def update_trading_param(self, bot_name: str, param_path: str, value: Any):
        """Update a specific trading parameter"""
        # Parse the parameter path (e.g., "entry.min_iv_rank")
        parts = param_path.split('.')
        
        # Get the strategy params
        strategy_params = self.trading_params.get_strategy_params(bot_name)
        if not strategy_params:
            raise ValueError(f"Unknown bot: {bot_name}")
        
        # Navigate to the parameter and update it
        obj = strategy_params
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
        
        # Save the updated configuration
        self.save_all()
    
    def reload(self):
        """Reload all configurations from disk"""
        self.app_config = self._load_app_config()
        self.trading_params = self._load_trading_params()
        self.settings = self._load_settings()
        
        # Re-apply overrides
        self.trading_params.apply_global_overrides()
        self.settings.apply_environment_overrides()
        
        # Re-validate
        self._validate_all()
    
    def export_all(self, export_dir: str):
        """Export all configurations to a directory"""
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Export complete configuration
        complete_config = {
            'app_config': self.app_config.to_dict(),
            'trading_params': self.trading_params.to_dict(),
            'settings': {
                'notifications': self.settings.notifications.__dict__,
                'ui': self.settings.ui.__dict__,
                'backtest': self.settings.backtest.__dict__,
                'data': self.settings.data.__dict__,
                'security': self.settings.security.__dict__,
                'performance': self.settings.performance.__dict__,
                'user_preferences': {
                    'user_name': self.settings.user_name,
                    'preferred_broker': self.settings.preferred_broker,
                    'paper_trading_default': self.settings.paper_trading_default,
                    'debug_mode': self.settings.debug_mode,
                }
            }
        }
        
        with open(export_path / "complete_config.json", 'w') as f:
            json.dump(complete_config, f, indent=4)
        
        print(f"Configuration exported to {export_path}")
    
    def get_summary(self) -> str:
        """Get a summary of the current configuration"""
        return f"""
Configuration Summary:
=====================
Environment: {self.app_config.system.environment}
Total Capital: {self.app_config.system.total_capital:,.0f}
Available Capital: {self.app_config.system.available_capital:,.0f}

OpenAlgo API: {self.app_config.domains.openalgo_api_url}
WebSocket: {self.app_config.domains.websocket_url}
Database: {self.app_config.domains.database_type}

API Server: http://{self.app_config.api.host}:{self.app_config.api.port}
Logging Level: {self.app_config.logging.level}

Active Notification Channels: {', '.join(self.settings.get_notification_channels())}
Paper Trading Default: {self.settings.paper_trading_default}
Debug Mode: {self.settings.debug_mode}

Bot Configurations:
- Short Straddle: Max Positions = {self.trading_params.short_straddle.risk.max_positions}
- Iron Condor: Max Positions = {self.trading_params.iron_condor.risk.max_positions}
- Volatility Expander: Max Positions = {self.trading_params.volatility_expander.risk.max_positions}
- Momentum Rider: Max Positions = {self.trading_params.momentum_rider.risk.max_positions}

Global Risk Multiplier: {self.trading_params.global_risk_multiplier}
"""


# Singleton instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_dir: str = "config") -> ConfigManager:
    """Get the singleton configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_dir)
    return _config_manager


def reload_config():
    """Reload the configuration manager"""
    global _config_manager
    if _config_manager is not None:
        _config_manager.reload()