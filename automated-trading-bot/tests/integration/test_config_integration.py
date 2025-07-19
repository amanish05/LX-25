"""
Integration Tests for Configuration System
Tests the complete configuration management system
"""

import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.config import (
    ConfigManager, get_config_manager, reload_config,
    AppConfig, TradingParameters, Settings,
    TIME_CONSTANTS, TRADING_CONSTANTS, BOT_CONSTANTS
)


class TestConfigurationIntegration:
    """Integration tests for Configuration System"""
    
    @pytest.mark.asyncio
    async def test_config_manager_initialization(self, tmp_path):
        """Test configuration manager initialization"""
        # Create config directory
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create config files
        config_data = {
            "system": {
                "environment": "test",
                "total_capital": 1000000,
                "emergency_reserve": 100000
            },
            "domains": {
                "openalgo_api_host": "http://127.0.0.1",
                "openalgo_api_port": 5000
            }
        }
        
        with open(config_dir / "config.json", 'w') as f:
            json.dump(config_data, f)
        
        # Initialize config manager
        config_manager = ConfigManager(str(config_dir))
        
        # Verify configuration loaded
        assert config_manager.app_config.system.environment == "test"
        assert config_manager.app_config.system.total_capital == 1000000
    
    @pytest.mark.asyncio
    async def test_environment_variable_overrides(self, tmp_path):
        """Test environment variable overrides"""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create minimal config
        config_data = {"system": {"environment": "development"}}
        with open(config_dir / "config.json", 'w') as f:
            json.dump(config_data, f)
        
        # Set environment variables
        env_vars = {
            "ENVIRONMENT": "production",
            "TOTAL_CAPITAL": "2000000",
            "OPENALGO_API_HOST": "http://192.168.1.100",
            "LOG_LEVEL": "DEBUG"
        }
        
        with patch.dict(os.environ, env_vars):
            config_manager = ConfigManager(str(config_dir))
            
            # Verify overrides applied
            assert config_manager.app_config.system.environment == "production"
            assert config_manager.app_config.system.total_capital == 2000000
            assert config_manager.app_config.domains.openalgo_api_host == "http://192.168.1.100"
            assert config_manager.app_config.logging.level == "DEBUG"
    
    @pytest.mark.asyncio
    async def test_bot_specific_configuration(self, test_config_manager):
        """Test getting bot-specific configuration"""
        # Get configuration for short straddle bot
        bot_config = test_config_manager.get_bot_config(BOT_CONSTANTS.TYPE_SHORT_STRADDLE)
        
        # Verify structure
        assert "bot_name" in bot_config
        assert "system" in bot_config
        assert "domains" in bot_config
        assert "execution" in bot_config
        assert "monitoring" in bot_config
        assert "strategy_params" in bot_config
        assert "notifications" in bot_config
        assert "paper_trading" in bot_config
        
        # Verify values
        assert bot_config["bot_name"] == BOT_CONSTANTS.TYPE_SHORT_STRADDLE
        assert bot_config["system"]["environment"] == "test"
        assert bot_config["domains"]["openalgo_api_url"] == "http://127.0.0.1:5000/api/v1"
    
    @pytest.mark.asyncio
    async def test_trading_parameter_updates(self, test_config_manager):
        """Test updating trading parameters"""
        # Get initial value
        initial_iv_rank = test_config_manager.trading_params.short_straddle.entry.min_iv_rank
        
        # Update parameter
        test_config_manager.update_trading_param(
            BOT_CONSTANTS.TYPE_SHORT_STRADDLE,
            'entry.min_iv_rank',
            80
        )
        
        # Verify update
        assert test_config_manager.trading_params.short_straddle.entry.min_iv_rank == 80
        assert test_config_manager.trading_params.short_straddle.entry.min_iv_rank != initial_iv_rank
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self, tmp_path):
        """Test configuration validation"""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create invalid config (negative capital)
        invalid_config = {
            "system": {
                "total_capital": -1000,
                "emergency_reserve": 2000
            }
        }
        
        with open(config_dir / "config.json", 'w') as f:
            json.dump(invalid_config, f)
        
        # Should raise validation error
        with pytest.raises(ValueError, match="Total capital must be positive"):
            ConfigManager(str(config_dir))
    
    @pytest.mark.asyncio
    async def test_configuration_persistence(self, tmp_path):
        """Test saving and loading configuration"""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create config manager
        config_manager = ConfigManager(str(config_dir))
        
        # Modify configurations
        config_manager.app_config.system.total_capital = 2000000
        config_manager.trading_params.short_straddle.entry.min_iv_rank = 85
        config_manager.settings.notifications.email = True
        
        # Save all configurations
        config_manager.save_all()
        
        # Create new config manager to load saved configs
        new_config_manager = ConfigManager(str(config_dir))
        
        # Verify persistence
        assert new_config_manager.app_config.system.total_capital == 2000000
        assert new_config_manager.trading_params.short_straddle.entry.min_iv_rank == 85
        assert new_config_manager.settings.notifications.email is True
    
    @pytest.mark.asyncio
    async def test_configuration_export(self, test_config_manager, tmp_path):
        """Test configuration export functionality"""
        export_dir = tmp_path / "export"
        
        # Export configuration
        test_config_manager.export_all(str(export_dir))
        
        # Verify export file created
        export_file = export_dir / "complete_config.json"
        assert export_file.exists()
        
        # Load and verify exported data
        with open(export_file, 'r') as f:
            exported_data = json.load(f)
        
        assert "app_config" in exported_data
        assert "trading_params" in exported_data
        assert "settings" in exported_data
        
        # Verify content
        assert exported_data["app_config"]["system"]["environment"] == "test"
    
    @pytest.mark.asyncio
    async def test_constants_access(self):
        """Test accessing configuration constants"""
        # Time constants
        assert TIME_CONSTANTS.MARKET_OPEN_TIME == "09:15"
        assert TIME_CONSTANTS.MARKET_CLOSE_TIME == "15:30"
        
        # Trading constants
        assert TRADING_CONSTANTS.NIFTY_LOT_SIZE == 50
        assert TRADING_CONSTANTS.BANKNIFTY_LOT_SIZE == 25
        
        # Bot constants
        assert BOT_CONSTANTS.TYPE_SHORT_STRADDLE == "short_straddle"
        assert BOT_CONSTANTS.STATE_RUNNING == "running"
    
    @pytest.mark.asyncio
    async def test_global_risk_multiplier(self, test_config_manager):
        """Test global risk multiplier application"""
        # Set global risk multiplier
        test_config_manager.trading_params.global_risk_multiplier = 0.5
        test_config_manager.trading_params.apply_global_overrides()
        
        # Verify risk parameters adjusted
        original_risk = 0.10  # Original max_capital_per_position
        assert test_config_manager.trading_params.short_straddle.risk.max_capital_per_position == original_risk * 0.5
    
    @pytest.mark.asyncio
    async def test_settings_validation(self, test_config_manager):
        """Test settings validation and warnings"""
        # Enable email without SMTP config
        test_config_manager.settings.notifications.email = True
        test_config_manager.settings.notifications.email_smtp_host = ""
        
        # Get validation warnings
        warnings = test_config_manager.settings.validate()
        
        assert len(warnings) > 0
        assert any("Email notifications enabled but SMTP host not configured" in w for w in warnings)
    
    @pytest.mark.asyncio
    async def test_configuration_reload(self, tmp_path):
        """Test configuration hot reload"""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Initial config
        config_data = {"system": {"total_capital": 1000000}}
        config_file = config_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Create config manager
        config_manager = ConfigManager(str(config_dir))
        assert config_manager.app_config.system.total_capital == 1000000
        
        # Update config file
        config_data["system"]["total_capital"] = 2000000
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Reload configuration
        config_manager.reload()
        
        # Verify reload
        assert config_manager.app_config.system.total_capital == 2000000
    
    @pytest.mark.asyncio
    async def test_api_configuration(self, test_config_manager):
        """Test API configuration retrieval"""
        api_config = test_config_manager.get_api_config()
        
        # Verify structure
        assert "host" in api_config
        assert "port" in api_config
        assert "cors_origins" in api_config
        assert "docs_enabled" in api_config
        assert "require_api_key" in api_config
        assert "allowed_ips" in api_config
        
        # Verify values
        assert api_config["port"] == 8081  # Test port
        assert api_config["require_api_key"] is True
    
    @pytest.mark.asyncio
    async def test_logging_configuration(self, test_config_manager):
        """Test logging configuration"""
        logging_config = test_config_manager.get_logging_config()
        
        # Verify structure
        assert "level" in logging_config
        assert "file" in logging_config
        assert "max_size_mb" in logging_config
        assert "backup_count" in logging_config
        assert "format" in logging_config
        assert "console_output" in logging_config
        assert "debug_mode" in logging_config
        
        # Verify values
        assert logging_config["level"] == "DEBUG"  # Test level
    
    @pytest.mark.asyncio
    async def test_performance_configuration(self, test_config_manager):
        """Test performance configuration"""
        perf_config = test_config_manager.get_performance_config()
        
        # Verify structure
        assert "use_multiprocessing" in perf_config
        assert "worker_processes" in perf_config
        assert "thread_pool_size" in perf_config
        assert "db_connection_pool_size" in perf_config
        assert "max_memory_usage_mb" in perf_config
        assert "use_numba" in perf_config
        assert "vectorize_calculations" in perf_config
        
        # Verify values
        assert perf_config["thread_pool_size"] == 2  # Test value
    
    @pytest.mark.asyncio
    async def test_configuration_summary(self, test_config_manager):
        """Test configuration summary generation"""
        summary = test_config_manager.get_summary()
        
        # Verify summary contains key information
        assert "Configuration Summary" in summary
        assert "Environment: test" in summary
        assert "Total Capital:" in summary
        assert "OpenAlgo API:" in summary
        assert "Bot Configurations:" in summary
        assert "Short Straddle:" in summary
    
    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """Test configuration manager singleton"""
        # Get config manager instances
        config1 = get_config_manager()
        config2 = get_config_manager()
        
        # Should be the same instance
        assert config1 is config2
        
        # Modify one instance
        config1.app_config.system.total_capital = 3000000
        
        # Change should be reflected in other reference
        assert config2.app_config.system.total_capital == 3000000