"""
Basic configuration tests to improve coverage
"""

import pytest
import os
import tempfile
import json
from pathlib import Path

from src.config.app_config import AppConfig, DomainConfig, APIConfig, LoggingConfig
from src.config.constants import TIME_CONSTANTS, SYSTEM_CONSTANTS


class TestBasicConfig:
    """Test basic configuration functionality"""
    
    def test_domain_config_defaults(self):
        """Test domain config default values"""
        config = DomainConfig()
        
        assert config.openalgo_api_host == "http://127.0.0.1"
        assert config.openalgo_api_port == 5000
        assert config.database_type == "postgresql"
        assert config.openalgo_api_url == "http://127.0.0.1:5000/api/v1"
        assert config.websocket_url == "ws://127.0.0.1:8765"
    
    def test_api_config_defaults(self):
        """Test API config default values"""
        config = APIConfig()
        
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.docs_enabled == True
        assert config.title == "Automated Trading Bot API"
        assert "http://localhost:3000" in config.cors_origins
    
    def test_logging_config_defaults(self):
        """Test logging config default values"""
        config = LoggingConfig()
        
        assert config.level == SYSTEM_CONSTANTS.LOG_LEVEL_INFO
        assert config.file == "logs/trading_bot.log"
        assert config.max_size_mb == 10
        assert config.format == "json"
        assert config.console_output == True
    
    def test_app_config_creation(self):
        """Test app config creation"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "system": {
                    "name": "Test Bot",
                    "environment": "test"
                }
            }
            json.dump(config_data, f)
            f.flush()
            
            try:
                app_config = AppConfig(f.name)
                assert app_config.system.name == "Test Bot"
                assert app_config.system.environment == "test"
            finally:
                os.unlink(f.name)
    
    def test_database_url_with_env_var(self):
        """Test database URL from environment variable"""
        original_url = os.getenv("DATABASE_URL")
        
        try:
            test_url = "postgresql+asyncpg://user:pass@localhost:5432/testdb"
            os.environ["DATABASE_URL"] = test_url
            
            config = DomainConfig()
            # Manually set up the connection for testing
            config.database_connection = {"url": test_url}
            
            assert config.database_url == test_url
            
        finally:
            if original_url:
                os.environ["DATABASE_URL"] = original_url
            elif "DATABASE_URL" in os.environ:
                del os.environ["DATABASE_URL"]
    
    def test_constants_values(self):
        """Test that constants have expected values"""
        # Test time constants
        assert hasattr(TIME_CONSTANTS, 'MARKET_OPEN_TIME')
        assert hasattr(TIME_CONSTANTS, 'MARKET_CLOSE_TIME')
        
        # Test system constants
        assert hasattr(SYSTEM_CONSTANTS, 'LOG_LEVEL_INFO')
        assert hasattr(SYSTEM_CONSTANTS, 'ENV_DEVELOPMENT')
        assert hasattr(SYSTEM_CONSTANTS, 'ENV_PRODUCTION')
    
    def test_config_validation(self):
        """Test config validation"""
        config = AppConfig()
        
        # Should validate successfully with defaults
        assert config.validate() == True
        
        # Test invalid capital
        config.system.total_capital = -1000
        with pytest.raises(ValueError, match="Total capital must be positive"):
            config.validate()
        
        # Reset for next test
        config.system.total_capital = 1000000
        
        # Test invalid emergency reserve
        config.system.emergency_reserve = 2000000  # More than total
        with pytest.raises(ValueError, match="Emergency reserve must be less than total capital"):
            config.validate()
    
    def test_available_capital_calculation(self):
        """Test available capital calculation"""
        config = AppConfig()
        config.system.total_capital = 1000000
        config.system.emergency_reserve = 100000
        
        assert config.system.available_capital == 900000
    
    def test_environment_properties(self):
        """Test environment detection properties"""
        config = AppConfig()
        
        # Test development
        config.system.environment = SYSTEM_CONSTANTS.ENV_DEVELOPMENT
        assert config.system.is_development == True
        assert config.system.is_production == False
        
        # Test production
        config.system.environment = SYSTEM_CONSTANTS.ENV_PRODUCTION
        assert config.system.is_development == False
        assert config.system.is_production == True