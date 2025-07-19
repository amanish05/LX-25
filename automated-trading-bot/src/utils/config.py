"""
Configuration management for the trading bot system
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv


class Config:
    """Configuration manager with support for JSON, YAML, and environment variables"""
    
    def __init__(self, config_path: str = None):
        # Load environment variables
        load_dotenv()
        
        # Load configuration file
        self.config_path = config_path or os.getenv("TRADING_BOT_CONFIG", "config/config.json")
        self.config = self._load_config()
        
        # Override with environment variables
        self._apply_env_overrides()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(config_file, 'r') as f:
            if config_file.suffix == '.json':
                return json.load(f)
            elif config_file.suffix in ['.yml', '.yaml']:
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_file.suffix}")
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # OpenAlgo configuration
        if api_key := os.getenv("OPENALGO_API_KEY"):
            self.set("openalgo.api_key", api_key)
        
        if api_url := os.getenv("OPENALGO_API_URL"):
            self.set("openalgo.api_url", api_url)
        
        # Database configuration
        if db_url := os.getenv("DATABASE_URL"):
            self.set("database.url", db_url)
        
        # API configuration
        if api_port := os.getenv("API_PORT"):
            self.set("api.port", int(api_port))
        
        # Environment
        if env := os.getenv("ENVIRONMENT"):
            self.set("system.environment", env)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_bot_config(self, bot_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific bot"""
        return self.get(f"bots.{bot_name}")
    
    def get_all_bot_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configurations for all bots"""
        return self.get("bots", {})
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.get("system.environment") == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.get("system.environment") == "development"
    
    def validate(self) -> bool:
        """Validate configuration"""
        required_fields = [
            "openalgo.api_url",
            "openalgo.api_key",
            "database.url",
            "api.port"
        ]
        
        for field in required_fields:
            if not self.get(field):
                raise ValueError(f"Missing required configuration: {field}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        # Create a copy and mask sensitive data
        config_copy = json.loads(json.dumps(self.config))
        
        # Mask sensitive fields
        if "openalgo" in config_copy and "api_key" in config_copy["openalgo"]:
            config_copy["openalgo"]["api_key"] = "***masked***"
        
        if "database" in config_copy and "url" in config_copy["database"]:
            db_url = config_copy["database"]["url"]
            if "@" in db_url:  # Contains credentials
                parts = db_url.split("@")
                config_copy["database"]["url"] = f"{parts[0].split('//')[0]}//***:***@{parts[1]}"
        
        return config_copy