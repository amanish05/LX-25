# Configuration Guide

## Overview

The Automated Trading Bot uses a centralized configuration system that separates different aspects of configuration into logical modules:

1. **App Configuration** (`config/config.json`) - System-level settings like API endpoints, logging, monitoring
2. **Trading Parameters** (`config/trading_params.json`) - Strategy-specific trading parameters
3. **User Settings** (`config/settings.json`) - User preferences for UI, notifications, security

All configurations are managed through a unified `ConfigManager` that provides easy access and validation.

## Configuration Structure

```
config/
├── config.json                    # Application configuration
├── config.example.json            # Example application configuration
├── trading_params.json            # Trading parameters
├── trading_params.example.json    # Example trading parameters
├── settings.json                  # User settings
└── settings.example.json          # Example user settings
```

## Quick Start

1. Copy the example files to create your configuration:
```bash
cd config
cp config.example.json config.json
cp trading_params.example.json trading_params.json
cp settings.example.json settings.json
```

2. Edit the files according to your needs

3. Use the configuration in your code:
```python
from src.config import get_config_manager

# Get the configuration manager
config = get_config_manager()

# Access different configurations
app_config = config.app_config
trading_params = config.trading_params
settings = config.settings

# Get bot-specific configuration
bot_config = config.get_bot_config('short_straddle')
```

## Configuration Details

### Application Configuration

Controls system-level settings:

```json
{
    "system": {
        "environment": "development",  // development, staging, production
        "total_capital": 1000000,
        "emergency_reserve": 100000
    },
    "domains": {
        "openalgo_api_host": "http://127.0.0.1",
        "openalgo_api_port": 5000,
        "database_type": "sqlite"  // sqlite or postgresql
    },
    "api": {
        "port": 8080,
        "cors_origins": ["http://localhost:3000"]
    },
    "logging": {
        "level": "INFO",  // DEBUG, INFO, WARNING, ERROR, CRITICAL
        "file": "logs/trading_bot.log"
    }
}
```

### Trading Parameters

Controls strategy-specific parameters:

```json
{
    "short_straddle": {
        "entry": {
            "min_iv_rank": 70,
            "entry_start_time": "09:30",
            "position_size_method": "FIXED"
        },
        "exit": {
            "profit_target_percent": 50.0,
            "stop_loss_percent": 100.0
        },
        "risk": {
            "max_positions": 2,
            "max_risk_per_trade": 0.02
        }
    }
}
```

### User Settings

Controls user preferences:

```json
{
    "notifications": {
        "enabled": true,
        "console": true,
        "email": false
    },
    "ui": {
        "theme": "dark",
        "refresh_interval": 5
    },
    "security": {
        "require_api_key": true,
        "allowed_ips": ["127.0.0.1"]
    }
}
```

## Environment Variables

You can override certain configuration values using environment variables:

```bash
# System overrides
export ENVIRONMENT=production
export TOTAL_CAPITAL=2000000

# API overrides
export OPENALGO_API_HOST=http://192.168.1.100
export OPENALGO_API_PORT=5001
export API_PORT=8081

# Database override
export DATABASE_URL=postgresql://user:pass@localhost/db

# Logging override
export LOG_LEVEL=DEBUG

# Other overrides
export DEBUG_MODE=true
export PAPER_TRADING=true
export WORKER_PROCESSES=8
```

## Constants

All constants are defined in `src/config/constants.py` and organized by category:

```python
from src.config import TIME_CONSTANTS, TRADING_CONSTANTS, RISK_CONSTANTS

# Access constants
market_open = TIME_CONSTANTS.MARKET_OPEN_TIME
lot_size = TRADING_CONSTANTS.NIFTY_LOT_SIZE
max_risk = RISK_CONSTANTS.MAX_PORTFOLIO_RISK
```

## Using the Configuration Manager

### Basic Usage

```python
from src.config import get_config_manager

config = get_config_manager()

# Get complete bot configuration
bot_config = config.get_bot_config('short_straddle')

# Get API configuration
api_config = config.get_api_config()

# Get logging configuration
logging_config = config.get_logging_config()
```

### Updating Configuration

```python
# Update a trading parameter
config.update_trading_param('short_straddle', 'entry.min_iv_rank', 75)

# Save all configurations
config.save_all()

# Reload configurations from disk
config.reload()
```

### Configuration Summary

```python
# Get a summary of current configuration
print(config.get_summary())
```

### Export Configuration

```python
# Export all configurations to a directory
config.export_all('/path/to/export')
```

## Configuration in Bot Implementation

Here's how to use configuration in a bot:

```python
from src.config import get_config_manager, BOT_CONSTANTS
from src.bots.base_bot import BaseBot

class ShortStraddleBot(BaseBot):
    def __init__(self, name: str):
        super().__init__(name, BOT_CONSTANTS.TYPE_SHORT_STRADDLE)
        
        # Get configuration
        config = get_config_manager()
        self.bot_config = config.get_bot_config(self.bot_type)
        
        # Access strategy parameters
        self.params = self.bot_config['strategy_params']
        self.min_iv_rank = self.params.entry.min_iv_rank
        self.profit_target = self.params.exit.profit_target_percent
        self.max_positions = self.params.risk.max_positions
```

## Configuration Validation

The configuration manager automatically validates configurations:

```python
config = get_config_manager()  # Validates on load

# Manual validation
try:
    config.app_config.validate()
except ValueError as e:
    print(f"Configuration error: {e}")

# Check for warnings
warnings = config.settings.validate()
for warning in warnings:
    print(f"Warning: {warning}")
```

## Best Practices

1. **Use Example Files**: Start with the provided example files and modify as needed

2. **Environment-Specific Configs**: Use different config files for different environments:
   ```
   config/
   ├── config.development.json
   ├── config.staging.json
   └── config.production.json
   ```

3. **Secure Sensitive Data**: Never commit sensitive data like API keys. Use environment variables:
   ```bash
   export OPENALGO_API_KEY=your-secret-key
   ```

4. **Version Control**: Add actual config files to `.gitignore`, only commit example files:
   ```gitignore
   config/config.json
   config/trading_params.json
   config/settings.json
   ```

5. **Regular Backups**: Backup your configuration files regularly:
   ```python
   config.export_all(f'/backups/config_{datetime.now().strftime("%Y%m%d")}')
   ```

6. **Gradual Changes**: When adjusting trading parameters, make small incremental changes

7. **Paper Trading First**: Always test configuration changes in paper trading mode first

## Troubleshooting

### Configuration Not Loading

1. Check file paths and permissions
2. Verify JSON syntax (use a JSON validator)
3. Check for missing required fields
4. Look for validation errors in logs

### Environment Variables Not Working

1. Ensure variables are exported: `export VAR=value`
2. Check variable names match exactly
3. Restart the application after setting variables

### Performance Issues

1. Adjust `thread_pool_size` and `worker_processes`
2. Check `max_memory_usage_mb` setting
3. Enable/disable performance features like `use_numba`

## Advanced Topics

### Custom Configuration Sources

You can extend the configuration system to load from other sources:

```python
class CustomConfigManager(ConfigManager):
    def _load_from_database(self):
        # Load configuration from database
        pass
    
    def _load_from_remote(self):
        # Load configuration from remote service
        pass
```

### Configuration Hooks

Add hooks to respond to configuration changes:

```python
def on_config_change(old_config, new_config):
    # Handle configuration changes
    if old_config.app_config.system.environment != new_config.app_config.system.environment:
        # Environment changed, restart services
        pass

config.add_change_listener(on_config_change)
```

### Configuration Profiles

Create profiles for different trading scenarios:

```python
profiles = {
    'conservative': {
        'global_risk_multiplier': 0.5,
        'global_stop_loss_multiplier': 0.8
    },
    'aggressive': {
        'global_risk_multiplier': 1.5,
        'global_stop_loss_multiplier': 1.2
    }
}

def apply_profile(profile_name):
    profile = profiles[profile_name]
    config.trading_params.global_risk_multiplier = profile['global_risk_multiplier']
    config.trading_params.global_stop_loss_multiplier = profile['global_stop_loss_multiplier']
    config.save_all()
```