"""
Example: Using the Centralized Configuration System
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    get_config_manager, 
    TIME_CONSTANTS, 
    TRADING_CONSTANTS,
    BOT_CONSTANTS
)


def main():
    """Demonstrate configuration usage"""
    
    # Get the configuration manager
    config = get_config_manager()
    
    print("=== Configuration System Demo ===\n")
    
    # 1. Access Constants
    print("1. Accessing Constants:")
    print(f"   Market Open Time: {TIME_CONSTANTS.MARKET_OPEN_TIME}")
    print(f"   NIFTY Lot Size: {TRADING_CONSTANTS.NIFTY_LOT_SIZE}")
    print(f"   Max Positions per Bot: {BOT_CONSTANTS.MAX_ERROR_COUNT}")
    print()
    
    # 2. Access Application Configuration
    print("2. Application Configuration:")
    print(f"   Environment: {config.app_config.system.environment}")
    print(f"   Total Capital: ₹{config.app_config.system.total_capital:,.0f}")
    print(f"   Available Capital: ₹{config.app_config.system.available_capital:,.0f}")
    print(f"   OpenAlgo API: {config.app_config.domains.openalgo_api_url}")
    print()
    
    # 3. Access Trading Parameters
    print("3. Trading Parameters (Short Straddle):")
    ss_params = config.trading_params.short_straddle
    print(f"   Min IV Rank: {ss_params.entry.min_iv_rank}")
    print(f"   Profit Target: {ss_params.exit.profit_target_percent}%")
    print(f"   Stop Loss: {ss_params.exit.stop_loss_percent}%")
    print(f"   Max Positions: {ss_params.risk.max_positions}")
    print()
    
    # 4. Access User Settings
    print("4. User Settings:")
    print(f"   Theme: {config.settings.ui.theme}")
    print(f"   Notifications Enabled: {config.settings.notifications.enabled}")
    print(f"   Active Channels: {', '.join(config.settings.get_notification_channels())}")
    print(f"   Paper Trading Default: {config.settings.paper_trading_default}")
    print()
    
    # 5. Get Bot-Specific Configuration
    print("5. Bot-Specific Configuration:")
    bot_config = config.get_bot_config(BOT_CONSTANTS.TYPE_SHORT_STRADDLE)
    print(f"   Bot: {bot_config['bot_name']}")
    print(f"   Environment: {bot_config['system']['environment']}")
    print(f"   Paper Trading: {bot_config['paper_trading']}")
    print(f"   Entry Start Time: {bot_config['strategy_params'].entry.entry_start_time}")
    print()
    
    # 6. Get API Configuration
    print("6. API Server Configuration:")
    api_config = config.get_api_config()
    print(f"   Host: {api_config['host']}")
    print(f"   Port: {api_config['port']}")
    print(f"   CORS Origins: {api_config['cors_origins']}")
    print(f"   Docs Enabled: {api_config['docs_enabled']}")
    print()
    
    # 7. Configuration Summary
    print("7. Configuration Summary:")
    print(config.get_summary())
    
    # 8. Example: Updating a Parameter
    print("8. Updating Trading Parameters:")
    print(f"   Current Min IV Rank: {config.trading_params.short_straddle.entry.min_iv_rank}")
    
    # Update parameter
    config.update_trading_param(
        BOT_CONSTANTS.TYPE_SHORT_STRADDLE, 
        'entry.min_iv_rank', 
        75
    )
    
    print(f"   Updated Min IV Rank: {config.trading_params.short_straddle.entry.min_iv_rank}")
    print()
    
    # 9. Validation
    print("9. Configuration Validation:")
    try:
        config.app_config.validate()
        print("   ✓ App configuration is valid")
    except ValueError as e:
        print(f"   ✗ Validation error: {e}")
    
    warnings = config.settings.validate()
    if warnings:
        print("   Warnings:")
        for warning in warnings:
            print(f"     - {warning}")
    else:
        print("   ✓ No warnings")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()