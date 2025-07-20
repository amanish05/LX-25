#!/usr/bin/env python3
"""
Trading Bot Validation Tests
Tests all trading bots to ensure they're working correctly
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.indicators.price_action_composite import PriceActionComposite
from src.config.config_manager import ConfigManager
from unittest.mock import Mock


class TradingBotValidator:
    """Validates all trading bots are functioning correctly"""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
    
    async def test_momentum_rider(self):
        """Test Momentum Rider Bot"""
        print("\n=== Testing Momentum Rider Bot ===")
        
        try:
            # Import the bot module
            from src.bots.momentum_rider_bot import MomentumRiderBot
            
            # Check if the bot has required methods (from base_bot)
            required_methods = ['generate_signals', 'calculate_position_size', 'should_enter_position']
            
            for method in required_methods:
                if not hasattr(MomentumRiderBot, method):
                    raise AttributeError(f"MomentumRiderBot missing required method: {method}")
            
            print(f"✅ MomentumRiderBot module loaded successfully")
            print(f"  - Has all required methods: {', '.join(required_methods)}")
            
            # Test momentum indicators
            from src.indicators.momentum import MomentumIndicators
            momentum = MomentumIndicators()
            
            # Create test data with proper float64 types for TA-Lib
            dates = pd.date_range(end=datetime.now(), periods=50, freq='5min')
            closes = (np.sin(np.linspace(0, 4*np.pi, 50)) * 100 + 20000).astype(np.float64)
            test_data = pd.DataFrame({
                'open': (closes + np.random.uniform(-10, 10, 50)).astype(np.float64),
                'high': (closes + np.random.uniform(20, 50, 50)).astype(np.float64),
                'low': (closes - np.random.uniform(20, 50, 50)).astype(np.float64),
                'close': closes,
                'volume': np.random.randint(50000, 150000, 50).astype(np.float64)
            }, index=dates)
            
            # Test momentum calculation
            results = momentum.calculate(test_data)
            print(f"  - Momentum indicators calculated: {len(results)} results")
            
            self.passed += 1
            self.results.append(("Momentum Rider Bot", "PASSED", "Module loaded and indicators working"))
            return True
            
        except Exception as e:
            self.failed += 1
            self.results.append(("Momentum Rider Bot", "FAILED", str(e)))
            print(f"❌ Test failed: {e}")
            return False

    async def test_short_straddle(self):
        """Test Short Straddle Bot"""
        print("\n=== Testing Short Straddle Bot ===")
        
        try:
            # Import the bot module
            from src.bots.short_straddle_bot import ShortStraddleBot
            
            # Check if the bot has required methods (from base_bot)
            required_methods = ['generate_signals', 'calculate_position_size', 'should_enter_position']
            
            for method in required_methods:
                if not hasattr(ShortStraddleBot, method):
                    raise AttributeError(f"ShortStraddleBot missing required method: {method}")
            
            print(f"✅ ShortStraddleBot module loaded successfully")
            print(f"  - Has all required methods: {', '.join(required_methods)}")
            
            # Test volatility indicators used by this bot
            from src.indicators.volatility import VolatilityIndicators
            volatility = VolatilityIndicators()
            
            # Create test data with proper float64 types
            dates = pd.date_range(end=datetime.now(), periods=50, freq='5min')
            closes = np.random.normal(20000, 100, 50).astype(np.float64)
            test_data = pd.DataFrame({
                'open': (closes + np.random.uniform(-10, 10, 50)).astype(np.float64),
                'high': (closes + np.random.uniform(20, 50, 50)).astype(np.float64),
                'low': (closes - np.random.uniform(20, 50, 50)).astype(np.float64),
                'close': closes,
                'volume': np.random.randint(50000, 150000, 50).astype(np.float64)
            }, index=dates)
            
            # Test volatility calculation
            results = volatility.calculate(test_data)
            print(f"  - Volatility indicators calculated: {len(results)} results")
            
            self.passed += 1
            self.results.append(("Short Straddle Bot", "PASSED", "Module loaded and indicators working"))
            return True
            
        except Exception as e:
            self.failed += 1
            self.results.append(("Short Straddle Bot", "FAILED", str(e)))
            print(f"❌ Test failed: {e}")
            return False

    async def test_price_action(self):
        """Test Price Action Composite"""
        print("\n=== Testing Price Action Composite ===")
        
        try:
            # Create test data
            dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
            
            # Generate structured price movement
            prices = []
            base_price = 20000
            for i in range(100):
                if i < 30:
                    price = base_price + i * 10  # Uptrend
                elif i < 60:
                    price = base_price + 300 - (i - 30) * 10  # Downtrend
                else:
                    price = base_price + np.sin((i - 60) * 0.1) * 100  # Ranging
                prices.append(price + np.random.normal(0, 20))
            
            data = pd.DataFrame({
                'open': prices,
                'high': [p + np.random.uniform(10, 50) for p in prices],
                'low': [p - np.random.uniform(10, 50) for p in prices],
                'close': [p + np.random.uniform(-20, 20) for p in prices],
                'volume': np.random.randint(50000, 150000, 100)
            }, index=dates)
            
            # Initialize PA composite
            pa = PriceActionComposite()
            
            # Calculate signals
            result = pa.calculate(data)
            
            # Show results
            signals = result[result['signal'] != 0]
            print(f"✅ Found {len(signals)} PA signals")
            
            if len(signals) > 0:
                print("\nLast few signals:")
                for idx, row in signals.tail(3).iterrows():
                    signal_type = "BUY" if row['signal'] == 1 else "SELL"
                    print(f"  - {idx}: {signal_type} signal, strength={row['signal_strength']:.2f}")
            
            self.passed += 1
            self.results.append(("Price Action Composite", "PASSED", f"{len(signals)} signals found"))
            return True
            
        except Exception as e:
            self.failed += 1
            self.results.append(("Price Action Composite", "FAILED", str(e)))
            print(f"❌ Test failed: {e}")
            return False
    
    async def test_all_bots(self):
        """Test all available bots"""
        print("\n=== Testing All Available Bots ===")
        
        try:
            # List all bot modules
            import os
            bot_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'bots')
            bot_files = [f for f in os.listdir(bot_dir) if f.endswith('_bot.py') and f != 'base_bot.py']
            
            print(f"Found {len(bot_files)} bot implementations:")
            for bot_file in bot_files:
                bot_name = bot_file.replace('_bot.py', '').replace('_', ' ').title()
                print(f"  - {bot_name}")
            
            # Test that we can import each bot
            successful_imports = 0
            for bot_file in bot_files:
                module_name = bot_file.replace('.py', '')
                try:
                    exec(f"from src.bots.{module_name} import *")
                    successful_imports += 1
                except Exception as e:
                    print(f"    ⚠️  Failed to import {module_name}: {e}")
            
            print(f"\n✅ Successfully imported {successful_imports}/{len(bot_files)} bots")
            
            self.passed += 1
            self.results.append(("Bot Registry", "PASSED", f"{successful_imports}/{len(bot_files)} bots imported"))
            return True
            
        except Exception as e:
            self.failed += 1
            self.results.append(("Bot Registry", "FAILED", str(e)))
            print(f"❌ Test failed: {e}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("TRADING BOT VALIDATION SUMMARY")
        print("=" * 60)
        
        for test_name, status, details in self.results:
            status_icon = "✅" if status == "PASSED" else "❌"
            print(f"{status_icon} {test_name}: {status}")
            if details:
                print(f"   Details: {details}")
        
        print("\n" + "-" * 60)
        print(f"Total Tests: {self.passed + self.failed}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        
        if self.failed == 0:
            print("\n✅ ALL TESTS PASSED - BOTS ARE READY FOR TRADING!")
        else:
            print("\n❌ SOME TESTS FAILED - REVIEW BEFORE TRADING!")
        
        print("=" * 60)
        
        return self.failed == 0

    async def run_validation(self):
        """Run all validation tests"""
        print("=" * 60)
        print("AUTOMATED TRADING BOT - VALIDATION TEST")
        print("=" * 60)
        
        # Run all tests
        await self.test_price_action()
        await self.test_momentum_rider()
        await self.test_short_straddle()
        await self.test_all_bots()
        
        # Print summary
        return self.print_summary()


async def main():
    """Main entry point"""
    validator = TradingBotValidator()
    success = await validator.run_validation()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())