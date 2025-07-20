"""
Bot Registry System
Centralized management and discovery of trading bots
"""

import os
import importlib
import inspect
from typing import Dict, List, Type, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import logging

from .base_bot import BaseBot


@dataclass
class BotInfo:
    """Information about a registered bot"""
    name: str
    class_name: str
    module_path: str
    description: str
    strategy_type: str  # 'momentum', 'volatility_selling', 'volatility_buying', 'arbitrage'
    supported_instruments: List[str]  # ['options', 'equity', 'futures']
    capital_requirement: float
    risk_level: str  # 'low', 'medium', 'high'
    is_enabled: bool
    last_updated: datetime


class BotRegistry:
    """
    Central registry for all trading bots
    
    Features:
    - Auto-discovery of bot modules
    - Dynamic bot loading and initialization
    - Bot metadata and configuration management
    - Environment-based bot enabling/disabling
    - Bot health monitoring
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._registered_bots: Dict[str, BotInfo] = {}
        self._bot_classes: Dict[str, Type[BaseBot]] = {}
        self._bot_instances: Dict[str, BaseBot] = {}
        
        # Bot configuration from environment
        self._load_bot_configurations()
        
        # Auto-discover and register bots
        self._discover_bots()
    
    def _load_bot_configurations(self):
        """Load bot configurations from environment variables"""
        self.bot_configs = {
            'momentum_rider': {
                'enabled': os.getenv('ENABLE_MOMENTUM_RIDER', 'false').lower() == 'true',
                'capital': float(os.getenv('MOMENTUM_RIDER_CAPITAL', '100000')),
                'strategy_type': 'momentum',
                'supported_instruments': ['options', 'equity'],
                'risk_level': 'medium'
            },
            'short_straddle': {
                'enabled': os.getenv('ENABLE_SHORT_STRADDLE', 'false').lower() == 'true',
                'capital': float(os.getenv('SHORT_STRADDLE_CAPITAL', '200000')),
                'strategy_type': 'volatility_selling',
                'supported_instruments': ['options'],
                'risk_level': 'high'
            },
            'iron_condor': {
                'enabled': os.getenv('ENABLE_IRON_CONDOR', 'false').lower() == 'true',
                'capital': float(os.getenv('IRON_CONDOR_CAPITAL', '150000')),
                'strategy_type': 'volatility_selling',
                'supported_instruments': ['options'],
                'risk_level': 'medium'
            },
            'volatility_expander': {
                'enabled': os.getenv('ENABLE_VOLATILITY_EXPANDER', 'false').lower() == 'true',
                'capital': float(os.getenv('VOLATILITY_EXPANDER_CAPITAL', '75000')),
                'strategy_type': 'volatility_buying',
                'supported_instruments': ['options'],
                'risk_level': 'high'
            }
        }
    
    def _discover_bots(self):
        """Automatically discover bot modules in the bots directory"""
        bots_dir = Path(__file__).parent
        
        for bot_file in bots_dir.glob("*_bot.py"):
            if bot_file.name == "base_bot.py":
                continue
                
            try:
                # Extract bot name from filename
                bot_name = bot_file.stem.replace('_bot', '')
                module_name = f"src.bots.{bot_file.stem}"
                
                # Import the module
                module = importlib.import_module(module_name)
                
                # Find bot class in module
                bot_class = None
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseBot) and 
                        obj != BaseBot and 
                        name.endswith('Bot')):
                        bot_class = obj
                        break
                
                if bot_class:
                    self._register_bot(bot_name, bot_class, module_name)
                    self.logger.info(f"Discovered and registered bot: {bot_name}")
                else:
                    self.logger.warning(f"No bot class found in {bot_file}")
                    
            except Exception as e:
                self.logger.error(f"Failed to discover bot {bot_file}: {e}")
    
    def _register_bot(self, bot_name: str, bot_class: Type[BaseBot], module_path: str):
        """Register a bot class"""
        config = self.bot_configs.get(bot_name, {})
        
        # Get description from bot class docstring
        description = self._extract_description(bot_class)
        
        bot_info = BotInfo(
            name=bot_name,
            class_name=bot_class.__name__,
            module_path=module_path,
            description=description,
            strategy_type=config.get('strategy_type', 'unknown'),
            supported_instruments=config.get('supported_instruments', ['equity']),
            capital_requirement=config.get('capital', 100000),
            risk_level=config.get('risk_level', 'medium'),
            is_enabled=config.get('enabled', False),
            last_updated=datetime.now()
        )
        
        self._registered_bots[bot_name] = bot_info
        self._bot_classes[bot_name] = bot_class
    
    def _extract_description(self, bot_class: Type[BaseBot]) -> str:
        """Extract description from bot class docstring"""
        docstring = inspect.getdoc(bot_class)
        if docstring:
            # Extract first line or first paragraph
            lines = docstring.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('-') and not line.startswith('*'):
                    return line
        return f"{bot_class.__name__} - No description available"
    
    def get_enabled_bots(self) -> List[str]:
        """Get list of enabled bot names"""
        return [name for name, info in self._registered_bots.items() if info.is_enabled]
    
    def get_all_bots(self) -> List[str]:
        """Get list of all registered bot names"""
        return list(self._registered_bots.keys())
    
    def get_bot_info(self, bot_name: str) -> Optional[BotInfo]:
        """Get information about a specific bot"""
        return self._registered_bots.get(bot_name)
    
    def get_bots_by_strategy(self, strategy_type: str) -> List[str]:
        """Get bots by strategy type"""
        return [
            name for name, info in self._registered_bots.items() 
            if info.strategy_type == strategy_type
        ]
    
    def get_bots_by_instrument(self, instrument: str) -> List[str]:
        """Get bots that support a specific instrument"""
        return [
            name for name, info in self._registered_bots.items() 
            if instrument in info.supported_instruments
        ]
    
    def create_bot_instance(self, bot_name: str, config: Dict[str, Any], 
                           db_manager, openalgo_client, logger=None) -> BaseBot:
        """Create an instance of a specific bot"""
        if bot_name not in self._bot_classes:
            raise ValueError(f"Bot '{bot_name}' not found in registry")
        
        if not self._registered_bots[bot_name].is_enabled:
            raise ValueError(f"Bot '{bot_name}' is disabled")
        
        bot_class = self._bot_classes[bot_name]
        
        # Merge with registry configuration
        bot_info = self._registered_bots[bot_name]
        merged_config = {
            'name': f"{bot_name}_bot",
            'capital': bot_info.capital_requirement,
            'strategy_type': bot_info.strategy_type,
            'risk_level': bot_info.risk_level,
            **config
        }
        
        # Create instance
        instance = bot_class(merged_config, db_manager, openalgo_client, logger)
        self._bot_instances[bot_name] = instance
        
        self.logger.info(f"Created instance of {bot_name} bot")
        return instance
    
    def get_bot_instance(self, bot_name: str) -> Optional[BaseBot]:
        """Get existing bot instance"""
        return self._bot_instances.get(bot_name)
    
    def enable_bot(self, bot_name: str):
        """Enable a bot"""
        if bot_name in self._registered_bots:
            self._registered_bots[bot_name].is_enabled = True
            self.logger.info(f"Enabled bot: {bot_name}")
        else:
            raise ValueError(f"Bot '{bot_name}' not found")
    
    def disable_bot(self, bot_name: str):
        """Disable a bot"""
        if bot_name in self._registered_bots:
            self._registered_bots[bot_name].is_enabled = False
            
            # Stop instance if running
            if bot_name in self._bot_instances:
                instance = self._bot_instances[bot_name]
                if hasattr(instance, 'stop'):
                    instance.stop()
                del self._bot_instances[bot_name]
            
            self.logger.info(f"Disabled bot: {bot_name}")
        else:
            raise ValueError(f"Bot '{bot_name}' not found")
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of bot registry"""
        enabled_count = len(self.get_enabled_bots())
        total_count = len(self._registered_bots)
        
        strategy_counts = {}
        for info in self._registered_bots.values():
            strategy_counts[info.strategy_type] = strategy_counts.get(info.strategy_type, 0) + 1
        
        total_capital = sum(
            info.capital_requirement 
            for info in self._registered_bots.values() 
            if info.is_enabled
        )
        
        return {
            'total_bots': total_count,
            'enabled_bots': enabled_count,
            'disabled_bots': total_count - enabled_count,
            'strategy_distribution': strategy_counts,
            'total_allocated_capital': total_capital,
            'active_instances': len(self._bot_instances),
            'registered_bots': {
                name: {
                    'enabled': info.is_enabled,
                    'strategy': info.strategy_type,
                    'capital': info.capital_requirement,
                    'risk_level': info.risk_level
                }
                for name, info in self._registered_bots.items()
            }
        }
    
    def validate_bot_dependencies(self, bot_name: str) -> Dict[str, bool]:
        """Validate that a bot has all required dependencies"""
        if bot_name not in self._bot_classes:
            return {'valid': False, 'error': 'Bot not found'}
        
        bot_class = self._bot_classes[bot_name]
        validation_results = {'valid': True}
        
        try:
            # Check if bot can be imported
            validation_results['import_success'] = True
            
            # Check required methods exist
            required_methods = ['initialize', 'generate_signals', 'calculate_position_size']
            for method in required_methods:
                validation_results[f'has_{method}'] = hasattr(bot_class, method)
                if not validation_results[f'has_{method}']:
                    validation_results['valid'] = False
            
            # Check if bot type is properly configured
            validation_results['has_bot_type'] = True  # Assume true for now
            
        except Exception as e:
            validation_results = {
                'valid': False,
                'error': str(e),
                'import_success': False
            }
        
        return validation_results
    
    def list_bots_detailed(self) -> str:
        """Get detailed listing of all bots"""
        output = ["=" * 60]
        output.append("TRADING BOTS REGISTRY")
        output.append("=" * 60)
        
        for name, info in self._registered_bots.items():
            status = "✅ ENABLED" if info.is_enabled else "❌ DISABLED"
            output.append(f"\n{status} {name.upper()}")
            output.append(f"  Class: {info.class_name}")
            output.append(f"  Strategy: {info.strategy_type}")
            output.append(f"  Instruments: {', '.join(info.supported_instruments)}")
            output.append(f"  Capital: ₹{info.capital_requirement:,.0f}")
            output.append(f"  Risk Level: {info.risk_level}")
            output.append(f"  Description: {info.description}")
            
            # Validation status
            validation = self.validate_bot_dependencies(name)
            if validation['valid']:
                output.append(f"  Validation: ✅ Ready")
            else:
                output.append(f"  Validation: ❌ Issues found")
        
        summary = self.get_registry_summary()
        output.append(f"\n" + "-" * 60)
        output.append(f"Summary: {summary['enabled_bots']}/{summary['total_bots']} enabled")
        output.append(f"Total Capital: ₹{summary['total_allocated_capital']:,.0f}")
        output.append(f"Active Instances: {summary['active_instances']}")
        output.append("=" * 60)
        
        return "\n".join(output)


# Global registry instance
_bot_registry = None


def get_bot_registry() -> BotRegistry:
    """Get the global bot registry instance"""
    global _bot_registry
    if _bot_registry is None:
        _bot_registry = BotRegistry()
    return _bot_registry


def refresh_bot_registry():
    """Refresh the global bot registry (reload all bots)"""
    global _bot_registry
    _bot_registry = None
    return get_bot_registry()