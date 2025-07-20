"""
Bot Selection Module
Smart bot selection based on market regime detection
"""

from .market_regime_detector import MarketRegimeDetector, MarketRegime, BotRecommendation
from .smart_bot_orchestrator import SmartBotOrchestrator

__all__ = [
    'MarketRegimeDetector',
    'MarketRegime', 
    'BotRecommendation',
    'SmartBotOrchestrator'
]