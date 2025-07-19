"""
Technical Indicators Module
Modular system for calculating various technical indicators
"""

from .base import BaseIndicator, IndicatorResult
from .trend import TrendIndicators
from .momentum import MomentumIndicators
from .volatility import VolatilityIndicators
from .volume import VolumeIndicators
from .composite import CompositeIndicators
from .rsi_advanced import AdvancedRSI
from .oscillator_matrix import OscillatorMatrix
from .advanced_confirmation import AdvancedConfirmationSystem
from .signal_validator import SignalValidator

# LuxAlgo Price Action Indicators
from .market_structure import MarketStructure
from .order_blocks import OrderBlocks
from .fair_value_gaps import FairValueGaps
from .liquidity_zones import LiquidityZones
from .pattern_recognition import PatternRecognition
from .price_action_composite import PriceActionComposite

__all__ = [
    'BaseIndicator',
    'IndicatorResult',
    'TrendIndicators',
    'MomentumIndicators',
    'VolatilityIndicators',
    'VolumeIndicators',
    'CompositeIndicators',
    'AdvancedRSI',
    'OscillatorMatrix',
    'AdvancedConfirmationSystem',
    'SignalValidator',
    # Price Action
    'MarketStructure',
    'OrderBlocks',
    'FairValueGaps',
    'LiquidityZones',
    'PatternRecognition',
    'PriceActionComposite'
]