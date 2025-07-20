"""
Technical Indicators Module
Modular system for calculating various technical indicators
"""

from .base import BaseIndicator, IndicatorResult
# Removed unused: TrendIndicators, VolumeIndicators, CompositeIndicators
from .momentum import MomentumIndicators
from .volatility import VolatilityIndicators
from .rsi_advanced import AdvancedRSI
from .oscillator_matrix import OscillatorMatrix
from .advanced_confirmation import AdvancedConfirmationSystem
from .signal_validator import SignalValidator
from .reversal_signals import ReversalSignal as ReversalSignals

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
    # Core indicators still in use
    'MomentumIndicators',
    'VolatilityIndicators',
    # Enhanced indicators
    'AdvancedRSI',
    'OscillatorMatrix',
    'AdvancedConfirmationSystem',
    'SignalValidator',
    'ReversalSignals',
    # Price Action (to be ML-enhanced)
    'MarketStructure',
    'OrderBlocks',
    'FairValueGaps',
    'LiquidityZones',
    'PatternRecognition',
    'PriceActionComposite'
]