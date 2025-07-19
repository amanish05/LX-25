"""
Trading Parameters
Centralized trading-specific parameters that can be adjusted per bot
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from .constants import *


@dataclass
class OptionSelectionParams:
    """Parameters for option selection"""
    min_dte: int = OPTION_CONSTANTS.MIN_DTE_FOR_ENTRY
    max_dte: int = OPTION_CONSTANTS.MAX_DTE_FOR_ENTRY
    strike_selection_method: str = "ATM"  # ATM, OTM, ITM, DELTA_BASED
    otm_strike_count: int = 2  # For OTM selection
    itm_strike_count: int = 2  # For ITM selection
    delta_target: float = 0.30  # For delta-based selection
    min_open_interest: int = OPTION_CONSTANTS.MIN_OPEN_INTEREST
    min_volume: int = OPTION_CONSTANTS.MIN_VOLUME
    max_bid_ask_spread: float = OPTION_CONSTANTS.MIN_BID_ASK_SPREAD
    prefer_weekly: bool = True
    avoid_event_days: bool = True


@dataclass
class EntryParams:
    """Parameters for trade entry"""
    # IV conditions
    min_iv_rank: float = 50
    max_iv_rank: float = 100
    min_iv_percentile: float = 50
    max_iv_percentile: float = 100
    
    # Time conditions
    entry_start_time: str = "09:30"
    entry_end_time: str = "14:30"
    avoid_first_minutes: int = 15
    avoid_last_minutes: int = 30
    
    # Market conditions
    min_market_trend: float = -0.5  # -1 to 1 scale
    max_market_trend: float = 0.5
    min_volume_ratio: float = 0.8  # Current volume / avg volume
    max_spread_percent: float = 1.0
    
    # Position sizing
    position_size_method: str = "FIXED"  # FIXED, KELLY, VOLATILITY_BASED
    fixed_position_size: int = 1  # Number of lots
    kelly_fraction: float = 0.25
    volatility_scalar: float = 1.0
    
    # Greeks limits
    max_portfolio_delta: float = RISK_CONSTANTS.PORTFOLIO_DELTA_MAX
    max_portfolio_gamma: float = RISK_CONSTANTS.PORTFOLIO_GAMMA_MAX
    max_portfolio_vega: float = RISK_CONSTANTS.PORTFOLIO_VEGA_MAX


@dataclass
class ExitParams:
    """Parameters for trade exit"""
    # Profit targets
    profit_target_percent: float = 50.0  # % of max profit
    profit_target_amount: Optional[float] = None  # Absolute amount
    
    # Stop loss
    stop_loss_percent: float = 100.0  # % of credit received
    stop_loss_amount: Optional[float] = None  # Absolute amount
    max_loss_percent: float = 200.0  # Emergency stop
    
    # Time-based exits
    dte_exit_threshold: int = OPTION_CONSTANTS.DTE_EXIT_THRESHOLD
    exit_time: str = "15:15"
    hold_till_expiry: bool = False
    
    # Dynamic exits
    trailing_stop_enabled: bool = True
    trailing_stop_trigger: float = 30.0  # % profit to activate
    trailing_stop_distance: float = 10.0  # % to trail by
    
    # Adjustment triggers
    delta_adjustment_threshold: float = 0.30
    loss_adjustment_threshold: float = 150.0  # % of credit
    
    # Market condition exits
    exit_on_trend_reversal: bool = True
    exit_on_volatility_spike: bool = True
    volatility_spike_threshold: float = 2.0  # Multiplier


@dataclass
class RiskParams:
    """Risk management parameters"""
    # Position limits
    max_positions: int = RISK_CONSTANTS.MAX_POSITIONS_PER_BOT
    max_positions_per_symbol: int = 2
    max_positions_per_expiry: int = 3
    
    # Capital allocation
    max_capital_per_position: float = 0.10  # 10% of allocated capital
    max_margin_usage: float = RISK_CONSTANTS.MARGIN_UTILIZATION_MAX
    margin_buffer: float = RISK_CONSTANTS.MARGIN_BUFFER
    
    # Risk per trade
    max_risk_per_trade: float = 0.02  # 2% of capital
    max_daily_risk: float = RISK_CONSTANTS.MAX_PORTFOLIO_RISK
    max_weekly_risk: float = 0.05  # 5%
    max_monthly_risk: float = 0.10  # 10%
    
    # Correlation limits
    max_correlation: float = RISK_CONSTANTS.MAX_CORRELATION_LIMIT
    min_time_between_entries: int = 300  # seconds
    
    # Circuit breakers
    max_consecutive_losses: int = BOT_CONSTANTS.MAX_CONSECUTIVE_LOSSES
    daily_loss_limit: float = 0.03  # 3% daily loss limit
    pause_after_limit: int = 3600  # seconds to pause after hitting limit
    
    # Recovery mode
    recovery_mode_threshold: float = -0.05  # -5% drawdown
    recovery_position_reduction: float = 0.5  # Reduce positions by 50%


@dataclass
class ShortStraddleParams:
    """Parameters specific to Short Straddle strategy"""
    entry: EntryParams = field(default_factory=lambda: EntryParams(
        min_iv_rank=70,
        entry_start_time="09:30",
        entry_end_time="14:00",
        position_size_method="FIXED",
        fixed_position_size=1
    ))
    
    exit: ExitParams = field(default_factory=lambda: ExitParams(
        profit_target_percent=50.0,
        stop_loss_percent=100.0,
        trailing_stop_enabled=True,
        trailing_stop_trigger=30.0,
        delta_adjustment_threshold=0.30
    ))
    
    risk: RiskParams = field(default_factory=lambda: RiskParams(
        max_positions=2,
        max_capital_per_position=0.10,
        max_risk_per_trade=0.02
    ))
    
    option_selection: OptionSelectionParams = field(default_factory=lambda: OptionSelectionParams(
        strike_selection_method="ATM",
        min_dte=15,
        max_dte=45
    ))
    
    # Strategy specific
    strike_selection: str = "ATM"  # ATM, SYNTHETIC_ATM
    adjustment_method: str = "ROLL_UNTESTED"  # ROLL_UNTESTED, CLOSE_TESTED, IRON_FLY
    min_credit: float = 200.0  # Minimum credit to receive
    max_strike_skew: float = 100.0  # Max difference between call/put strikes


@dataclass
class IronCondorParams:
    """Parameters specific to Iron Condor strategy"""
    entry: EntryParams = field(default_factory=lambda: EntryParams(
        min_iv_rank=50,
        entry_start_time="09:30",
        entry_end_time="13:00",
        position_size_method="FIXED",
        fixed_position_size=1
    ))
    
    exit: ExitParams = field(default_factory=lambda: ExitParams(
        profit_target_percent=25.0,
        stop_loss_percent=150.0,
        trailing_stop_enabled=False,
        hold_till_expiry=True
    ))
    
    risk: RiskParams = field(default_factory=lambda: RiskParams(
        max_positions=3,
        max_capital_per_position=0.08,
        max_risk_per_trade=0.015
    ))
    
    option_selection: OptionSelectionParams = field(default_factory=lambda: OptionSelectionParams(
        strike_selection_method="DELTA_BASED",
        delta_target=0.20,
        min_dte=30,
        max_dte=60
    ))
    
    # Strategy specific
    short_strike_delta: float = 0.20
    wing_width: int = 5  # Number of strikes for protection
    min_credit: float = 100.0
    max_risk_reward_ratio: float = 3.0
    iron_fly_mode: bool = False  # Convert to Iron Fly if true


@dataclass
class VolatilityExpanderParams:
    """Parameters specific to Volatility Expander strategy"""
    entry: EntryParams = field(default_factory=lambda: EntryParams(
        min_iv_rank=20,
        max_iv_rank=40,
        entry_start_time="09:30",
        entry_end_time="15:00",
        position_size_method="VOLATILITY_BASED"
    ))
    
    exit: ExitParams = field(default_factory=lambda: ExitParams(
        profit_target_percent=100.0,
        stop_loss_percent=50.0,
        trailing_stop_enabled=True,
        trailing_stop_trigger=50.0,
        exit_on_volatility_spike=False  # We want volatility
    ))
    
    risk: RiskParams = field(default_factory=lambda: RiskParams(
        max_positions=5,
        max_capital_per_position=0.05,
        max_risk_per_trade=0.01
    ))
    
    option_selection: OptionSelectionParams = field(default_factory=lambda: OptionSelectionParams(
        strike_selection_method="OTM",
        otm_strike_count=2,
        min_dte=7,
        max_dte=30
    ))
    
    # Strategy specific
    strategy_type: str = "LONG_STRADDLE"  # LONG_STRADDLE, LONG_STRANGLE, CALENDAR
    min_expected_move: float = 2.0  # % expected move
    vega_target: float = 100.0  # Target vega exposure
    event_trading: bool = True  # Trade around events


@dataclass
class MomentumRiderParams:
    """Parameters specific to Momentum Rider strategy"""
    entry: EntryParams = field(default_factory=lambda: EntryParams(
        entry_start_time="09:30",
        entry_end_time="14:30",
        min_market_trend=0.3,
        position_size_method="KELLY"
    ))
    
    exit: ExitParams = field(default_factory=lambda: ExitParams(
        profit_target_percent=200.0,
        stop_loss_percent=30.0,
        trailing_stop_enabled=True,
        trailing_stop_trigger=20.0,
        trailing_stop_distance=10.0,
        exit_on_trend_reversal=True
    ))
    
    risk: RiskParams = field(default_factory=lambda: RiskParams(
        max_positions=4,
        max_capital_per_position=0.10,
        max_risk_per_trade=0.015,
        max_consecutive_losses=3
    ))
    
    option_selection: OptionSelectionParams = field(default_factory=lambda: OptionSelectionParams(
        strike_selection_method="OTM",
        otm_strike_count=1,
        min_dte=0,  # Can trade 0DTE
        max_dte=7
    ))
    
    # Strategy specific
    momentum_lookback: int = 20  # Bars for momentum calculation
    momentum_threshold: float = INDICATOR_CONSTANTS.MOMENTUM_THRESHOLD
    breakout_confirmation_bars: int = INDICATOR_CONSTANTS.BREAKOUT_CONFIRMATION_BARS
    volume_surge_required: bool = True
    min_rsi: float = 60  # For bullish momentum
    max_rsi: float = 40  # For bearish momentum
    use_market_internals: bool = True  # Use advance/decline, etc.


@dataclass
class TradingParameters:
    """Central repository for all trading parameters"""
    # Strategy-specific parameters
    short_straddle: ShortStraddleParams = field(default_factory=ShortStraddleParams)
    iron_condor: IronCondorParams = field(default_factory=IronCondorParams)
    volatility_expander: VolatilityExpanderParams = field(default_factory=VolatilityExpanderParams)
    momentum_rider: MomentumRiderParams = field(default_factory=MomentumRiderParams)
    
    # Global overrides (apply to all strategies)
    global_risk_multiplier: float = 1.0  # Scale all position sizes
    global_stop_loss_multiplier: float = 1.0  # Scale all stop losses
    paper_trading_mode: bool = False  # Paper trade all strategies
    
    def get_strategy_params(self, strategy_name: str) -> Any:
        """Get parameters for a specific strategy"""
        strategy_map = {
            BOT_CONSTANTS.TYPE_SHORT_STRADDLE: self.short_straddle,
            BOT_CONSTANTS.TYPE_IRON_CONDOR: self.iron_condor,
            BOT_CONSTANTS.TYPE_VOLATILITY_EXPANDER: self.volatility_expander,
            BOT_CONSTANTS.TYPE_MOMENTUM_RIDER: self.momentum_rider
        }
        return strategy_map.get(strategy_name)
    
    def apply_global_overrides(self):
        """Apply global overrides to all strategies"""
        for strategy in [self.short_straddle, self.iron_condor, 
                        self.volatility_expander, self.momentum_rider]:
            # Apply risk multiplier
            strategy.risk.max_capital_per_position *= self.global_risk_multiplier
            strategy.risk.max_risk_per_trade *= self.global_risk_multiplier
            
            # Apply stop loss multiplier
            strategy.exit.stop_loss_percent *= self.global_stop_loss_multiplier
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        from dataclasses import asdict
        return {
            'short_straddle': asdict(self.short_straddle),
            'iron_condor': asdict(self.iron_condor),
            'volatility_expander': asdict(self.volatility_expander),
            'momentum_rider': asdict(self.momentum_rider),
            'global_risk_multiplier': self.global_risk_multiplier,
            'global_stop_loss_multiplier': self.global_stop_loss_multiplier,
            'paper_trading_mode': self.paper_trading_mode
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingParameters':
        """Create from dictionary"""
        params = cls()
        
        if 'short_straddle' in data:
            params.short_straddle = ShortStraddleParams(**data['short_straddle'])
        
        if 'iron_condor' in data:
            params.iron_condor = IronCondorParams(**data['iron_condor'])
            
        if 'volatility_expander' in data:
            params.volatility_expander = VolatilityExpanderParams(**data['volatility_expander'])
            
        if 'momentum_rider' in data:
            params.momentum_rider = MomentumRiderParams(**data['momentum_rider'])
        
        params.global_risk_multiplier = data.get('global_risk_multiplier', 1.0)
        params.global_stop_loss_multiplier = data.get('global_stop_loss_multiplier', 1.0)
        params.paper_trading_mode = data.get('paper_trading_mode', False)
        
        return params