"""
Application Constants
All constants used throughout the application
"""

from enum import Enum
from dataclasses import dataclass


# Time Constants
@dataclass(frozen=True)
class TimeConstants:
    """Time-related constants"""
    MARKET_OPEN_TIME = "09:15"
    MARKET_CLOSE_TIME = "15:30"
    PRE_MARKET_OPEN = "09:00"
    POST_MARKET_CLOSE = "15:30"
    TIMEZONE = "Asia/Kolkata"
    
    # Intervals
    TICK_INTERVAL_SECONDS = 1
    HEALTH_CHECK_INTERVAL = 60  # seconds
    POSITION_SYNC_INTERVAL = 30  # seconds
    METRICS_UPDATE_INTERVAL = 5  # seconds
    OPTION_CHAIN_UPDATE_INTERVAL = 300  # 5 minutes
    CACHE_TTL_SECONDS = 300  # 5 minutes
    
    # Timeouts
    API_TIMEOUT_SECONDS = 30
    ORDER_TIMEOUT_SECONDS = 10
    WEBSOCKET_PING_INTERVAL = 30
    WEBSOCKET_PING_TIMEOUT = 10


# Trading Constants
@dataclass(frozen=True)
class TradingConstants:
    """Trading-related constants"""
    # Lot sizes
    NIFTY_LOT_SIZE = 50
    BANKNIFTY_LOT_SIZE = 25
    FINNIFTY_LOT_SIZE = 25
    
    # Strike intervals
    NIFTY_STRIKE_INTERVAL = 50
    BANKNIFTY_STRIKE_INTERVAL = 100
    FINNIFTY_STRIKE_INTERVAL = 50
    
    # Products
    PRODUCT_MIS = "MIS"  # Intraday
    PRODUCT_NRML = "NRML"  # Normal/Positional
    PRODUCT_CNC = "CNC"  # Cash and Carry
    
    # Order types
    ORDER_TYPE_MARKET = "MARKET"
    ORDER_TYPE_LIMIT = "LIMIT"
    ORDER_TYPE_SL = "SL"
    ORDER_TYPE_SL_M = "SL-M"
    
    # Exchanges
    EXCHANGE_NSE = "NSE"
    EXCHANGE_NFO = "NFO"
    EXCHANGE_BSE = "BSE"
    EXCHANGE_BFO = "BFO"
    EXCHANGE_MCX = "MCX"
    
    # Segments
    SEGMENT_EQUITY = "EQUITY"
    SEGMENT_FO = "FO"
    SEGMENT_COMMODITY = "COMMODITY"
    
    # Default symbols
    DEFAULT_INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY"]
    DEFAULT_STOCKS = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK"]


# Risk Constants
@dataclass(frozen=True)
class RiskConstants:
    """Risk management constants"""
    # Portfolio limits
    MAX_PORTFOLIO_RISK = 0.02  # 2% daily risk
    MAX_POSITION_RISK = 0.005  # 0.5% per position
    MAX_DRAWDOWN_LIMIT = 0.10  # 10% max drawdown
    
    # Position limits
    MAX_POSITIONS_PER_BOT = 10
    MAX_CORRELATION_LIMIT = 0.7
    POSITION_CONCENTRATION_MAX = 0.20  # 20% in single position
    
    # Margin requirements
    MARGIN_UTILIZATION_MAX = 0.80  # 80% max margin usage
    MARGIN_BUFFER = 0.20  # 20% buffer
    
    # Greeks limits (for options)
    PORTFOLIO_DELTA_MAX = 100
    PORTFOLIO_GAMMA_MAX = 50
    PORTFOLIO_VEGA_MAX = 2000
    PORTFOLIO_THETA_MIN = -1000
    
    # Stop loss defaults
    DEFAULT_STOP_LOSS_PERCENT = 0.02  # 2%
    DEFAULT_TRAILING_STOP_PERCENT = 0.01  # 1%
    
    # Slippage
    SLIPPAGE_TOLERANCE_PERCENT = 0.005  # 0.5%
    
    # Order limits
    MAX_ORDER_RETRY_ATTEMPTS = 3
    PARTIAL_FILL_WAIT_SECONDS = 30


# System Constants
@dataclass(frozen=True)
class SystemConstants:
    """System-level constants"""
    # Environments
    ENV_DEVELOPMENT = "development"
    ENV_STAGING = "staging"
    ENV_PRODUCTION = "production"
    
    # Log levels
    LOG_LEVEL_DEBUG = "DEBUG"
    LOG_LEVEL_INFO = "INFO"
    LOG_LEVEL_WARNING = "WARNING"
    LOG_LEVEL_ERROR = "ERROR"
    LOG_LEVEL_CRITICAL = "CRITICAL"
    
    # File paths
    DEFAULT_CONFIG_FILE = "config/config.json"
    DEFAULT_LOG_FILE = "logs/trading_bot.log"
    DEFAULT_DB_PATH = "postgresql://localhost/trading_bot"  # PostgreSQL only - use DATABASE_URL env var
    
    # Performance
    MAX_CONCURRENT_ORDERS = 10
    THREAD_POOL_SIZE = 4
    
    # Monitoring
    METRICS_RETENTION_DAYS = 30
    LOG_RETENTION_DAYS = 7
    MAX_LOG_SIZE_MB = 10
    LOG_BACKUP_COUNT = 5


# API Constants
@dataclass(frozen=True)
class APIConstants:
    """API-related constants"""
    # API versions
    API_VERSION_V1 = "v1"
    CURRENT_API_VERSION = API_VERSION_V1
    
    # HTTP methods
    METHOD_GET = "GET"
    METHOD_POST = "POST"
    METHOD_PUT = "PUT"
    METHOD_DELETE = "DELETE"
    METHOD_PATCH = "PATCH"
    
    # Status codes
    STATUS_OK = 200
    STATUS_CREATED = 201
    STATUS_ACCEPTED = 202
    STATUS_NO_CONTENT = 204
    STATUS_BAD_REQUEST = 400
    STATUS_UNAUTHORIZED = 401
    STATUS_FORBIDDEN = 403
    STATUS_NOT_FOUND = 404
    STATUS_CONFLICT = 409
    STATUS_INTERNAL_ERROR = 500
    STATUS_SERVICE_UNAVAILABLE = 503
    
    # Headers
    HEADER_API_KEY = "X-API-Key"
    HEADER_CONTENT_TYPE = "Content-Type"
    HEADER_ACCEPT = "Accept"
    
    # Content types
    CONTENT_TYPE_JSON = "application/json"
    CONTENT_TYPE_FORM = "application/x-www-form-urlencoded"
    
    # Rate limits
    RATE_LIMIT_PER_MINUTE = 600
    RATE_LIMIT_PER_HOUR = 10000


# Database Constants
@dataclass(frozen=True)
class DatabaseConstants:
    """Database-related constants"""
    # Table names
    TABLE_BOT_POSITIONS = "bot_positions"
    TABLE_BOT_TRADES = "bot_trades"
    TABLE_BOT_PERFORMANCE = "bot_performance"
    TABLE_BOT_SIGNALS = "bot_signals"
    TABLE_BOT_CAPITAL = "bot_capital"
    TABLE_MARKET_DATA_CACHE = "market_data_cache"
    
    # Connection settings
    CONNECTION_POOL_SIZE = 5
    CONNECTION_MAX_OVERFLOW = 10
    CONNECTION_TIMEOUT = 30
    
    # Query limits
    MAX_QUERY_LIMIT = 1000
    DEFAULT_QUERY_LIMIT = 100
    
    # Status values
    STATUS_OPEN = "OPEN"
    STATUS_CLOSED = "CLOSED"
    STATUS_PENDING = "PENDING"
    STATUS_EXECUTED = "EXECUTED"
    STATUS_CANCELLED = "CANCELLED"
    STATUS_REJECTED = "REJECTED"


# Indicator Constants
@dataclass(frozen=True)
class IndicatorConstants:
    """Technical indicator constants"""
    # Moving averages
    EMA_FAST_PERIOD = 5
    EMA_SLOW_PERIOD = 20
    EMA_LONG_PERIOD = 50
    EMA_SUPER_LONG_PERIOD = 200
    
    SMA_SHORT_PERIOD = 10
    SMA_MEDIUM_PERIOD = 20
    SMA_LONG_PERIOD = 50
    
    # Oscillators
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    
    STOCH_K_PERIOD = 14
    STOCH_D_PERIOD = 3
    STOCH_SMOOTH = 3
    
    # Volatility
    ATR_PERIOD = 14
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD_DEV = 2
    
    # Volume
    VOLUME_MA_PERIOD = 20
    OBV_PERIOD = 14
    MFI_PERIOD = 14
    
    # Trend
    ADX_PERIOD = 14
    ADX_TREND_STRENGTH = 25
    
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # Thresholds
    MOMENTUM_THRESHOLD = 0.7
    VOLUME_SURGE_MULTIPLIER = 1.5
    BREAKOUT_CONFIRMATION_BARS = 2


# Order Constants
@dataclass(frozen=True)
class OrderConstants:
    """Order-related constants"""
    # Order actions
    ACTION_BUY = "BUY"
    ACTION_SELL = "SELL"
    
    # Order status
    STATUS_PENDING = "PENDING"
    STATUS_PLACED = "PLACED"
    STATUS_EXECUTED = "EXECUTED"
    STATUS_PARTIALLY_EXECUTED = "PARTIALLY_EXECUTED"
    STATUS_CANCELLED = "CANCELLED"
    STATUS_REJECTED = "REJECTED"
    STATUS_EXPIRED = "EXPIRED"
    
    # Order validity
    VALIDITY_DAY = "DAY"
    VALIDITY_IOC = "IOC"  # Immediate or Cancel
    VALIDITY_GTC = "GTC"  # Good Till Cancelled
    
    # Order attributes
    MIN_ORDER_VALUE = 100  # Minimum order value in currency
    MAX_ORDER_VALUE = 10000000  # Maximum order value
    
    # Execution
    FILL_OR_KILL = "FOK"
    ALL_OR_NONE = "AON"


# Position Constants
@dataclass(frozen=True)
class PositionConstants:
    """Position-related constants"""
    # Position types
    TYPE_LONG = "LONG"
    TYPE_SHORT = "SHORT"
    TYPE_SHORT_STRADDLE = "SHORT_STRADDLE"
    TYPE_SHORT_STRANGLE = "SHORT_STRANGLE"
    TYPE_IRON_CONDOR = "IRON_CONDOR"
    TYPE_BUTTERFLY = "BUTTERFLY"
    TYPE_CALENDAR_SPREAD = "CALENDAR_SPREAD"
    
    # Position status
    STATUS_PENDING = "PENDING"
    STATUS_OPENING = "OPENING"
    STATUS_OPEN = "OPEN"
    STATUS_ADJUSTING = "ADJUSTING"
    STATUS_CLOSING = "CLOSING"
    STATUS_CLOSED = "CLOSED"
    
    # Exit reasons
    EXIT_PROFIT_TARGET = "PROFIT_TARGET"
    EXIT_STOP_LOSS = "STOP_LOSS"
    EXIT_TIME_BASED = "TIME_BASED"
    EXIT_MANUAL = "MANUAL"
    EXIT_RISK_LIMIT = "RISK_LIMIT"
    EXIT_SIGNAL = "SIGNAL"


# Bot Constants
@dataclass(frozen=True)
class BotConstants:
    """Bot-related constants"""
    # Bot types
    TYPE_SHORT_STRADDLE = "short_straddle"
    TYPE_IRON_CONDOR = "iron_condor"
    TYPE_VOLATILITY_EXPANDER = "volatility_expander"
    TYPE_MOMENTUM_RIDER = "momentum_rider"
    TYPE_MEAN_REVERSION = "mean_reversion"
    TYPE_TREND_FOLLOWING = "trend_following"
    TYPE_REVERSAL_TRADER = "reversal_trader"
    
    # Bot categories
    CATEGORY_OPTION_SELLING = "option_selling"
    CATEGORY_OPTION_BUYING = "option_buying"
    
    # Bot category mapping
    BOT_CATEGORIES = {
        "short_straddle": "option_selling",
        "iron_condor": "option_selling",
        "volatility_expander": "option_buying",
        "momentum_rider": "option_buying",
        "mean_reversion": "option_buying",
        "trend_following": "option_buying",
        "reversal_trader": "option_buying"
    }
    
    # Category risk profiles
    CATEGORY_RISK_PROFILES = {
        "option_selling": {
            "max_risk_per_trade": 0.02,  # 2%
            "max_concurrent_positions": 4,
            "win_rate_target": 0.65,  # 65%
            "profit_factor_target": 1.5,
            "max_drawdown": 0.20  # 20%
        },
        "option_buying": {
            "max_risk_per_trade": 0.01,  # 1%
            "max_concurrent_positions": 6,
            "win_rate_target": 0.45,  # 45%
            "profit_factor_target": 1.3,
            "max_drawdown": 0.25  # 25%
        }
    }
    
    # Bot states
    STATE_INITIALIZED = "initialized"
    STATE_STARTING = "starting"
    STATE_RUNNING = "running"
    STATE_PAUSED = "paused"
    STATE_STOPPING = "stopping"
    STATE_STOPPED = "stopped"
    STATE_ERROR = "error"
    
    # Bot priorities
    PRIORITY_HIGH = 1
    PRIORITY_MEDIUM = 5
    PRIORITY_LOW = 10
    
    # Error handling
    MAX_ERROR_COUNT = 10
    ERROR_RESET_INTERVAL = 3600  # 1 hour
    AUTO_RESTART_DELAY = 60  # seconds
    
    # Performance
    MIN_WIN_RATE = 0.3  # 30%
    MIN_SHARPE_RATIO = 0.5
    MAX_CONSECUTIVE_LOSSES = 5


# Signal Constants
@dataclass(frozen=True)
class SignalConstants:
    """Signal-related constants"""
    # Signal types
    TYPE_BUY = "BUY"
    TYPE_SELL = "SELL"
    TYPE_HOLD = "HOLD"
    TYPE_EXIT = "EXIT"
    TYPE_ADJUST = "ADJUST"
    
    # Signal strength
    STRENGTH_WEAK = 0.3
    STRENGTH_MEDIUM = 0.6
    STRENGTH_STRONG = 0.9
    
    # Signal sources
    SOURCE_TECHNICAL = "TECHNICAL"
    SOURCE_FUNDAMENTAL = "FUNDAMENTAL"
    SOURCE_SENTIMENT = "SENTIMENT"
    SOURCE_ML_MODEL = "ML_MODEL"
    SOURCE_COMPOSITE = "COMPOSITE"
    
    # Confidence levels
    CONFIDENCE_LOW = 0.3
    CONFIDENCE_MEDIUM = 0.6
    CONFIDENCE_HIGH = 0.9
    
    # Expiry
    SIGNAL_EXPIRY_MINUTES = 5


# Option Constants
@dataclass(frozen=True)
class OptionConstants:
    """Option-specific constants"""
    # Option types
    TYPE_CALL = "CE"
    TYPE_PUT = "PE"
    
    # Greeks thresholds
    DELTA_NEUTRAL_THRESHOLD = 0.1
    GAMMA_RISK_THRESHOLD = 0.05
    THETA_DECAY_ALERT = -50
    VEGA_RISK_THRESHOLD = 100
    
    # IV parameters
    IV_RANK_HIGH = 75
    IV_RANK_LOW = 25
    IV_PERCENTILE_HIGH = 80
    IV_PERCENTILE_LOW = 20
    
    # DTE (Days to Expiry)
    MIN_DTE_FOR_ENTRY = 15
    MAX_DTE_FOR_ENTRY = 45
    DTE_EXIT_THRESHOLD = 5
    
    # Liquidity
    MIN_OPEN_INTEREST = 1000
    MIN_VOLUME = 100
    MIN_BID_ASK_SPREAD = 0.05


# Create singleton instances
TIME_CONSTANTS = TimeConstants()
TRADING_CONSTANTS = TradingConstants()
RISK_CONSTANTS = RiskConstants()
SYSTEM_CONSTANTS = SystemConstants()
API_CONSTANTS = APIConstants()
DATABASE_CONSTANTS = DatabaseConstants()
INDICATOR_CONSTANTS = IndicatorConstants()
ORDER_CONSTANTS = OrderConstants()
POSITION_CONSTANTS = PositionConstants()
BOT_CONSTANTS = BotConstants()
SIGNAL_CONSTANTS = SignalConstants()
OPTION_CONSTANTS = OptionConstants()