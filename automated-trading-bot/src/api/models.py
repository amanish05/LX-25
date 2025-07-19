"""
Pydantic models for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class BotState(str, Enum):
    """Bot operational states"""
    initialized = "initialized"
    starting = "starting"
    running = "running"
    paused = "paused"
    stopping = "stopping"
    stopped = "stopped"
    error = "error"


class SignalType(str, Enum):
    """Signal types"""
    buy = "BUY"
    sell = "SELL"
    hold = "HOLD"
    exit = "EXIT"


class PositionType(str, Enum):
    """Position types"""
    long = "LONG"
    short = "SHORT"
    short_straddle = "SHORT_STRADDLE"
    iron_condor = "IRON_CONDOR"


# Request Models

class BotActionRequest(BaseModel):
    """Request model for bot actions"""
    action: str = Field(..., description="Action to perform")
    params: Optional[Dict[str, Any]] = Field(default={}, description="Action parameters")


# Response Models

class CapitalInfo(BaseModel):
    """Capital information"""
    initial: float = Field(..., description="Initial capital")
    current: float = Field(..., description="Current capital")
    available: float = Field(..., description="Available capital")
    locked: float = Field(..., description="Locked capital")


class PerformanceInfo(BaseModel):
    """Performance information"""
    total_trades: int = Field(default=0, description="Total number of trades")
    win_rate: float = Field(default=0, description="Win rate percentage")
    total_pnl: float = Field(default=0, description="Total profit/loss")
    max_drawdown: float = Field(default=0, description="Maximum drawdown percentage")


class BotStatusResponse(BaseModel):
    """Bot status response model"""
    name: str = Field(..., description="Bot name")
    type: str = Field(..., description="Bot type")
    state: BotState = Field(..., description="Current bot state")
    enabled: bool = Field(..., description="Whether bot is enabled")
    capital: CapitalInfo = Field(..., description="Capital information")
    positions: int = Field(default=0, description="Number of open positions")
    open_orders: int = Field(default=0, description="Number of open orders")
    performance: PerformanceInfo = Field(..., description="Performance metrics")
    last_health_check: datetime = Field(..., description="Last health check timestamp")
    error_count: int = Field(default=0, description="Current error count")


class SystemStatsInfo(BaseModel):
    """System statistics"""
    total_trades: int = Field(default=0, description="Total trades across all bots")
    successful_trades: int = Field(default=0, description="Successful trades")
    failed_trades: int = Field(default=0, description="Failed trades")
    total_pnl: float = Field(default=0, description="Total P&L")
    active_bots: int = Field(default=0, description="Number of active bots")


class SystemStatusResponse(BaseModel):
    """System status response model"""
    uptime_hours: float = Field(..., description="System uptime in hours")
    is_running: bool = Field(..., description="Whether system is running")
    market_hours: bool = Field(..., description="Whether market is open")
    stats: SystemStatsInfo = Field(..., description="System statistics")
    bots: Dict[str, Dict[str, Any]] = Field(..., description="Bot statuses")
    config: Dict[str, Any] = Field(..., description="System configuration")


class PositionResponse(BaseModel):
    """Position response model"""
    id: int = Field(..., description="Position ID")
    bot_name: str = Field(..., description="Bot name")
    symbol: str = Field(..., description="Symbol")
    exchange: str = Field(..., description="Exchange")
    position_type: PositionType = Field(..., description="Position type")
    quantity: int = Field(..., description="Quantity")
    entry_price: float = Field(..., description="Entry price")
    current_price: Optional[float] = Field(None, description="Current price")
    entry_time: datetime = Field(..., description="Entry time")
    pnl: Optional[float] = Field(None, description="Unrealized P&L")
    status: str = Field(..., description="Position status")


class SignalResponse(BaseModel):
    """Signal response model"""
    id: int = Field(..., description="Signal ID")
    bot_name: str = Field(..., description="Bot name")
    symbol: str = Field(..., description="Symbol")
    exchange: str = Field(..., description="Exchange")
    signal_type: SignalType = Field(..., description="Signal type")
    signal_strength: Optional[float] = Field(None, description="Signal strength (0-1)")
    executed: bool = Field(default=False, description="Whether signal was executed")
    created_at: datetime = Field(..., description="Signal creation time")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional signal data")


class PerformanceResponse(BaseModel):
    """Performance response model"""
    total_pnl: float = Field(..., description="Total P&L")
    total_trades: int = Field(..., description="Total number of trades")
    win_rate: float = Field(..., description="Win rate percentage")
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    daily_pnl: Dict[str, float] = Field(default={}, description="Daily P&L")
    bot_performances: Dict[str, Dict[str, Any]] = Field(
        default={}, 
        description="Individual bot performances"
    )


class OrderResponse(BaseModel):
    """Order response model"""
    order_id: str = Field(..., description="Order ID")
    bot_name: str = Field(..., description="Bot name")
    symbol: str = Field(..., description="Symbol")
    exchange: str = Field(..., description="Exchange")
    action: str = Field(..., description="Buy/Sell")
    quantity: int = Field(..., description="Quantity")
    price: float = Field(..., description="Price")
    order_type: str = Field(..., description="Order type")
    status: str = Field(..., description="Order status")
    placed_at: datetime = Field(..., description="Order placement time")
    executed_at: Optional[datetime] = Field(None, description="Execution time")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    services: Optional[Dict[str, str]] = Field(None, description="Service statuses")