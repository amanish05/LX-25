"""
Logging configuration for the trading bot system
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Any, Dict
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from rich.logging import RichHandler
from rich.console import Console
import structlog


console = Console()


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'bot_name'):
            log_data['bot_name'] = record.bot_name
        
        if hasattr(record, 'symbol'):
            log_data['symbol'] = record.symbol
        
        if hasattr(record, 'trade_id'):
            log_data['trade_id'] = record.trade_id
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def setup_logging(log_level: str = "INFO", log_file: str = None, 
                 console_output: bool = True, json_format: bool = False) -> logging.Logger:
    """Setup logging configuration"""
    
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if json_format else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if console_output:
        if json_format:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler = RichHandler(
                console=console,
                rich_tracebacks=True,
                tracebacks_show_locals=True
            )
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        if json_format:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler = TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=30
            )
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    return structlog.get_logger()


class TradingLogger:
    """Specialized logger for trading operations"""
    
    def __init__(self, name: str, bot_name: str = None):
        self.logger = structlog.get_logger(name)
        self.bot_name = bot_name
    
    def _add_context(self, **kwargs) -> Dict[str, Any]:
        """Add standard context to log entries"""
        context = {}
        if self.bot_name:
            context['bot_name'] = self.bot_name
        context.update(kwargs)
        return context
    
    def trade_executed(self, symbol: str, action: str, quantity: int, 
                      price: float, order_id: str, **kwargs):
        """Log trade execution"""
        self.logger.info(
            "Trade executed",
            **self._add_context(
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=price,
                order_id=order_id,
                **kwargs
            )
        )
    
    def signal_generated(self, symbol: str, signal_type: str, 
                        strength: float, **kwargs):
        """Log signal generation"""
        self.logger.info(
            "Signal generated",
            **self._add_context(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                **kwargs
            )
        )
    
    def position_opened(self, symbol: str, quantity: int, 
                       entry_price: float, **kwargs):
        """Log position opening"""
        self.logger.info(
            "Position opened",
            **self._add_context(
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                **kwargs
            )
        )
    
    def position_closed(self, symbol: str, quantity: int, 
                       exit_price: float, pnl: float, **kwargs):
        """Log position closing"""
        self.logger.info(
            "Position closed",
            **self._add_context(
                symbol=symbol,
                quantity=quantity,
                exit_price=exit_price,
                pnl=pnl,
                **kwargs
            )
        )
    
    def error(self, message: str, **kwargs):
        """Log error"""
        self.logger.error(message, **self._add_context(**kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning"""
        self.logger.warning(message, **self._add_context(**kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info"""
        self.logger.info(message, **self._add_context(**kwargs))
    
    def debug(self, message: str, **kwargs):
        """Log debug"""
        self.logger.debug(message, **self._add_context(**kwargs))