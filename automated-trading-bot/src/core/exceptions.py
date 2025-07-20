"""
Custom Exception Classes for Automated Trading Bot
Provides specific exception types for better error handling and debugging
"""

from typing import Optional, Dict, Any
from datetime import datetime


class TradingBotException(Exception):
    """Base exception for all trading bot related errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization"""
        return {
            'type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context,
            'timestamp': self.timestamp.isoformat()
        }


# ========================================
# Trading-Specific Exceptions
# ========================================

class TradingOperationException(TradingBotException):
    """Exceptions related to trading operations"""
    pass


class InsufficientFundsException(TradingOperationException):
    """Raised when insufficient funds for trading operation"""
    
    def __init__(self, required_amount: float, available_amount: float, 
                 symbol: str = None):
        self.required_amount = required_amount
        self.available_amount = available_amount
        self.symbol = symbol
        
        message = (f"Insufficient funds: Required ₹{required_amount:,.2f}, "
                  f"Available ₹{available_amount:,.2f}")
        if symbol:
            message += f" for {symbol}"
            
        super().__init__(
            message=message,
            error_code="INSUFFICIENT_FUNDS",
            context={
                'required_amount': required_amount,
                'available_amount': available_amount,
                'symbol': symbol
            }
        )


class OrderExecutionException(TradingOperationException):
    """Raised when order execution fails"""
    
    def __init__(self, order_id: str, reason: str, order_details: Dict[str, Any] = None):
        self.order_id = order_id
        self.reason = reason
        self.order_details = order_details or {}
        
        super().__init__(
            message=f"Order execution failed: {reason} (Order ID: {order_id})",
            error_code="ORDER_EXECUTION_FAILED",
            context={
                'order_id': order_id,
                'reason': reason,
                'order_details': order_details
            }
        )


class PositionManagementException(TradingOperationException):
    """Raised when position management operations fail"""
    
    def __init__(self, symbol: str, operation: str, reason: str):
        self.symbol = symbol
        self.operation = operation
        self.reason = reason
        
        super().__init__(
            message=f"Position management failed for {symbol}: {operation} - {reason}",
            error_code="POSITION_MANAGEMENT_FAILED",
            context={
                'symbol': symbol,
                'operation': operation,
                'reason': reason
            }
        )


class RiskManagementException(TradingOperationException):
    """Raised when risk management rules are violated"""
    
    def __init__(self, rule_violated: str, current_value: float, 
                 limit_value: float, symbol: str = None):
        self.rule_violated = rule_violated
        self.current_value = current_value
        self.limit_value = limit_value
        self.symbol = symbol
        
        message = (f"Risk management violation: {rule_violated} "
                  f"(Current: {current_value}, Limit: {limit_value})")
        if symbol:
            message += f" for {symbol}"
            
        super().__init__(
            message=message,
            error_code="RISK_MANAGEMENT_VIOLATION",
            context={
                'rule_violated': rule_violated,
                'current_value': current_value,
                'limit_value': limit_value,
                'symbol': symbol
            }
        )


# ========================================
# Data-Related Exceptions
# ========================================

class DataException(TradingBotException):
    """Base exception for data-related errors"""
    pass


class DataCollectionException(DataException):
    """Raised when data collection fails"""
    
    def __init__(self, source: str, symbol: str, reason: str, 
                 timeframe: str = None):
        self.source = source
        self.symbol = symbol
        self.reason = reason
        self.timeframe = timeframe
        
        message = f"Data collection failed from {source} for {symbol}: {reason}"
        if timeframe:
            message += f" (Timeframe: {timeframe})"
            
        super().__init__(
            message=message,
            error_code="DATA_COLLECTION_FAILED",
            context={
                'source': source,
                'symbol': symbol,
                'reason': reason,
                'timeframe': timeframe
            }
        )


class DataValidationException(DataException):
    """Raised when data validation fails"""
    
    def __init__(self, field: str, expected: str, actual: str, 
                 data_type: str = None):
        self.field = field
        self.expected = expected
        self.actual = actual
        self.data_type = data_type
        
        message = f"Data validation failed for {field}: Expected {expected}, got {actual}"
        if data_type:
            message += f" in {data_type}"
            
        super().__init__(
            message=message,
            error_code="DATA_VALIDATION_FAILED",
            context={
                'field': field,
                'expected': expected,
                'actual': actual,
                'data_type': data_type
            }
        )


class MarketDataException(DataCollectionException):
    """Raised when market data operations fail"""
    
    def __init__(self, symbol: str, operation: str, reason: str):
        super().__init__(
            source="market_data",
            symbol=symbol,
            reason=f"{operation}: {reason}"
        )
        self.operation = operation


# ========================================
# API/Network Exceptions
# ========================================

class APIException(TradingBotException):
    """Base exception for API-related errors"""
    pass


class OpenAlgoException(APIException):
    """Raised when OpenAlgo API operations fail"""
    
    def __init__(self, endpoint: str, status_code: Optional[int] = None, 
                 response: Optional[str] = None, operation: str = None):
        self.endpoint = endpoint
        self.status_code = status_code
        self.response = response
        self.operation = operation
        
        message = f"OpenAlgo API error"
        if operation:
            message += f" during {operation}"
        message += f" at {endpoint}"
        if status_code:
            message += f" (Status: {status_code})"
        if response:
            message += f": {response}"
            
        super().__init__(
            message=message,
            error_code="OPENALGO_API_ERROR",
            context={
                'endpoint': endpoint,
                'status_code': status_code,
                'response': response,
                'operation': operation
            }
        )


class WebSocketConnectionException(APIException):
    """Raised when WebSocket connection fails"""
    
    def __init__(self, url: str, reason: str, connection_attempt: int = 1):
        self.url = url
        self.reason = reason
        self.connection_attempt = connection_attempt
        
        super().__init__(
            message=f"WebSocket connection failed to {url}: {reason} (Attempt {connection_attempt})",
            error_code="WEBSOCKET_CONNECTION_FAILED",
            context={
                'url': url,
                'reason': reason,
                'connection_attempt': connection_attempt
            }
        )


class APIRateLimitException(APIException):
    """Raised when API rate limit is exceeded"""
    
    def __init__(self, api_name: str, limit: int, window: str, 
                 retry_after: Optional[int] = None):
        self.api_name = api_name
        self.limit = limit
        self.window = window
        self.retry_after = retry_after
        
        message = f"Rate limit exceeded for {api_name}: {limit} requests per {window}"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
            
        super().__init__(
            message=message,
            error_code="API_RATE_LIMIT_EXCEEDED",
            context={
                'api_name': api_name,
                'limit': limit,
                'window': window,
                'retry_after': retry_after
            }
        )


class AuthenticationException(APIException):
    """Raised when API authentication fails"""
    
    def __init__(self, api_name: str, reason: str = None):
        self.api_name = api_name
        self.reason = reason
        
        message = f"Authentication failed for {api_name}"
        if reason:
            message += f": {reason}"
            
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_FAILED",
            context={
                'api_name': api_name,
                'reason': reason
            }
        )


# ========================================
# Database Exceptions
# ========================================

class DatabaseException(TradingBotException):
    """Base exception for database-related errors"""
    pass


class ConnectionException(DatabaseException):
    """Raised when database connection fails"""
    
    def __init__(self, database_type: str, host: str, database: str, 
                 reason: str = None):
        self.database_type = database_type
        self.host = host
        self.database = database
        self.reason = reason
        
        message = f"Database connection failed: {database_type} at {host}/{database}"
        if reason:
            message += f" - {reason}"
            
        super().__init__(
            message=message,
            error_code="DATABASE_CONNECTION_FAILED",
            context={
                'database_type': database_type,
                'host': host,
                'database': database,
                'reason': reason
            }
        )


class QueryException(DatabaseException):
    """Raised when database query fails"""
    
    def __init__(self, query_type: str, table: str = None, reason: str = None,
                 query: str = None):
        self.query_type = query_type
        self.table = table
        self.reason = reason
        self.query = query
        
        message = f"Database query failed: {query_type}"
        if table:
            message += f" on {table}"
        if reason:
            message += f" - {reason}"
            
        super().__init__(
            message=message,
            error_code="DATABASE_QUERY_FAILED",
            context={
                'query_type': query_type,
                'table': table,
                'reason': reason,
                'query': query[:200] if query else None  # Truncate long queries
            }
        )


class MigrationException(DatabaseException):
    """Raised when database migration fails"""
    
    def __init__(self, migration_name: str, direction: str, reason: str):
        self.migration_name = migration_name
        self.direction = direction  # 'up' or 'down'
        self.reason = reason
        
        super().__init__(
            message=f"Database migration failed: {migration_name} ({direction}) - {reason}",
            error_code="DATABASE_MIGRATION_FAILED",
            context={
                'migration_name': migration_name,
                'direction': direction,
                'reason': reason
            }
        )


# ========================================
# ML/AI Model Exceptions
# ========================================

class MLModelException(TradingBotException):
    """Base exception for ML/AI model errors"""
    pass


class ModelLoadingException(MLModelException):
    """Raised when ML model loading fails"""
    
    def __init__(self, model_name: str, model_path: str, reason: str):
        self.model_name = model_name
        self.model_path = model_path
        self.reason = reason
        
        super().__init__(
            message=f"Model loading failed: {model_name} from {model_path} - {reason}",
            error_code="MODEL_LOADING_FAILED",
            context={
                'model_name': model_name,
                'model_path': model_path,
                'reason': reason
            }
        )


class ModelPredictionException(MLModelException):
    """Raised when ML model prediction fails"""
    
    def __init__(self, model_name: str, input_shape: Optional[tuple] = None, 
                 reason: str = None):
        self.model_name = model_name
        self.input_shape = input_shape
        self.reason = reason
        
        message = f"Model prediction failed for {model_name}"
        if input_shape:
            message += f" with input shape {input_shape}"
        if reason:
            message += f": {reason}"
            
        super().__init__(
            message=message,
            error_code="MODEL_PREDICTION_FAILED",
            context={
                'model_name': model_name,
                'input_shape': input_shape,
                'reason': reason
            }
        )


class ModelTrainingException(MLModelException):
    """Raised when ML model training fails"""
    
    def __init__(self, model_name: str, epoch: Optional[int] = None, 
                 reason: str = None):
        self.model_name = model_name
        self.epoch = epoch
        self.reason = reason
        
        message = f"Model training failed for {model_name}"
        if epoch is not None:
            message += f" at epoch {epoch}"
        if reason:
            message += f": {reason}"
            
        super().__init__(
            message=message,
            error_code="MODEL_TRAINING_FAILED",
            context={
                'model_name': model_name,
                'epoch': epoch,
                'reason': reason
            }
        )


# ========================================
# Configuration Exceptions
# ========================================

class ConfigurationException(TradingBotException):
    """Base exception for configuration-related errors"""
    pass


class InvalidConfigurationException(ConfigurationException):
    """Raised when configuration is invalid"""
    
    def __init__(self, config_key: str, value: Any, expected_type: str = None,
                 validation_rule: str = None):
        self.config_key = config_key
        self.value = value
        self.expected_type = expected_type
        self.validation_rule = validation_rule
        
        message = f"Invalid configuration for '{config_key}': {value}"
        if expected_type:
            message += f" (Expected type: {expected_type})"
        if validation_rule:
            message += f" (Rule: {validation_rule})"
            
        super().__init__(
            message=message,
            error_code="INVALID_CONFIGURATION",
            context={
                'config_key': config_key,
                'value': str(value),
                'expected_type': expected_type,
                'validation_rule': validation_rule
            }
        )


class MissingConfigurationException(ConfigurationException):
    """Raised when required configuration is missing"""
    
    def __init__(self, config_key: str, config_file: str = None):
        self.config_key = config_key
        self.config_file = config_file
        
        message = f"Missing required configuration: '{config_key}'"
        if config_file:
            message += f" in {config_file}"
            
        super().__init__(
            message=message,
            error_code="MISSING_CONFIGURATION",
            context={
                'config_key': config_key,
                'config_file': config_file
            }
        )


# ========================================
# Utility Functions
# ========================================

def handle_exception(exception: Exception, logger, context: Dict[str, Any] = None) -> TradingBotException:
    """
    Convert generic exceptions to specific trading bot exceptions
    
    Args:
        exception: The original exception
        logger: Logger instance
        context: Additional context information
        
    Returns:
        TradingBotException: Specific exception type
    """
    context = context or {}
    
    # Map common exception types to specific trading bot exceptions
    if isinstance(exception, ValueError):
        return DataValidationException(
            field=context.get('field', 'unknown'),
            expected=context.get('expected', 'valid value'),
            actual=str(exception),
            data_type=context.get('data_type')
        )
    
    elif isinstance(exception, ConnectionError):
        return ConnectionException(
            database_type=context.get('database_type', 'unknown'),
            host=context.get('host', 'unknown'),
            database=context.get('database', 'unknown'),
            reason=str(exception)
        )
    
    elif isinstance(exception, FileNotFoundError):
        return ModelLoadingException(
            model_name=context.get('model_name', 'unknown'),
            model_path=str(exception.filename) if hasattr(exception, 'filename') else 'unknown',
            reason=str(exception)
        )
    
    else:
        # Log the original exception and return generic trading bot exception
        logger.error(f"Unhandled exception type: {type(exception).__name__}: {exception}")
        return TradingBotException(
            message=str(exception),
            error_code="UNHANDLED_EXCEPTION",
            context={'original_type': type(exception).__name__, **context}
        )


def get_exception_hierarchy() -> Dict[str, list]:
    """Get the exception hierarchy for documentation/debugging"""
    return {
        'TradingBotException': [
            'TradingOperationException',
            'DataException', 
            'APIException',
            'DatabaseException',
            'MLModelException',
            'ConfigurationException'
        ],
        'TradingOperationException': [
            'InsufficientFundsException',
            'OrderExecutionException', 
            'PositionManagementException',
            'RiskManagementException'
        ],
        'DataException': [
            'DataCollectionException',
            'DataValidationException',
            'MarketDataException'
        ],
        'APIException': [
            'OpenAlgoException',
            'WebSocketConnectionException',
            'APIRateLimitException',
            'AuthenticationException'
        ],
        'DatabaseException': [
            'ConnectionException',
            'QueryException',
            'MigrationException'
        ],
        'MLModelException': [
            'ModelLoadingException',
            'ModelPredictionException',
            'ModelTrainingException'
        ],
        'ConfigurationException': [
            'InvalidConfigurationException',
            'MissingConfigurationException'
        ]
    }