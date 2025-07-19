"""
Base classes for technical indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class IndicatorResult:
    """Standard result structure for indicators"""
    name: str
    value: Union[float, pd.Series, np.ndarray]
    signal: Optional[str] = None  # BUY, SELL, NEUTRAL
    strength: Optional[float] = None  # 0.0 to 1.0
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


class BaseIndicator(ABC):
    """Base class for all technical indicators"""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self.min_periods = self._calculate_min_periods()
        
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> Union[IndicatorResult, List[IndicatorResult]]:
        """Calculate the indicator and return results"""
        pass
    
    @abstractmethod
    def _calculate_min_periods(self) -> int:
        """Calculate minimum periods required for this indicator"""
        pass
    
    def validate_data(self, df: pd.DataFrame, required_columns: List[str] = None) -> bool:
        """Validate that dataframe has required columns and sufficient data"""
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")
        
        if len(df) < self.min_periods:
            raise ValueError(f"Insufficient data. Need at least {self.min_periods} periods")
        
        required = required_columns or ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return True
    
    @staticmethod
    def safe_divide(numerator: Union[pd.Series, np.ndarray], 
                   denominator: Union[pd.Series, np.ndarray], 
                   fill_value: float = 0.0) -> Union[pd.Series, np.ndarray]:
        """Safely divide two series/arrays, handling division by zero"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
            if isinstance(result, pd.Series):
                return result.fillna(fill_value).replace([np.inf, -np.inf], fill_value)
            else:
                return np.nan_to_num(result, nan=fill_value, posinf=fill_value, neginf=fill_value)
    
    @staticmethod
    def normalize(series: pd.Series, min_val: float = 0, max_val: float = 100) -> pd.Series:
        """Normalize series to specified range"""
        s_min, s_max = series.min(), series.max()
        if s_max == s_min:
            return pd.Series([(min_val + max_val) / 2] * len(series), index=series.index)
        
        normalized = (series - s_min) / (s_max - s_min)
        return normalized * (max_val - min_val) + min_val
    
    @staticmethod
    def rolling_rank(series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling rank (percentile) of values"""
        return series.rolling(window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == window else np.nan
        )
    
    def generate_signal(self, value: float, upper_threshold: float, 
                       lower_threshold: float) -> tuple[str, float]:
        """Generate trading signal based on thresholds"""
        if value >= upper_threshold:
            strength = min((value - upper_threshold) / (100 - upper_threshold), 1.0)
            return "SELL", strength
        elif value <= lower_threshold:
            strength = min((lower_threshold - value) / lower_threshold, 1.0)
            return "BUY", strength
        else:
            strength = 1.0 - abs(value - 50) / 50
            return "NEUTRAL", strength