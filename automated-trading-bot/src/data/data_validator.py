"""
Data Validation Pipeline for Trading Bot System
Ensures data quality, handles corporate actions, and detects anomalies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
import statistics
from scipy import stats


class DataQualityIssue(Enum):
    """Types of data quality issues"""
    MISSING_DATA = "missing_data"
    OUTLIER = "outlier"
    GAP = "gap"
    DUPLICATE = "duplicate"
    CORPORATE_ACTION = "corporate_action"
    PRICE_SPIKE = "price_spike"
    VOLUME_ANOMALY = "volume_anomaly"
    SPREAD_ANOMALY = "spread_anomaly"
    TIMESTAMP_ERROR = "timestamp_error"


@dataclass
class ValidationResult:
    """Container for validation results"""
    is_valid: bool
    quality_score: float  # 0.0 to 1.0
    issues: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    corrections_applied: List[str] = field(default_factory=list)


@dataclass
class CorporateAction:
    """Corporate action information"""
    date: datetime
    action_type: str  # SPLIT, DIVIDEND, BONUS, MERGER
    adjustment_factor: float
    symbol: str
    description: str = ""


class DataValidator:
    """
    Comprehensive data validation pipeline that:
    - Detects and handles gaps in data
    - Identifies outliers and anomalies
    - Handles corporate actions
    - Validates data consistency
    - Provides quality scoring
    """
    
    def __init__(self, 
                 outlier_threshold: float = 3.0,
                 max_gap_minutes: int = 5,
                 price_spike_threshold: float = 0.1,
                 volume_spike_threshold: float = 10.0):
        """Initialize data validator
        
        Args:
            outlier_threshold: Z-score threshold for outlier detection
            max_gap_minutes: Maximum allowed gap in minutes
            price_spike_threshold: Maximum allowed price change (fraction)
            volume_spike_threshold: Volume spike detection threshold
        """
        self.logger = logging.getLogger(__name__)
        self.outlier_threshold = outlier_threshold
        self.max_gap_minutes = max_gap_minutes
        self.price_spike_threshold = price_spike_threshold
        self.volume_spike_threshold = volume_spike_threshold
        
        # Corporate actions database (in production, this would be from DB)
        self.corporate_actions: Dict[str, List[CorporateAction]] = {}
        
    def validate_ohlcv_data(self, 
                           df: pd.DataFrame, 
                           symbol: str,
                           timeframe: str,
                           auto_correct: bool = True) -> ValidationResult:
        """Validate OHLCV data comprehensively
        
        Args:
            df: OHLCV DataFrame with datetime index
            symbol: Trading symbol
            timeframe: Data timeframe (1min, 5min, etc.)
            auto_correct: Apply automatic corrections
            
        Returns:
            ValidationResult with details
        """
        self.logger.info(f"Validating {len(df)} records for {symbol} ({timeframe})")
        
        result = ValidationResult(is_valid=True, quality_score=1.0)
        df_validated = df.copy()
        
        # 1. Check basic structure
        structure_issues = self._validate_structure(df_validated)
        if structure_issues:
            result.issues.extend(structure_issues)
            result.quality_score *= 0.8
        
        # 2. Check for missing data
        missing_issues = self._check_missing_data(df_validated, timeframe)
        if missing_issues:
            result.issues.extend(missing_issues)
            result.quality_score *= 0.9
            
            if auto_correct:
                df_validated = self._fill_missing_data(df_validated, timeframe)
                result.corrections_applied.append("filled_missing_data")
        
        # 3. Check for duplicates
        duplicate_issues = self._check_duplicates(df_validated)
        if duplicate_issues:
            result.issues.extend(duplicate_issues)
            result.quality_score *= 0.95
            
            if auto_correct:
                df_validated = self._remove_duplicates(df_validated)
                result.corrections_applied.append("removed_duplicates")
        
        # 4. Check for outliers
        outlier_issues = self._detect_outliers(df_validated)
        if outlier_issues:
            result.issues.extend(outlier_issues)
            result.quality_score *= 0.9
            
            if auto_correct:
                df_validated = self._handle_outliers(df_validated, outlier_issues)
                result.corrections_applied.append("handled_outliers")
        
        # 5. Check for price spikes
        spike_issues = self._detect_price_spikes(df_validated)
        if spike_issues:
            result.issues.extend(spike_issues)
            result.quality_score *= 0.85
        
        # 6. Check for volume anomalies
        volume_issues = self._detect_volume_anomalies(df_validated)
        if volume_issues:
            result.issues.extend(volume_issues)
            result.quality_score *= 0.95
        
        # 7. Apply corporate actions
        if symbol in self.corporate_actions:
            df_validated, applied_actions = self._apply_corporate_actions(
                df_validated, symbol, auto_correct
            )
            if applied_actions:
                result.corrections_applied.extend(applied_actions)
        
        # 8. Validate OHLC relationships
        ohlc_issues = self._validate_ohlc_relationships(df_validated)
        if ohlc_issues:
            result.issues.extend(ohlc_issues)
            result.quality_score *= 0.9
            
            if auto_correct:
                df_validated = self._fix_ohlc_relationships(df_validated)
                result.corrections_applied.append("fixed_ohlc_relationships")
        
        # Calculate final statistics
        result.stats = self._calculate_data_stats(df_validated)
        result.is_valid = result.quality_score >= 0.7
        
        # Store validated data
        df.update(df_validated)
        
        self.logger.info(f"Validation complete. Quality score: {result.quality_score:.2f}")
        return result
    
    def validate_tick_data(self,
                          ticks: List[Dict[str, Any]],
                          symbol: str) -> ValidationResult:
        """Validate tick-by-tick data
        
        Args:
            ticks: List of tick data dictionaries
            symbol: Trading symbol
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True, quality_score=1.0)
        
        if not ticks:
            result.is_valid = False
            result.quality_score = 0.0
            return result
        
        # Convert to DataFrame for easier processing
        tick_df = pd.DataFrame(ticks)
        tick_df['timestamp'] = pd.to_datetime(tick_df['timestamp'])
        tick_df.sort_values('timestamp', inplace=True)
        
        # 1. Check timestamp consistency
        timestamp_issues = self._validate_tick_timestamps(tick_df)
        if timestamp_issues:
            result.issues.extend(timestamp_issues)
            result.quality_score *= 0.9
        
        # 2. Check bid-ask spread validity
        spread_issues = self._validate_bid_ask_spreads(tick_df)
        if spread_issues:
            result.issues.extend(spread_issues)
            result.quality_score *= 0.95
        
        # 3. Check price continuity
        price_issues = self._validate_price_continuity(tick_df)
        if price_issues:
            result.issues.extend(price_issues)
            result.quality_score *= 0.9
        
        # 4. Detect unusual trading patterns
        pattern_issues = self._detect_unusual_patterns(tick_df)
        if pattern_issues:
            result.issues.extend(pattern_issues)
            result.quality_score *= 0.95
        
        result.stats = {
            'total_ticks': len(tick_df),
            'avg_spread': (tick_df['ask'] - tick_df['bid']).mean(),
            'total_volume': tick_df['volume'].sum(),
            'price_range': tick_df['price'].max() - tick_df['price'].min()
        }
        
        result.is_valid = result.quality_score >= 0.7
        return result
    
    def _validate_structure(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate basic DataFrame structure"""
        issues = []
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            issues.append({
                'type': DataQualityIssue.MISSING_DATA,
                'severity': 'high',
                'description': f"Missing columns: {missing_columns}"
            })
        
        # Check index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append({
                'type': DataQualityIssue.TIMESTAMP_ERROR,
                'severity': 'high',
                'description': "Index is not DatetimeIndex"
            })
        
        # Check data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                issues.append({
                    'type': DataQualityIssue.TIMESTAMP_ERROR,
                    'severity': 'medium',
                    'description': f"Column {col} is not numeric"
                })
        
        return issues
    
    def _check_missing_data(self, df: pd.DataFrame, timeframe: str) -> List[Dict[str, Any]]:
        """Check for gaps and missing data"""
        issues = []
        
        # Check for NaN values
        nan_counts = df.isnull().sum()
        if nan_counts.any():
            issues.append({
                'type': DataQualityIssue.MISSING_DATA,
                'severity': 'medium',
                'description': f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}"
            })
        
        # Check for time gaps
        if len(df) > 1:
            time_diffs = df.index.to_series().diff()
            
            # Expected frequency
            freq_map = {
                '1min': timedelta(minutes=1),
                '5min': timedelta(minutes=5),
                '15min': timedelta(minutes=15),
                '30min': timedelta(minutes=30),
                '1hour': timedelta(hours=1),
                '1day': timedelta(days=1)
            }
            
            expected_freq = freq_map.get(timeframe, timedelta(minutes=5))
            
            # Find gaps larger than expected
            gaps = time_diffs[time_diffs > expected_freq * 1.5]
            
            if len(gaps) > 0:
                for idx, gap in gaps.items():
                    issues.append({
                        'type': DataQualityIssue.GAP,
                        'severity': 'medium',
                        'description': f"Time gap of {gap} at {idx}",
                        'timestamp': idx.isoformat()
                    })
        
        return issues
    
    def _check_duplicates(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for duplicate timestamps"""
        issues = []
        
        duplicates = df.index.duplicated()
        if duplicates.any():
            duplicate_times = df.index[duplicates].unique()
            issues.append({
                'type': DataQualityIssue.DUPLICATE,
                'severity': 'medium',
                'description': f"Found {len(duplicate_times)} duplicate timestamps",
                'timestamps': [t.isoformat() for t in duplicate_times[:10]]  # First 10
            })
        
        return issues
    
    def _detect_outliers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect statistical outliers using multiple methods"""
        issues = []
        
        # Z-score method for returns
        if 'close' in df.columns:
            returns = df['close'].pct_change().dropna()
            z_scores = np.abs(stats.zscore(returns))
            outliers = returns[z_scores > self.outlier_threshold]
            
            if len(outliers) > 0:
                for idx, value in outliers.items():
                    issues.append({
                        'type': DataQualityIssue.OUTLIER,
                        'severity': 'low',
                        'description': f"Return outlier: {value:.4%}",
                        'timestamp': idx.isoformat(),
                        'field': 'returns',
                        'value': value
                    })
        
        # IQR method for volume
        if 'volume' in df.columns:
            Q1 = df['volume'].quantile(0.25)
            Q3 = df['volume'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            volume_outliers = df[(df['volume'] < lower_bound) | (df['volume'] > upper_bound)]
            
            if len(volume_outliers) > 0:
                for idx, row in volume_outliers.iterrows():
                    issues.append({
                        'type': DataQualityIssue.VOLUME_ANOMALY,
                        'severity': 'low',
                        'description': f"Volume outlier: {row['volume']}",
                        'timestamp': idx.isoformat(),
                        'field': 'volume',
                        'value': row['volume']
                    })
        
        return issues
    
    def _detect_price_spikes(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect sudden price spikes"""
        issues = []
        
        if 'close' in df.columns:
            price_changes = df['close'].pct_change().abs()
            spikes = price_changes[price_changes > self.price_spike_threshold]
            
            if len(spikes) > 0:
                for idx, change in spikes.items():
                    issues.append({
                        'type': DataQualityIssue.PRICE_SPIKE,
                        'severity': 'medium',
                        'description': f"Price spike: {change:.2%}",
                        'timestamp': idx.isoformat(),
                        'change': change
                    })
        
        return issues
    
    def _detect_volume_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect volume anomalies"""
        issues = []
        
        if 'volume' in df.columns:
            # Rolling average volume
            vol_ma = df['volume'].rolling(window=20, min_periods=1).mean()
            vol_ratio = df['volume'] / vol_ma
            
            anomalies = vol_ratio[vol_ratio > self.volume_spike_threshold]
            
            if len(anomalies) > 0:
                for idx, ratio in anomalies.items():
                    issues.append({
                        'type': DataQualityIssue.VOLUME_ANOMALY,
                        'severity': 'low',
                        'description': f"Volume spike: {ratio:.1f}x average",
                        'timestamp': idx.isoformat(),
                        'ratio': ratio
                    })
        
        return issues
    
    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate OHLC price relationships"""
        issues = []
        
        # High should be >= Low
        invalid_hl = df[df['high'] < df['low']]
        if len(invalid_hl) > 0:
            for idx, row in invalid_hl.iterrows():
                issues.append({
                    'type': DataQualityIssue.OUTLIER,
                    'severity': 'high',
                    'description': f"High < Low: H={row['high']}, L={row['low']}",
                    'timestamp': idx.isoformat()
                })
        
        # Open and Close should be between High and Low
        invalid_oc = df[
            (df['open'] > df['high']) | (df['open'] < df['low']) |
            (df['close'] > df['high']) | (df['close'] < df['low'])
        ]
        
        if len(invalid_oc) > 0:
            for idx, row in invalid_oc.iterrows():
                issues.append({
                    'type': DataQualityIssue.OUTLIER,
                    'severity': 'high',
                    'description': "OHLC relationship violation",
                    'timestamp': idx.isoformat(),
                    'ohlc': {
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close']
                    }
                })
        
        return issues
    
    def _fill_missing_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Fill missing data using appropriate methods"""
        df_filled = df.copy()
        
        # Forward fill for small gaps
        df_filled = df_filled.fillna(method='ffill', limit=5)
        
        # Interpolate for remaining NaNs
        numeric_columns = df_filled.select_dtypes(include=[np.number]).columns
        df_filled[numeric_columns] = df_filled[numeric_columns].interpolate(method='linear')
        
        # Fill any remaining NaNs with column means
        df_filled = df_filled.fillna(df_filled.mean())
        
        return df_filled
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate timestamps, keeping the first occurrence"""
        return df[~df.index.duplicated(keep='first')]
    
    def _handle_outliers(self, df: pd.DataFrame, outlier_issues: List[Dict]) -> pd.DataFrame:
        """Handle outliers using appropriate methods"""
        df_cleaned = df.copy()
        
        # Group outliers by field
        for issue in outlier_issues:
            if issue['type'] == DataQualityIssue.OUTLIER and 'field' in issue:
                timestamp = pd.Timestamp(issue['timestamp'])
                field = issue['field']
                
                if field == 'returns' and timestamp in df_cleaned.index:
                    # Cap extreme returns
                    idx = df_cleaned.index.get_loc(timestamp)
                    if idx > 0:
                        # Use previous close for extreme moves
                        prev_close = df_cleaned.iloc[idx-1]['close']
                        max_change = prev_close * self.price_spike_threshold
                        
                        current_close = df_cleaned.loc[timestamp, 'close']
                        if abs(current_close - prev_close) > max_change:
                            # Cap the change
                            direction = 1 if current_close > prev_close else -1
                            df_cleaned.loc[timestamp, 'close'] = prev_close + direction * max_change
                            
                            # Adjust OHLC accordingly
                            df_cleaned.loc[timestamp, 'high'] = min(df_cleaned.loc[timestamp, 'high'], 
                                                                   df_cleaned.loc[timestamp, 'close'])
                            df_cleaned.loc[timestamp, 'low'] = max(df_cleaned.loc[timestamp, 'low'], 
                                                                  df_cleaned.loc[timestamp, 'close'])
        
        return df_cleaned
    
    def _fix_ohlc_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix invalid OHLC relationships"""
        df_fixed = df.copy()
        
        # Fix High < Low
        mask = df_fixed['high'] < df_fixed['low']
        df_fixed.loc[mask, ['high', 'low']] = df_fixed.loc[mask, ['low', 'high']].values
        
        # Fix Open/Close outside High/Low
        df_fixed['open'] = df_fixed[['open', 'high', 'low']].apply(
            lambda x: max(min(x['open'], x['high']), x['low']), axis=1
        )
        df_fixed['close'] = df_fixed[['close', 'high', 'low']].apply(
            lambda x: max(min(x['close'], x['high']), x['low']), axis=1
        )
        
        return df_fixed
    
    def _apply_corporate_actions(self, 
                                df: pd.DataFrame, 
                                symbol: str,
                                auto_correct: bool) -> Tuple[pd.DataFrame, List[str]]:
        """Apply corporate action adjustments"""
        df_adjusted = df.copy()
        applied_actions = []
        
        actions = self.corporate_actions.get(symbol, [])
        
        for action in actions:
            if action.date in df_adjusted.index:
                if auto_correct:
                    # Apply adjustment to all price columns before the action date
                    price_columns = ['open', 'high', 'low', 'close']
                    mask = df_adjusted.index < action.date
                    
                    for col in price_columns:
                        if col in df_adjusted.columns:
                            df_adjusted.loc[mask, col] *= action.adjustment_factor
                    
                    # Adjust volume inversely
                    if 'volume' in df_adjusted.columns:
                        df_adjusted.loc[mask, 'volume'] /= action.adjustment_factor
                    
                    applied_actions.append(f"{action.action_type} on {action.date.date()}")
        
        return df_adjusted, applied_actions
    
    def _validate_tick_timestamps(self, tick_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate tick data timestamps"""
        issues = []
        
        # Check for timestamps in the future
        current_time = datetime.now()
        future_ticks = tick_df[tick_df['timestamp'] > current_time]
        
        if len(future_ticks) > 0:
            issues.append({
                'type': DataQualityIssue.TIMESTAMP_ERROR,
                'severity': 'high',
                'description': f"Found {len(future_ticks)} ticks with future timestamps"
            })
        
        # Check for microsecond precision
        if len(tick_df) > 0:
            # Check timestamp ordering
            time_diffs = tick_df['timestamp'].diff()
            backwards = time_diffs[time_diffs < timedelta(0)]
            
            if len(backwards) > 0:
                issues.append({
                    'type': DataQualityIssue.TIMESTAMP_ERROR,
                    'severity': 'high',
                    'description': f"Found {len(backwards)} backwards timestamps"
                })
        
        return issues
    
    def _validate_bid_ask_spreads(self, tick_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate bid-ask spreads"""
        issues = []
        
        if 'bid' in tick_df.columns and 'ask' in tick_df.columns:
            # Check for negative spreads
            negative_spreads = tick_df[tick_df['bid'] > tick_df['ask']]
            
            if len(negative_spreads) > 0:
                issues.append({
                    'type': DataQualityIssue.SPREAD_ANOMALY,
                    'severity': 'high',
                    'description': f"Found {len(negative_spreads)} negative spreads"
                })
            
            # Check for excessive spreads
            spreads = (tick_df['ask'] - tick_df['bid']) / tick_df['bid']
            excessive_spreads = spreads[spreads > 0.01]  # 1% spread threshold
            
            if len(excessive_spreads) > 0:
                issues.append({
                    'type': DataQualityIssue.SPREAD_ANOMALY,
                    'severity': 'medium',
                    'description': f"Found {len(excessive_spreads)} excessive spreads (>1%)"
                })
        
        return issues
    
    def _validate_price_continuity(self, tick_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check price continuity in tick data"""
        issues = []
        
        if 'price' in tick_df.columns and len(tick_df) > 1:
            price_changes = tick_df['price'].pct_change().abs()
            large_jumps = price_changes[price_changes > 0.05]  # 5% jump
            
            if len(large_jumps) > 0:
                for idx, jump in large_jumps.iterrows():
                    issues.append({
                        'type': DataQualityIssue.PRICE_SPIKE,
                        'severity': 'medium',
                        'description': f"Large price jump: {jump:.2%}",
                        'index': idx
                    })
        
        return issues
    
    def _detect_unusual_patterns(self, tick_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect unusual trading patterns in tick data"""
        issues = []
        
        # Check for wash trades (same price, high volume in short time)
        if len(tick_df) > 10:
            # Group by price and count occurrences in 1-second windows
            tick_df['time_bucket'] = tick_df['timestamp'].dt.floor('1S')
            grouped = tick_df.groupby(['time_bucket', 'price']).size()
            
            suspicious = grouped[grouped > 10]  # More than 10 trades at same price in 1 second
            
            if len(suspicious) > 0:
                issues.append({
                    'type': DataQualityIssue.VOLUME_ANOMALY,
                    'severity': 'low',
                    'description': f"Possible wash trading detected in {len(suspicious)} time buckets"
                })
        
        return issues
    
    def _calculate_data_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data statistics"""
        stats = {
            'record_count': len(df),
            'date_range': f"{df.index[0]} to {df.index[-1]}" if len(df) > 0 else "N/A",
            'completeness': 1.0 - df.isnull().sum().sum() / (len(df) * len(df.columns))
        }
        
        if 'close' in df.columns:
            stats.update({
                'price_mean': df['close'].mean(),
                'price_std': df['close'].std(),
                'returns_mean': df['close'].pct_change().mean(),
                'returns_std': df['close'].pct_change().std(),
                'price_range': df['close'].max() - df['close'].min()
            })
        
        if 'volume' in df.columns:
            stats.update({
                'volume_mean': df['volume'].mean(),
                'volume_total': df['volume'].sum(),
                'zero_volume_count': (df['volume'] == 0).sum()
            })
        
        return stats
    
    def add_corporate_action(self, 
                           symbol: str, 
                           date: Union[str, datetime],
                           action_type: str,
                           adjustment_factor: float,
                           description: str = ""):
        """Add a corporate action to the validator
        
        Args:
            symbol: Stock symbol
            date: Action date
            action_type: Type of action (SPLIT, DIVIDEND, etc.)
            adjustment_factor: Price adjustment factor
            description: Optional description
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        action = CorporateAction(
            date=date,
            action_type=action_type,
            adjustment_factor=adjustment_factor,
            symbol=symbol,
            description=description
        )
        
        if symbol not in self.corporate_actions:
            self.corporate_actions[symbol] = []
        
        self.corporate_actions[symbol].append(action)
        self.logger.info(f"Added {action_type} for {symbol} on {date}")
    
    def generate_quality_report(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive quality report from multiple validations"""
        if not validation_results:
            return {'error': 'No validation results provided'}
        
        report = {
            'total_validations': len(validation_results),
            'average_quality_score': statistics.mean([r.quality_score for r in validation_results]),
            'valid_count': sum(1 for r in validation_results if r.is_valid),
            'issue_summary': {},
            'corrections_summary': {}
        }
        
        # Aggregate issues by type
        for result in validation_results:
            for issue in result.issues:
                issue_type = issue['type'].value
                if issue_type not in report['issue_summary']:
                    report['issue_summary'][issue_type] = 0
                report['issue_summary'][issue_type] += 1
        
        # Aggregate corrections
        for result in validation_results:
            for correction in result.corrections_applied:
                if correction not in report['corrections_summary']:
                    report['corrections_summary'][correction] = 0
                report['corrections_summary'][correction] += 1
        
        return report