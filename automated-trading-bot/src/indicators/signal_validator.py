"""
Signal Validator
Advanced false positive filtering and signal validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from collections import defaultdict


class SignalValidator:
    """
    Validates trading signals to reduce false positives
    Tracks performance and adapts filtering based on results
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize signal validator"""
        self.config = config or {}
        
        # Validation rules
        self.rules = {
            'market_hours': True,
            'volatility_filter': True,
            'correlation_check': True,
            'news_impact': True,
            'pattern_recognition': True,
            'time_of_day': True
        }
        
        # Performance tracking
        self.signal_outcomes = defaultdict(list)
        self.pattern_performance = defaultdict(lambda: {'success': 0, 'failure': 0})
        self.time_performance = defaultdict(lambda: {'success': 0, 'failure': 0})
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            'min_success_rate': 0.55,
            'volatility_threshold': 0.30,
            'correlation_limit': 0.80,
            'pattern_confidence': 0.65
        }
        
        # Load historical patterns if available
        self.load_historical_patterns()
    
    def validate(self, signal: Dict, market_data: pd.DataFrame, 
                market_context: Optional[Dict] = None) -> Tuple[bool, Dict]:
        """
        Validate a signal against multiple criteria
        
        Args:
            signal: Trading signal to validate
            market_data: Historical price data
            market_context: Additional market information
            
        Returns:
            Tuple of (is_valid, validation_details)
        """
        validation_results = {}
        failed_rules = []
        
        # 1. Market hours validation
        if self.rules['market_hours']:
            hours_valid = self._validate_market_hours(signal)
            validation_results['market_hours'] = hours_valid
            if not hours_valid:
                failed_rules.append('market_hours')
        
        # 2. Volatility filter
        if self.rules['volatility_filter']:
            vol_valid, vol_details = self._validate_volatility(market_data)
            validation_results['volatility'] = vol_valid
            validation_results['volatility_details'] = vol_details
            if not vol_valid:
                failed_rules.append('volatility_filter')
        
        # 3. Correlation check
        if self.rules['correlation_check']:
            corr_valid = self._validate_correlation(signal, market_context)
            validation_results['correlation'] = corr_valid
            if not corr_valid:
                failed_rules.append('correlation_check')
        
        # 4. Pattern recognition
        if self.rules['pattern_recognition']:
            pattern_valid, pattern_confidence = self._validate_pattern(signal, market_data)
            validation_results['pattern'] = pattern_valid
            validation_results['pattern_confidence'] = pattern_confidence
            if not pattern_valid:
                failed_rules.append('pattern_recognition')
        
        # 5. Time of day analysis
        if self.rules['time_of_day']:
            time_valid, time_score = self._validate_time_of_day(signal)
            validation_results['time_of_day'] = time_valid
            validation_results['time_score'] = time_score
            if not time_valid:
                failed_rules.append('time_of_day')
        
        # 6. False positive probability
        fp_probability = self._calculate_false_positive_probability(
            signal, market_data, validation_results
        )
        validation_results['false_positive_probability'] = fp_probability
        
        # Overall validation
        is_valid = len(failed_rules) == 0 and fp_probability < 0.35
        
        validation_details = {
            'is_valid': is_valid,
            'failed_rules': failed_rules,
            'validation_results': validation_results,
            'recommendation': self._get_recommendation(is_valid, failed_rules, fp_probability)
        }
        
        # Track for learning
        self._track_signal_validation(signal, validation_details)
        
        return is_valid, validation_details
    
    def _validate_market_hours(self, signal: Dict) -> bool:
        """Validate if signal is during optimal market hours"""
        signal_time = datetime.fromisoformat(signal.get('timestamp', datetime.now().isoformat()))
        hour = signal_time.hour
        minute = signal_time.minute
        
        # Avoid first and last 15 minutes
        if hour == 9 and minute < 30:
            return False
        if hour == 15 and minute > 15:
            return False
        
        # Optimal hours based on historical performance
        optimal_hours = [(10, 11), (11, 12), (14, 15)]
        
        for start, end in optimal_hours:
            if start <= hour < end:
                return True
        
        # Check historical performance for this hour
        hour_key = f"{hour:02d}:00"
        if hour_key in self.time_performance:
            perf = self.time_performance[hour_key]
            success_rate = perf['success'] / (perf['success'] + perf['failure'] + 1)
            return success_rate >= self.adaptive_thresholds['min_success_rate']
        
        return True  # Default to valid if no historical data
    
    def _validate_volatility(self, market_data: pd.DataFrame) -> Tuple[bool, Dict]:
        """Validate volatility conditions"""
        # Calculate various volatility measures
        returns = market_data['close'].pct_change()
        
        # Current volatility
        current_vol = returns.iloc[-20:].std() * np.sqrt(252)
        
        # Historical volatility
        hist_vol = returns.std() * np.sqrt(252)
        
        # Volatility of volatility
        vol_changes = returns.rolling(20).std()
        vol_of_vol = vol_changes.pct_change().std()
        
        # ATR-based volatility
        atr = self._calculate_atr(market_data)
        atr_percent = atr / market_data['close'].iloc[-1]
        
        # Validation logic
        is_valid = True
        details = {
            'current_volatility': current_vol,
            'historical_volatility': hist_vol,
            'volatility_ratio': current_vol / hist_vol if hist_vol > 0 else 1,
            'atr_percent': atr_percent,
            'vol_of_vol': vol_of_vol
        }
        
        # Check conditions
        if current_vol > self.adaptive_thresholds['volatility_threshold']:
            is_valid = False
            details['reason'] = 'Volatility too high'
        elif current_vol < 0.05:  # Too low volatility
            is_valid = False
            details['reason'] = 'Volatility too low'
        elif vol_of_vol > 0.5:  # Unstable volatility
            is_valid = False
            details['reason'] = 'Volatility regime unstable'
        
        return is_valid, details
    
    def _validate_correlation(self, signal: Dict, market_context: Optional[Dict]) -> bool:
        """Check correlation with other positions/signals"""
        if not market_context or 'active_positions' not in market_context:
            return True
        
        # Check correlation with existing positions
        active_positions = market_context['active_positions']
        signal_symbol = signal['symbol']
        
        # Simple correlation check - in production would use actual price correlation
        correlated_symbols = {
            'NIFTY': ['BANKNIFTY', 'FINNIFTY'],
            'BANKNIFTY': ['NIFTY', 'FINNIFTY'],
            'FINNIFTY': ['NIFTY', 'BANKNIFTY']
        }
        
        # Count correlated positions
        correlated_count = 0
        for position in active_positions:
            if position['symbol'] in correlated_symbols.get(signal_symbol, []):
                correlated_count += 1
        
        # Limit correlated positions
        return correlated_count < 2
    
    def _validate_pattern(self, signal: Dict, market_data: pd.DataFrame) -> Tuple[bool, float]:
        """Validate signal against known patterns"""
        pattern_key = self._extract_pattern_key(signal, market_data)
        
        # Check historical performance of this pattern
        if pattern_key in self.pattern_performance:
            perf = self.pattern_performance[pattern_key]
            total = perf['success'] + perf['failure']
            
            if total >= 10:  # Minimum sample size
                success_rate = perf['success'] / total
                confidence = min(total / 50, 1.0) * success_rate  # Confidence increases with samples
                
                return (
                    success_rate >= self.adaptive_thresholds['min_success_rate'],
                    confidence
                )
        
        # Default validation based on signal characteristics
        confidence = signal.get('confluence_score', 0.5)
        return confidence >= self.adaptive_thresholds['pattern_confidence'], confidence
    
    def _validate_time_of_day(self, signal: Dict) -> Tuple[bool, float]:
        """Validate based on time of day performance"""
        signal_time = datetime.fromisoformat(signal.get('timestamp', datetime.now().isoformat()))
        time_key = f"{signal_time.hour:02d}:{signal_time.minute//15*15:02d}"
        
        # Check historical performance
        if time_key in self.time_performance:
            perf = self.time_performance[time_key]
            total = perf['success'] + perf['failure']
            
            if total >= 5:
                success_rate = perf['success'] / total
                return success_rate >= 0.5, success_rate
        
        # Default scores for different times
        hour = signal_time.hour
        if 10 <= hour <= 11 or 14 <= hour <= 15:
            return True, 0.7  # Optimal hours
        elif 9 <= hour <= 10 or 11 <= hour <= 14:
            return True, 0.5  # Acceptable hours
        else:
            return False, 0.3  # Poor hours
    
    def _calculate_false_positive_probability(self, signal: Dict, 
                                            market_data: pd.DataFrame,
                                            validation_results: Dict) -> float:
        """Calculate probability of false positive"""
        base_probability = 0.35
        
        # Adjust based on signal strength
        confluence_score = signal.get('confluence_score', 0.5)
        probability = base_probability * (1 - confluence_score)
        
        # Adjust based on validation results
        passed_validations = sum(1 for k, v in validation_results.items() 
                               if k not in ['validation_results', 'false_positive_probability'] 
                               and v is True)
        total_validations = len([k for k in validation_results.keys() 
                               if k not in ['validation_results', 'false_positive_probability']])
        
        if total_validations > 0:
            validation_score = passed_validations / total_validations
            probability *= (1 - validation_score * 0.5)
        
        # Adjust based on market conditions
        volatility = validation_results.get('volatility_details', {}).get('current_volatility', 0.15)
        if volatility > 0.25:
            probability *= 1.2
        elif volatility < 0.10:
            probability *= 1.1  # Low volatility can also be problematic
        
        # Historical adjustment
        pattern_confidence = validation_results.get('pattern_confidence', 0.5)
        probability *= (1 - pattern_confidence * 0.3)
        
        return max(0.1, min(0.9, probability))
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        if len(tr) >= period:
            return np.mean(tr[-period:])
        return np.mean(tr) if len(tr) > 0 else 0
    
    def _extract_pattern_key(self, signal: Dict, market_data: pd.DataFrame) -> str:
        """Extract pattern key for tracking"""
        # Create a pattern identifier based on signal characteristics
        signal_type = signal.get('type', 'unknown')
        
        # Price action context
        price_trend = 'up' if market_data['close'].iloc[-1] > market_data['close'].iloc[-20] else 'down'
        
        # Volatility context
        vol = market_data['close'].pct_change().iloc[-20:].std() * np.sqrt(252)
        vol_regime = 'high' if vol > 0.20 else 'normal' if vol > 0.10 else 'low'
        
        # Volume context
        current_volume = market_data['volume'].iloc[-1]
        avg_volume = market_data['volume'].iloc[-20:].mean()
        volume_state = 'high' if current_volume > avg_volume * 1.5 else 'normal'
        
        return f"{signal_type}_{price_trend}_{vol_regime}_{volume_state}"
    
    def _get_recommendation(self, is_valid: bool, failed_rules: List[str], 
                          fp_probability: float) -> str:
        """Get recommendation based on validation results"""
        if is_valid and fp_probability < 0.25:
            return "STRONG_SIGNAL: Execute with full position size"
        elif is_valid and fp_probability < 0.35:
            return "MODERATE_SIGNAL: Execute with reduced position size"
        elif not is_valid and len(failed_rules) == 1:
            return f"WEAK_SIGNAL: Consider waiting. Failed: {failed_rules[0]}"
        else:
            return f"REJECT_SIGNAL: Multiple failures: {', '.join(failed_rules)}"
    
    def _track_signal_validation(self, signal: Dict, validation_details: Dict):
        """Track signal for learning and adaptation"""
        signal_record = {
            'timestamp': signal.get('timestamp', datetime.now().isoformat()),
            'signal': signal,
            'validation': validation_details,
            'outcome': None  # To be updated later
        }
        
        # Store for later analysis
        symbol = signal.get('symbol', 'unknown')
        self.signal_outcomes[symbol].append(signal_record)
        
        # Limit history
        if len(self.signal_outcomes[symbol]) > 1000:
            self.signal_outcomes[symbol] = self.signal_outcomes[symbol][-500:]
    
    def update_signal_outcome(self, signal_id: str, outcome: Dict):
        """Update signal outcome for learning"""
        # Update the tracked signal with actual outcome
        # This would be called after position is closed
        success = outcome.get('profitable', False)
        
        # Update pattern performance
        pattern_key = outcome.get('pattern_key')
        if pattern_key:
            if success:
                self.pattern_performance[pattern_key]['success'] += 1
            else:
                self.pattern_performance[pattern_key]['failure'] += 1
        
        # Update time performance
        time_key = outcome.get('time_key')
        if time_key:
            if success:
                self.time_performance[time_key]['success'] += 1
            else:
                self.time_performance[time_key]['failure'] += 1
        
        # Adapt thresholds if needed
        self._adapt_thresholds()
    
    def _adapt_thresholds(self):
        """Adapt thresholds based on performance"""
        # Calculate overall success rate
        total_success = sum(p['success'] for p in self.pattern_performance.values())
        total_failure = sum(p['failure'] for p in self.pattern_performance.values())
        total_trades = total_success + total_failure
        
        if total_trades >= 100:
            success_rate = total_success / total_trades
            
            # Adjust minimum success rate threshold
            if success_rate > 0.65:
                # Performing well, can be slightly less strict
                self.adaptive_thresholds['min_success_rate'] = max(0.50, 
                    self.adaptive_thresholds['min_success_rate'] - 0.02)
            elif success_rate < 0.45:
                # Performing poorly, be more strict
                self.adaptive_thresholds['min_success_rate'] = min(0.65,
                    self.adaptive_thresholds['min_success_rate'] + 0.02)
    
    def save_historical_patterns(self):
        """Save learned patterns to file"""
        patterns_data = {
            'pattern_performance': dict(self.pattern_performance),
            'time_performance': dict(self.time_performance),
            'adaptive_thresholds': self.adaptive_thresholds,
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open('data/signal_patterns.json', 'w') as f:
                json.dump(patterns_data, f, indent=2)
        except Exception as e:
            print(f"Error saving patterns: {e}")
    
    def load_historical_patterns(self):
        """Load historical patterns from file"""
        try:
            with open('data/signal_patterns.json', 'r') as f:
                patterns_data = json.load(f)
                
            self.pattern_performance = defaultdict(
                lambda: {'success': 0, 'failure': 0},
                patterns_data.get('pattern_performance', {})
            )
            self.time_performance = defaultdict(
                lambda: {'success': 0, 'failure': 0},
                patterns_data.get('time_performance', {})
            )
            self.adaptive_thresholds.update(
                patterns_data.get('adaptive_thresholds', {})
            )
        except Exception:
            # No historical data available
            pass
    
    def get_validation_stats(self) -> Dict:
        """Get validation statistics"""
        stats = {
            'total_patterns': len(self.pattern_performance),
            'total_time_slots': len(self.time_performance),
            'adaptive_thresholds': self.adaptive_thresholds,
            'pattern_stats': {},
            'time_stats': {}
        }
        
        # Pattern performance stats
        for pattern, perf in self.pattern_performance.items():
            total = perf['success'] + perf['failure']
            if total > 0:
                stats['pattern_stats'][pattern] = {
                    'success_rate': perf['success'] / total,
                    'total_signals': total
                }
        
        # Time performance stats
        for time_slot, perf in self.time_performance.items():
            total = perf['success'] + perf['failure']
            if total > 0:
                stats['time_stats'][time_slot] = {
                    'success_rate': perf['success'] / total,
                    'total_signals': total
                }
        
        return stats