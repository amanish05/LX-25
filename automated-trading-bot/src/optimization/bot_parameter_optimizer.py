"""
Bot Parameter Optimizer
Optimizes parameters for all trading bots using historical data and new indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import itertools
from datetime import datetime, timedelta
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

# Import indicators
from ..indicators.rsi_advanced import AdvancedRSI
from ..indicators.oscillator_matrix import OscillatorMatrix
from ..indicators.advanced_confirmation import AdvancedConfirmationSystem


@dataclass
class OptimizationResult:
    """Container for optimization results"""
    bot_name: str
    symbol: str
    timeframe: str
    parameters: Dict
    performance_metrics: Dict
    indicator_config: Dict
    market_regime: str  # 'all', 'high_vol', 'low_vol', 'trending', 'ranging'
    backtest_period: str
    sample_trades: List[Dict]


class BotParameterOptimizer:
    """
    Comprehensive parameter optimizer for all trading bots
    Tests different parameter combinations with new indicators
    """
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = data_dir
        self.results = []
        
        # Define parameter search space for each bot
        self.parameter_space = {
            'momentum_rider': {
                'momentum_threshold': [0.35, 0.40, 0.45, 0.50, 0.55],
                'volume_spike_multiplier': [1.5, 2.0, 2.5, 3.0],
                'min_confirmations': [2, 3, 4],
                'min_confluence_score': [0.55, 0.65, 0.75],
                'max_hold_minutes': [15, 30, 45],
                'trailing_stop_activate': [30, 50, 70],
                'use_rsi': [True, False],
                'use_oscillator_matrix': [True, False]
            },
            'volatility_expander': {
                'iv_percentile_threshold': [20, 25, 30, 35],
                'min_squeeze_periods': [5, 10, 15],
                'profit_target_percent': [75, 100, 150],
                'stop_loss_percent': [-40, -50, -60],
                'max_hold_days': [3, 5, 7],
                'use_oscillator_matrix': [True, False]
            },
            'short_straddle': {
                'iv_rank_threshold': [65, 70, 75, 80],
                'profit_target_percent': [20, 25, 30],
                'stop_loss_percent': [-35, -40, -45],
                'delta_adjustment_threshold': [75, 100, 125],
                'days_to_expiry_min': [20, 25, 30],
                'use_rsi_filter': [True, False]
            },
            'iron_condor': {
                'short_strike_delta': [0.15, 0.20, 0.25],
                'wing_width_percent': [2.0, 2.5, 3.0],
                'profit_target_percent': [40, 50, 60],
                'iv_percentile_min': [40, 50, 60],
                'use_oscillator_confirmation': [True, False]
            }
        }
        
        # Market regime detection thresholds
        self.regime_thresholds = {
            'high_vol': {'vix_min': 25},
            'low_vol': {'vix_max': 15},
            'trending': {'trend_strength_min': 0.7},
            'ranging': {'trend_strength_max': 0.3}
        }
        
    def optimize_all_bots(self, symbols: List[str] = ['NIFTY', 'BANKNIFTY'],
                         start_date: str = '2023-01-01',
                         end_date: str = '2024-12-31') -> pd.DataFrame:
        """
        Optimize parameters for all bots
        
        Args:
            symbols: List of symbols to optimize
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            DataFrame with optimization results
        """
        print(f"Starting bot parameter optimization from {start_date} to {end_date}")
        
        for bot_name in self.parameter_space.keys():
            print(f"\n{'='*60}")
            print(f"Optimizing {bot_name}...")
            print(f"{'='*60}")
            
            for symbol in symbols:
                print(f"\nSymbol: {symbol}")
                
                # Load data
                data = self._load_data(symbol, start_date, end_date)
                if data is None or len(data) < 100:
                    print(f"Insufficient data for {symbol}")
                    continue
                
                # Detect market regimes
                regime_data = self._segment_by_regime(data)
                
                # Optimize for each regime
                for regime, regime_df in regime_data.items():
                    if len(regime_df) < 50:
                        continue
                        
                    print(f"  Optimizing for {regime} market conditions...")
                    
                    # Generate parameter combinations
                    param_combinations = self._generate_param_combinations(bot_name)
                    
                    # Use multiprocessing for faster optimization
                    best_result = self._parallel_optimize(
                        bot_name, symbol, regime_df, param_combinations, regime
                    )
                    
                    if best_result:
                        self.results.append(best_result)
        
        # Generate optimization report
        self._generate_optimization_report()
        
        # Convert results to DataFrame
        results_df = self._results_to_dataframe()
        
        return results_df
    
    def _load_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load historical data for symbol"""
        try:
            # This is a placeholder - implement actual data loading
            # For now, generate synthetic data
            dates = pd.date_range(start=start_date, end=end_date, freq='5T')
            
            # Filter for market hours
            dates = dates[(dates.time >= pd.Timestamp('09:15').time()) & 
                         (dates.time <= pd.Timestamp('15:30').time())]
            
            # Generate synthetic OHLCV data
            np.random.seed(42)
            base_price = 20000 if symbol == 'NIFTY' else 45000
            
            data = pd.DataFrame({
                'timestamp': dates,
                'open': base_price + np.random.randn(len(dates)) * 50,
                'high': base_price + np.random.randn(len(dates)) * 60 + 30,
                'low': base_price + np.random.randn(len(dates)) * 60 - 30,
                'close': base_price + np.cumsum(np.random.randn(len(dates)) * 10),
                'volume': np.random.randint(50000, 200000, len(dates))
            })
            
            data.set_index('timestamp', inplace=True)
            
            # Add technical indicators
            data['returns'] = data['close'].pct_change()
            data['volatility'] = data['returns'].rolling(20).std() * np.sqrt(252)
            data['vix'] = 15 + data['volatility'] * 100  # Synthetic VIX
            
            return data
            
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return None
    
    def _segment_by_regime(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Segment data by market regime"""
        regimes = {'all': data.copy()}
        
        # High volatility regime
        if 'vix' in data.columns:
            high_vol_mask = data['vix'] >= self.regime_thresholds['high_vol']['vix_min']
            if high_vol_mask.sum() > 50:
                regimes['high_vol'] = data[high_vol_mask].copy()
            
            # Low volatility regime
            low_vol_mask = data['vix'] <= self.regime_thresholds['low_vol']['vix_max']
            if low_vol_mask.sum() > 50:
                regimes['low_vol'] = data[low_vol_mask].copy()
        
        # Trending vs ranging (simplified)
        data['trend'] = data['close'].rolling(50).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 50 else 0
        )
        
        trending_mask = abs(data['trend']) > self.regime_thresholds['trending']['trend_strength_min']
        if trending_mask.sum() > 50:
            regimes['trending'] = data[trending_mask].copy()
        
        ranging_mask = abs(data['trend']) < self.regime_thresholds['ranging']['trend_strength_max']
        if ranging_mask.sum() > 50:
            regimes['ranging'] = data[ranging_mask].copy()
        
        return regimes
    
    def _generate_param_combinations(self, bot_name: str) -> List[Dict]:
        """Generate parameter combinations for a bot"""
        param_space = self.parameter_space[bot_name]
        
        # Get all parameter names and values
        param_names = list(param_space.keys())
        param_values = [param_space[name] for name in param_names]
        
        # Generate all combinations
        combinations = []
        for values in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, values)))
        
        # Limit combinations if too many
        if len(combinations) > 1000:
            # Random sample to keep optimization tractable
            import random
            combinations = random.sample(combinations, 1000)
        
        return combinations
    
    def _parallel_optimize(self, bot_name: str, symbol: str, 
                          data: pd.DataFrame, param_combinations: List[Dict],
                          regime: str) -> Optional[OptimizationResult]:
        """Run parallel optimization"""
        best_result = None
        best_score = -float('inf')
        
        # Use smaller chunks for testing
        chunk_size = min(50, len(param_combinations))
        
        with ProcessPoolExecutor(max_workers=mp.cpu_count() - 1) as executor:
            # Submit tasks in chunks
            futures = []
            for i in range(0, len(param_combinations), chunk_size):
                chunk = param_combinations[i:i+chunk_size]
                future = executor.submit(
                    self._optimize_chunk,
                    bot_name, symbol, data, chunk, regime
                )
                futures.append(future)
            
            # Collect results
            for future in futures:
                chunk_results = future.result()
                for result, score in chunk_results:
                    if score > best_score:
                        best_score = score
                        best_result = result
        
        return best_result
    
    def _optimize_chunk(self, bot_name: str, symbol: str,
                       data: pd.DataFrame, param_chunk: List[Dict],
                       regime: str) -> List[Tuple[OptimizationResult, float]]:
        """Optimize a chunk of parameter combinations"""
        results = []
        
        for params in param_chunk:
            result = self._backtest_bot(bot_name, symbol, data, params, regime)
            if result:
                score = self._calculate_optimization_score(result.performance_metrics)
                results.append((result, score))
        
        return results
    
    def _backtest_bot(self, bot_name: str, symbol: str,
                     data: pd.DataFrame, params: Dict,
                     regime: str) -> Optional[OptimizationResult]:
        """Backtest a bot with specific parameters"""
        try:
            # Initialize indicators based on parameters
            indicators = {}
            
            if params.get('use_rsi', False):
                indicators['rsi'] = AdvancedRSI()
            
            if params.get('use_oscillator_matrix', False):
                indicators['oscillator_matrix'] = OscillatorMatrix()
            
            # Simulate bot trading based on type
            if bot_name == 'momentum_rider':
                trades = self._backtest_momentum_rider(data, params, indicators)
            elif bot_name == 'volatility_expander':
                trades = self._backtest_volatility_expander(data, params, indicators)
            elif bot_name == 'short_straddle':
                trades = self._backtest_short_straddle(data, params, indicators)
            elif bot_name == 'iron_condor':
                trades = self._backtest_iron_condor(data, params, indicators)
            else:
                return None
            
            if not trades:
                return None
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(trades)
            
            # Create result object
            result = OptimizationResult(
                bot_name=bot_name,
                symbol=symbol,
                timeframe='5min',
                parameters=params,
                performance_metrics=metrics,
                indicator_config={k: v.__class__.__name__ for k, v in indicators.items()},
                market_regime=regime,
                backtest_period=f"{data.index[0]} to {data.index[-1]}",
                sample_trades=trades[:5]  # Store first 5 trades as sample
            )
            
            return result
            
        except Exception as e:
            print(f"Error in backtest: {e}")
            return None
    
    def _backtest_momentum_rider(self, data: pd.DataFrame, params: Dict,
                               indicators: Dict) -> List[Dict]:
        """Backtest momentum rider strategy"""
        trades = []
        
        # Calculate momentum
        for i in range(20, len(data) - 10):
            # Primary momentum signal
            momentum = (data['close'].iloc[i] - data['close'].iloc[i-5]) / data['close'].iloc[i-5]
            
            if abs(momentum) < params['momentum_threshold']:
                continue
            
            # Volume check
            vol_ratio = data['volume'].iloc[i] / data['volume'].iloc[i-20:i].mean()
            if vol_ratio < params['volume_spike_multiplier']:
                continue
            
            # Additional confirmations
            confirmations = 0
            
            # RSI confirmation
            if 'rsi' in indicators:
                rsi_values = indicators['rsi'].calculate(data['close'][:i+1])
                if len(rsi_values) > 0 and not pd.isna(rsi_values.iloc[-1]):
                    current_rsi = rsi_values.iloc[-1]
                    if (momentum > 0 and current_rsi < 70) or (momentum < 0 and current_rsi > 30):
                        confirmations += 1
            
            # Oscillator matrix confirmation
            if 'oscillator_matrix' in indicators:
                osc_data = indicators['oscillator_matrix'].calculate_all_oscillators(data[:i+1])
                if len(osc_data) > 0 and 'composite_score' in osc_data.columns:
                    composite = osc_data['composite_score'].iloc[-1]
                    if (momentum > 0 and composite < 30) or (momentum < 0 and composite > -30):
                        confirmations += 1
            
            # Check minimum confirmations
            if confirmations < params['min_confirmations']:
                continue
            
            # Simulate trade
            entry_price = data['close'].iloc[i]
            direction = 'CALL' if momentum > 0 else 'PUT'
            
            # Simple exit after hold period
            exit_idx = min(i + params['max_hold_minutes'] // 5, len(data) - 1)
            exit_price = data['close'].iloc[exit_idx]
            
            # Calculate P&L
            if direction == 'CALL':
                pnl = (exit_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_price) / entry_price
            
            trades.append({
                'entry_time': data.index[i],
                'exit_time': data.index[exit_idx],
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_percent': pnl * 100,
                'confirmations': confirmations
            })
        
        return trades
    
    def _backtest_volatility_expander(self, data: pd.DataFrame, params: Dict,
                                    indicators: Dict) -> List[Dict]:
        """Backtest volatility expander strategy"""
        trades = []
        
        # Calculate IV proxy (using realized volatility)
        data['iv_proxy'] = data['returns'].rolling(20).std() * np.sqrt(252) * 100
        data['iv_rank'] = data['iv_proxy'].rolling(252).rank(pct=True) * 100
        
        for i in range(252, len(data) - params['max_hold_days'] * 75):
            # Check IV conditions
            if data['iv_rank'].iloc[i] > params['iv_percentile_threshold']:
                continue
            
            # Check for volatility squeeze
            recent_vol = data['iv_proxy'].iloc[i-params['min_squeeze_periods']:i+1]
            if recent_vol.std() > recent_vol.mean() * 0.1:  # Not squeezed enough
                continue
            
            # Oscillator confirmation
            if 'oscillator_matrix' in indicators:
                osc_data = indicators['oscillator_matrix'].calculate_all_oscillators(data[:i+1])
                if len(osc_data) > 0:
                    composite = osc_data['composite_score'].iloc[-1]
                    # Look for neutral oscillators (ready to expand)
                    if abs(composite) > 50:
                        continue
            
            # Simulate trade
            entry_price = data['close'].iloc[i]
            entry_vol = data['iv_proxy'].iloc[i]
            
            # Exit when vol expands or max hold reached
            for j in range(i+1, min(i + params['max_hold_days'] * 75, len(data))):
                current_vol = data['iv_proxy'].iloc[j]
                vol_expansion = (current_vol - entry_vol) / entry_vol
                
                if vol_expansion > 0.5:  # 50% vol expansion
                    exit_idx = j
                    break
            else:
                exit_idx = min(i + params['max_hold_days'] * 75, len(data) - 1)
            
            exit_price = data['close'].iloc[exit_idx]
            
            # Simplified P&L (assuming long straddle)
            price_move = abs(exit_price - entry_price) / entry_price
            vol_pnl = (data['iv_proxy'].iloc[exit_idx] - entry_vol) / entry_vol
            total_pnl = price_move * 100 + vol_pnl * 50  # Simplified
            
            trades.append({
                'entry_time': data.index[i],
                'exit_time': data.index[exit_idx],
                'strategy': 'long_straddle',
                'entry_iv': entry_vol,
                'exit_iv': data['iv_proxy'].iloc[exit_idx],
                'pnl_percent': total_pnl
            })
        
        return trades
    
    def _backtest_short_straddle(self, data: pd.DataFrame, params: Dict,
                                indicators: Dict) -> List[Dict]:
        """Backtest short straddle strategy"""
        trades = []
        
        # Calculate IV rank
        data['iv_proxy'] = data['returns'].rolling(20).std() * np.sqrt(252) * 100
        data['iv_rank'] = data['iv_proxy'].rolling(252).rank(pct=True) * 100
        
        for i in range(252, len(data) - 30 * 75):  # Need 30 days for expiry
            # Check IV rank
            if data['iv_rank'].iloc[i] < params['iv_rank_threshold']:
                continue
            
            # RSI filter
            if params.get('use_rsi_filter', False) and 'rsi' in indicators:
                rsi_values = indicators['rsi'].calculate(data['close'][:i+1])
                if len(rsi_values) > 0:
                    current_rsi = rsi_values.iloc[-1]
                    # Avoid extreme RSI values
                    if current_rsi < 20 or current_rsi > 80:
                        continue
            
            # Simulate trade
            entry_price = data['close'].iloc[i]
            premium_collected = entry_price * 0.015  # 1.5% premium estimate
            
            # Track P&L
            max_loss = 0
            for j in range(i+1, min(i + params['days_to_expiry_min'] * 75, len(data))):
                current_price = data['close'].iloc[j]
                price_move = abs(current_price - entry_price)
                current_loss = max(0, price_move - premium_collected)
                max_loss = max(max_loss, current_loss)
                
                # Check stop loss
                if current_loss > premium_collected * abs(params['stop_loss_percent']) / 100:
                    exit_idx = j
                    final_pnl = -current_loss
                    break
                
                # Check profit target
                if current_loss == 0 and j > i + 5 * 75:  # After 5 days
                    remaining_premium = premium_collected * (1 - (j-i)/(params['days_to_expiry_min']*75))
                    if remaining_premium < premium_collected * (1 - params['profit_target_percent']/100):
                        exit_idx = j
                        final_pnl = premium_collected - remaining_premium
                        break
            else:
                # Held to expiry
                exit_idx = min(i + params['days_to_expiry_min'] * 75, len(data) - 1)
                final_price = data['close'].iloc[exit_idx]
                final_move = abs(final_price - entry_price)
                final_pnl = premium_collected - max(0, final_move - premium_collected)
            
            trades.append({
                'entry_time': data.index[i],
                'exit_time': data.index[exit_idx],
                'entry_price': entry_price,
                'premium_collected': premium_collected,
                'pnl': final_pnl,
                'pnl_percent': (final_pnl / premium_collected) * 100
            })
        
        return trades
    
    def _backtest_iron_condor(self, data: pd.DataFrame, params: Dict,
                            indicators: Dict) -> List[Dict]:
        """Backtest iron condor strategy"""
        trades = []
        
        # Similar to short straddle but with limited risk
        # Simplified implementation
        data['iv_proxy'] = data['returns'].rolling(20).std() * np.sqrt(252) * 100
        data['iv_percentile'] = data['iv_proxy'].rolling(252).rank(pct=True) * 100
        
        for i in range(252, len(data) - 45 * 75):
            # Check IV percentile
            if data['iv_percentile'].iloc[i] < params['iv_percentile_min']:
                continue
            
            # Oscillator confirmation
            if params.get('use_oscillator_confirmation', False) and 'oscillator_matrix' in indicators:
                osc_data = indicators['oscillator_matrix'].calculate_all_oscillators(data[:i+1])
                if len(osc_data) > 0:
                    # Want neutral market for iron condor
                    composite = osc_data['composite_score'].iloc[-1]
                    if abs(composite) > 30:
                        continue
            
            # Simulate trade
            entry_price = data['close'].iloc[i]
            wing_width = entry_price * params['wing_width_percent'] / 100
            
            # Estimate premium (simplified)
            premium = wing_width * 0.3  # 30% of wing width
            max_loss = wing_width - premium
            
            # Track P&L
            for j in range(i+1, min(i + 45 * 75, len(data))):
                current_price = data['close'].iloc[j]
                
                # Check if price breached wings
                if abs(current_price - entry_price) > wing_width:
                    final_pnl = -max_loss
                    exit_idx = j
                    break
                
                # Check profit target
                time_decay = (j - i) / (45 * 75)
                current_value = premium * (1 - time_decay)
                if current_value < premium * (1 - params['profit_target_percent']/100):
                    final_pnl = premium - current_value
                    exit_idx = j
                    break
            else:
                # Held to expiry
                exit_idx = min(i + 45 * 75, len(data) - 1)
                final_pnl = premium  # Full premium collected
            
            trades.append({
                'entry_time': data.index[i],
                'exit_time': data.index[exit_idx],
                'entry_price': entry_price,
                'wing_width': wing_width,
                'premium': premium,
                'pnl': final_pnl,
                'pnl_percent': (final_pnl / premium) * 100
            })
        
        return trades
    
    def _calculate_performance_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'avg_hold_time': 0,
                'expectancy': 0
            }
        
        # Extract P&L
        pnls = [t['pnl_percent'] for t in trades]
        pnl_array = np.array(pnls)
        
        # Win/Loss statistics
        wins = pnl_array[pnl_array > 0]
        losses = pnl_array[pnl_array <= 0]
        
        win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0
        
        # Profit factor
        gross_profit = np.sum(wins) if len(wins) > 0 else 0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Risk metrics
        returns = pnl_array / 100  # Convert to decimal
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-6
        sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252)
        
        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100
        
        # Average metrics
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        # Hold time (if available)
        if 'entry_time' in trades[0] and 'exit_time' in trades[0]:
            hold_times = [(t['exit_time'] - t['entry_time']).total_seconds() / 3600 
                         for t in trades]
            avg_hold_time = np.mean(hold_times)
        else:
            avg_hold_time = 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': np.max(pnls) if len(pnls) > 0 else 0,
            'largest_loss': np.min(pnls) if len(pnls) > 0 else 0,
            'avg_hold_time': avg_hold_time,
            'expectancy': expectancy
        }
    
    def _calculate_optimization_score(self, metrics: Dict) -> float:
        """Calculate composite optimization score"""
        # Weighted scoring
        score = 0
        
        # Win rate (20%)
        score += metrics['win_rate'] * 20
        
        # Sharpe ratio (30%)
        score += min(metrics['sharpe_ratio'], 3) * 10  # Cap at 3
        
        # Profit factor (20%)
        score += min(metrics['profit_factor'], 3) * 6.67  # Cap at 3
        
        # Max drawdown (20%) - less is better
        score += max(0, 20 + metrics['max_drawdown'] / 5)  # -20% DD = 16 points
        
        # Trade frequency (10%) - prefer reasonable frequency
        if metrics['total_trades'] > 10:
            score += min(10, metrics['total_trades'] / 10)
        
        return score
    
    def _generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        if not self.results:
            return
        
        report = []
        report.append("# Bot Parameter Optimization Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Summary of Optimal Parameters\n")
        
        # Group by bot
        for bot_name in self.parameter_space.keys():
            bot_results = [r for r in self.results if r.bot_name == bot_name]
            if not bot_results:
                continue
            
            report.append(f"### {bot_name.replace('_', ' ').title()}")
            
            # Find best parameters for each symbol and regime
            for symbol in ['NIFTY', 'BANKNIFTY']:
                symbol_results = [r for r in bot_results if r.symbol == symbol]
                if not symbol_results:
                    continue
                
                report.append(f"\n**{symbol}:**")
                
                # Best overall
                best_overall = max(symbol_results, 
                                 key=lambda x: x.performance_metrics['sharpe_ratio'])
                
                report.append(f"- Best Overall Parameters:")
                report.append(f"  - Sharpe Ratio: {best_overall.performance_metrics['sharpe_ratio']:.2f}")
                report.append(f"  - Win Rate: {best_overall.performance_metrics['win_rate']:.1%}")
                report.append(f"  - Parameters: {json.dumps(best_overall.parameters, indent=4)}")
                
                # Best by regime
                for regime in ['high_vol', 'low_vol', 'trending', 'ranging']:
                    regime_results = [r for r in symbol_results if r.market_regime == regime]
                    if regime_results:
                        best_regime = max(regime_results, 
                                        key=lambda x: x.performance_metrics['sharpe_ratio'])
                        report.append(f"\n- Best for {regime.replace('_', ' ').title()}:")
                        report.append(f"  - Sharpe: {best_regime.performance_metrics['sharpe_ratio']:.2f}")
                        report.append(f"  - Key params: {self._get_key_params(best_regime.parameters)}")
            
            report.append("\n" + "-"*60 + "\n")
        
        # Save report
        report_text = "\n".join(report)
        with open('reports/bot_optimization_report.md', 'w') as f:
            f.write(report_text)
        
        print("\nOptimization report saved to reports/bot_optimization_report.md")
    
    def _get_key_params(self, params: Dict) -> str:
        """Extract key parameters for summary"""
        key_params = []
        
        # Define key parameters for each bot
        key_param_names = {
            'momentum_threshold': 'Mom',
            'min_confirmations': 'Conf',
            'iv_rank_threshold': 'IV',
            'profit_target_percent': 'PT',
            'stop_loss_percent': 'SL'
        }
        
        for param, value in params.items():
            if param in key_param_names:
                key_params.append(f"{key_param_names[param]}={value}")
        
        return ", ".join(key_params[:3])  # Show top 3
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        if not self.results:
            return pd.DataFrame()
        
        rows = []
        for result in self.results:
            row = {
                'bot': result.bot_name,
                'symbol': result.symbol,
                'regime': result.market_regime,
                'sharpe_ratio': result.performance_metrics['sharpe_ratio'],
                'win_rate': result.performance_metrics['win_rate'],
                'profit_factor': result.performance_metrics['profit_factor'],
                'max_drawdown': result.performance_metrics['max_drawdown'],
                'total_trades': result.performance_metrics['total_trades'],
                'parameters': json.dumps(result.parameters)
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_optimal_parameters(self):
        """Save optimal parameters to config file"""
        if not self.results:
            return
        
        optimal_params = {}
        
        # Get best parameters for each bot/symbol combination
        for bot_name in self.parameter_space.keys():
            optimal_params[bot_name] = {}
            
            for symbol in ['NIFTY', 'BANKNIFTY']:
                bot_symbol_results = [r for r in self.results 
                                    if r.bot_name == bot_name and r.symbol == symbol]
                
                if bot_symbol_results:
                    # Get best overall
                    best = max(bot_symbol_results, 
                             key=lambda x: x.performance_metrics['sharpe_ratio'])
                    
                    optimal_params[bot_name][symbol] = {
                        'parameters': best.parameters,
                        'performance': {
                            'sharpe_ratio': best.performance_metrics['sharpe_ratio'],
                            'win_rate': best.performance_metrics['win_rate'],
                            'max_drawdown': best.performance_metrics['max_drawdown']
                        },
                        'indicators': best.indicator_config,
                        'optimized_for': best.market_regime
                    }
        
        # Save to file
        with open('config/optimal_bot_parameters.json', 'w') as f:
            json.dump(optimal_params, f, indent=2)
        
        print("\nOptimal parameters saved to config/optimal_bot_parameters.json")