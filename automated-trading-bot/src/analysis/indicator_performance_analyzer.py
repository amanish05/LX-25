"""
Indicator Performance Analyzer
Tests different indicators with various parameters to find optimal combinations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

# Import all indicators
from ..indicators.rsi_advanced import AdvancedRSI
from ..indicators.oscillator_matrix import OscillatorMatrix
from ..indicators.momentum import MomentumIndicators
from ..indicators.volatility import VolatilityIndicators
from ..indicators.reversal_signals import ReversalSignalsIndicator
from ..indicators.advanced_confirmation import AdvancedConfirmationSystem
# Price Action indicators
from ..indicators.market_structure import MarketStructure
from ..indicators.order_blocks import OrderBlocks
from ..indicators.fair_value_gaps import FairValueGaps
from ..indicators.liquidity_zones import LiquidityZones
from ..indicators.pattern_recognition import PatternRecognition
from ..indicators.price_action_composite import PriceActionComposite


@dataclass
class IndicatorPerformance:
    """Container for indicator performance metrics"""
    indicator_name: str
    parameters: Dict
    scenario: str  # 'trending', 'ranging', 'volatile', 'calm'
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_signals: int
    false_positives: int
    avg_profit_per_signal: float
    best_timeframe: str  # '5min', '15min', '30min', '1hour'
    correlation_with_others: Dict[str, float]


class IndicatorPerformanceAnalyzer:
    """
    Analyzes performance of different indicators across various market scenarios
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize analyzer
        
        Args:
            data_path: Path to historical data
        """
        self.data_path = data_path
        self.results = []
        
        # Define market scenarios
        self.scenarios = {
            'trending': {'volatility': (0.01, 0.02), 'trend_strength': 0.7},
            'ranging': {'volatility': (0.005, 0.015), 'trend_strength': 0.3},
            'volatile': {'volatility': (0.02, 0.04), 'trend_strength': 0.5},
            'calm': {'volatility': (0.002, 0.008), 'trend_strength': 0.4}
        }
        
        # Define parameter grids for each indicator
        self.parameter_grids = {
            'rsi': {
                'period': [9, 14, 21],
                'overbought': [65, 70, 75],
                'oversold': [25, 30, 35]
            },
            'oscillator_matrix': {
                'rsi_period': [14, 21],
                'macd_fast': [8, 12],
                'macd_slow': [21, 26],
                'stoch_period': [9, 14]
            },
            'momentum': {
                'fast_period': [5, 10],
                'slow_period': [20, 30],
                'threshold': [0.3, 0.45, 0.6]
            },
            'volatility': {
                'bb_period': [20, 25],
                'bb_std': [1.5, 2.0, 2.5],
                'atr_period': [14, 21]
            },
            'reversal': {
                'lookback': [10, 20],
                'min_reversal_size': [0.5, 1.0],
                'volume_threshold': [1.5, 2.0]
            }
        }
        
    def analyze_all_indicators(self, data: pd.DataFrame, 
                             timeframes: List[str] = ['5min', '15min', '30min']) -> pd.DataFrame:
        """
        Analyze performance of all indicators across different scenarios
        
        Args:
            data: Historical OHLCV data
            timeframes: List of timeframes to test
            
        Returns:
            DataFrame with performance results
        """
        print("Starting comprehensive indicator analysis...")
        
        # Detect market scenarios in the data
        scenarios_data = self._segment_data_by_scenario(data)
        
        # Test each indicator
        for indicator_name in self.parameter_grids.keys():
            print(f"\nAnalyzing {indicator_name}...")
            
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations(indicator_name)
            
            # Test each combination in each scenario
            for scenario_name, scenario_data in scenarios_data.items():
                if len(scenario_data) < 100:  # Skip if not enough data
                    continue
                    
                print(f"  Testing in {scenario_name} market conditions...")
                
                for params in param_combinations:
                    for timeframe in timeframes:
                        # Resample data to timeframe
                        resampled_data = self._resample_data(scenario_data, timeframe)
                        
                        if len(resampled_data) < 50:
                            continue
                        
                        # Test indicator
                        performance = self._test_indicator(
                            indicator_name, 
                            params, 
                            resampled_data, 
                            scenario_name,
                            timeframe
                        )
                        
                        if performance:
                            self.results.append(performance)
        
        # Calculate correlations between indicators
        self._calculate_indicator_correlations()
        
        # Convert results to DataFrame
        results_df = pd.DataFrame([
            {
                'indicator': r.indicator_name,
                'scenario': r.scenario,
                'timeframe': r.best_timeframe,
                'win_rate': r.win_rate,
                'profit_factor': r.profit_factor,
                'sharpe_ratio': r.sharpe_ratio,
                'max_drawdown': r.max_drawdown,
                'total_signals': r.total_signals,
                'false_positives': r.false_positives,
                'avg_profit': r.avg_profit_per_signal,
                'parameters': json.dumps(r.parameters)
            }
            for r in self.results
        ])
        
        return results_df
    
    def _segment_data_by_scenario(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Segment data into different market scenarios
        
        Args:
            data: Historical data
            
        Returns:
            Dictionary of scenario -> data segments
        """
        segments = {}
        
        # Calculate market metrics
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(20).std()
        data['trend'] = data['close'].rolling(50).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 50 else 0
        )
        
        # Classify each period
        for scenario, criteria in self.scenarios.items():
            vol_min, vol_max = criteria['volatility']
            trend_threshold = criteria['trend_strength']
            
            if scenario in ['trending']:
                mask = (data['volatility'] >= vol_min) & \
                       (data['volatility'] <= vol_max) & \
                       (abs(data['trend']) > trend_threshold)
            elif scenario in ['ranging']:
                mask = (data['volatility'] >= vol_min) & \
                       (data['volatility'] <= vol_max) & \
                       (abs(data['trend']) < trend_threshold)
            else:
                mask = (data['volatility'] >= vol_min) & \
                       (data['volatility'] <= vol_max)
            
            segments[scenario] = data[mask].copy()
        
        return segments
    
    def _generate_parameter_combinations(self, indicator_name: str) -> List[Dict]:
        """Generate all parameter combinations for an indicator"""
        param_grid = self.parameter_grids[indicator_name]
        keys = param_grid.keys()
        values = param_grid.values()
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to specified timeframe"""
        # Map timeframe to pandas frequency
        freq_map = {
            '5min': '5T',
            '15min': '15T',
            '30min': '30T',
            '1hour': '1H'
        }
        
        freq = freq_map.get(timeframe, '5T')
        
        resampled = data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def _test_indicator(self, indicator_name: str, params: Dict, 
                       data: pd.DataFrame, scenario: str, 
                       timeframe: str) -> Optional[IndicatorPerformance]:
        """
        Test a specific indicator with given parameters
        
        Args:
            indicator_name: Name of indicator
            params: Parameter dictionary
            data: Test data
            scenario: Market scenario
            timeframe: Data timeframe
            
        Returns:
            IndicatorPerformance object or None
        """
        try:
            # Initialize indicator based on name
            if indicator_name == 'rsi':
                indicator = AdvancedRSI(
                    period=params['period'],
                    overbought=params['overbought'],
                    oversold=params['oversold']
                )
                signals = indicator.generate_signals(data['close'])
                
            elif indicator_name == 'oscillator_matrix':
                config = {
                    'rsi': {'period': params['rsi_period']},
                    'macd': {'fast': params['macd_fast'], 'slow': params['macd_slow']}
                }
                indicator = OscillatorMatrix(config)
                signals = indicator.generate_signals(data)
                
            elif indicator_name == 'momentum':
                indicator = MomentumIndicators()
                # Convert to format expected by momentum indicator
                momentum_signals = []
                for i in range(params['fast_period'], len(data)):
                    mom = indicator.calculate_momentum(
                        data['close'].values, 
                        params['fast_period']
                    )
                    if abs(mom[i]) > params['threshold']:
                        momentum_signals.append({
                            'timestamp': data.index[i],
                            'type': 'BUY' if mom[i] > 0 else 'SELL',
                            'strength': abs(mom[i])
                        })
                signals = momentum_signals
                
            else:
                return None
            
            if not signals:
                return None
            
            # Backtest signals
            metrics = self._backtest_signals(signals, data)
            
            # Create performance object
            performance = IndicatorPerformance(
                indicator_name=indicator_name,
                parameters=params,
                scenario=scenario,
                win_rate=metrics['win_rate'],
                profit_factor=metrics['profit_factor'],
                sharpe_ratio=metrics['sharpe_ratio'],
                max_drawdown=metrics['max_drawdown'],
                total_signals=metrics['total_signals'],
                false_positives=metrics['false_positives'],
                avg_profit_per_signal=metrics['avg_profit'],
                best_timeframe=timeframe,
                correlation_with_others={}
            )
            
            return performance
            
        except Exception as e:
            print(f"Error testing {indicator_name}: {e}")
            return None
    
    def _backtest_signals(self, signals: List, data: pd.DataFrame) -> Dict[str, float]:
        """
        Backtest signals and calculate performance metrics
        
        Args:
            signals: List of signals
            data: Price data
            
        Returns:
            Dictionary of performance metrics
        """
        if not signals:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_signals': 0,
                'false_positives': 0,
                'avg_profit': 0
            }
        
        trades = []
        
        for signal in signals:
            # Get signal timestamp
            if hasattr(signal, 'timestamp'):
                signal_time = signal.timestamp
                signal_type = getattr(signal, 'signal_type', 'BUY')
            else:
                signal_time = signal.get('timestamp')
                signal_type = signal.get('type', 'BUY')
            
            if signal_time not in data.index:
                continue
            
            # Find entry and exit
            entry_idx = data.index.get_loc(signal_time)
            if entry_idx + 10 >= len(data):
                continue
            
            entry_price = data['close'].iloc[entry_idx]
            exit_price = data['close'].iloc[entry_idx + 10]  # Exit after 10 bars
            
            # Calculate return based on signal type
            if 'BUY' in str(signal_type) or 'oversold' in str(signal_type) or 'bullish' in str(signal_type):
                returns = (exit_price - entry_price) / entry_price
            else:
                returns = (entry_price - exit_price) / entry_price
            
            trades.append(returns)
        
        if not trades:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_signals': len(signals),
                'false_positives': 0,
                'avg_profit': 0
            }
        
        trades_array = np.array(trades)
        
        # Calculate metrics
        win_rate = len(trades_array[trades_array > 0]) / len(trades_array)
        
        # Profit factor
        gross_profit = trades_array[trades_array > 0].sum() if len(trades_array[trades_array > 0]) > 0 else 0
        gross_loss = abs(trades_array[trades_array < 0].sum()) if len(trades_array[trades_array < 0]) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Sharpe ratio (simplified)
        sharpe_ratio = trades_array.mean() / (trades_array.std() + 1e-6) * np.sqrt(252)
        
        # Max drawdown
        cumulative_returns = (1 + trades_array).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # False positives (losses > 2%)
        false_positives = len(trades_array[trades_array < -0.02])
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_signals': len(signals),
            'false_positives': false_positives,
            'avg_profit': trades_array.mean()
        }
    
    def find_best_indicators_for_scenario(self, scenario: str) -> pd.DataFrame:
        """
        Find best performing indicators for a specific market scenario
        
        Args:
            scenario: Market scenario ('trending', 'ranging', etc.)
            
        Returns:
            DataFrame with top indicators for the scenario
        """
        scenario_results = [r for r in self.results if r.scenario == scenario]
        
        if not scenario_results:
            return pd.DataFrame()
        
        # Sort by composite score
        sorted_results = sorted(
            scenario_results,
            key=lambda x: x.win_rate * 0.3 + x.sharpe_ratio * 0.3 + x.profit_factor * 0.2 - x.false_positives * 0.2,
            reverse=True
        )
        
        # Get top 5
        top_indicators = sorted_results[:5]
        
        return pd.DataFrame([
            {
                'rank': i + 1,
                'indicator': r.indicator_name,
                'timeframe': r.best_timeframe,
                'win_rate': f"{r.win_rate:.2%}",
                'sharpe_ratio': f"{r.sharpe_ratio:.2f}",
                'profit_factor': f"{r.profit_factor:.2f}",
                'parameters': r.parameters
            }
            for i, r in enumerate(top_indicators)
        ])
    
    def create_indicator_heatmap(self):
        """Create heatmap showing indicator performance across scenarios"""
        if not self.results:
            print("No results to visualize")
            return
        
        # Create pivot table
        pivot_data = {}
        for result in self.results:
            key = f"{result.indicator_name}"
            if key not in pivot_data:
                pivot_data[key] = {}
            
            # Use best performance for each scenario
            if result.scenario not in pivot_data[key] or \
               result.sharpe_ratio > pivot_data[key][result.scenario]:
                pivot_data[key][result.scenario] = result.sharpe_ratio
        
        # Convert to DataFrame
        heatmap_df = pd.DataFrame(pivot_data).T
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
        plt.title('Indicator Performance Heatmap (Sharpe Ratio)')
        plt.xlabel('Market Scenario')
        plt.ylabel('Indicator')
        plt.tight_layout()
        plt.savefig('reports/indicator_performance_heatmap.png')
        plt.close()
        
        print("Heatmap saved to reports/indicator_performance_heatmap.png")
    
    def generate_recommendation_report(self) -> str:
        """Generate recommendation report for indicator usage"""
        report = []
        report.append("# Indicator Performance Analysis Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Executive Summary\n")
        
        # Overall best indicators
        all_sharpe = [(r.indicator_name, r.sharpe_ratio) for r in self.results]
        best_overall = sorted(all_sharpe, key=lambda x: x[1], reverse=True)[:3]
        
        report.append("### Top 3 Indicators Overall:")
        for i, (indicator, sharpe) in enumerate(best_overall):
            report.append(f"{i+1}. **{indicator}** - Sharpe Ratio: {sharpe:.2f}")
        
        # Recommendations by scenario
        report.append("\n## Scenario-Based Recommendations\n")
        
        for scenario in self.scenarios.keys():
            report.append(f"### {scenario.capitalize()} Market:")
            best_df = self.find_best_indicators_for_scenario(scenario)
            
            if not best_df.empty:
                report.append(best_df.to_string(index=False))
            else:
                report.append("No data available for this scenario")
            
            report.append("")
        
        # Parameter optimization insights
        report.append("\n## Key Parameter Insights\n")
        
        # RSI insights
        rsi_results = [r for r in self.results if r.indicator_name == 'rsi']
        if rsi_results:
            best_rsi = max(rsi_results, key=lambda x: x.sharpe_ratio)
            report.append(f"**RSI**: Best parameters - Period: {best_rsi.parameters['period']}, "
                         f"OB: {best_rsi.parameters['overbought']}, OS: {best_rsi.parameters['oversold']}")
        
        # Save report
        report_text = "\n".join(report)
        with open('reports/indicator_recommendations.md', 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def _calculate_indicator_correlations(self):
        """Calculate correlations between different indicators"""
        # This would require tracking actual signals
        # For now, we'll leave correlation_with_others empty
        pass