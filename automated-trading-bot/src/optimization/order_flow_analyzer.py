"""
Order Flow Analyzer for Market Microstructure Analysis
Analyzes order imbalances, VWAP deviations, and trade intensity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque
from enum import Enum
import statistics


class OrderType(Enum):
    """Types of orders"""
    MARKET_BUY = "market_buy"
    MARKET_SELL = "market_sell"
    LIMIT_BUY = "limit_buy"
    LIMIT_SELL = "limit_sell"
    AGGRESSIVE_BUY = "aggressive_buy"
    AGGRESSIVE_SELL = "aggressive_sell"
    PASSIVE = "passive"


class TradeClassification(Enum):
    """Trade classification types"""
    BUYER_INITIATED = "buyer_initiated"
    SELLER_INITIATED = "seller_initiated"
    NEUTRAL = "neutral"


@dataclass
class OrderFlowMetrics:
    """Container for order flow analysis metrics"""
    timestamp: datetime
    order_imbalance: float  # -1 to 1 (negative = sell pressure)
    normalized_imbalance: float
    vwap: float
    vwap_deviation: float
    trade_intensity: float
    large_order_ratio: float
    aggressive_ratio: float
    bid_ask_imbalance: float
    cumulative_delta: float
    price_impact: float
    

@dataclass
class MarketDepthAnalysis:
    """Market depth and liquidity analysis"""
    timestamp: datetime
    bid_depth: List[Tuple[float, int]]  # [(price, size), ...]
    ask_depth: List[Tuple[float, int]]
    total_bid_volume: int
    total_ask_volume: int
    bid_ask_ratio: float
    depth_imbalance: float
    weighted_mid_price: float
    liquidity_score: float


@dataclass
class OrderFlowSignal:
    """Trading signal from order flow analysis"""
    signal_type: str  # 'buy', 'sell', 'neutral'
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    reasons: List[str]
    metrics: OrderFlowMetrics


class OrderFlowAnalyzer:
    """
    Analyzes order flow to detect institutional activity and market sentiment
    
    Features:
    - Order imbalance detection
    - VWAP analysis and deviations
    - Trade intensity metrics
    - Large order detection
    - Bid-ask pressure analysis
    - Cumulative volume delta
    - Market depth analysis
    """
    
    def __init__(self,
                 large_order_threshold: int = 10000,
                 imbalance_window: int = 100,
                 vwap_window: int = 300,
                 intensity_window: int = 60):
        """Initialize order flow analyzer
        
        Args:
            large_order_threshold: Volume threshold for large orders
            imbalance_window: Window size for order imbalance calculation
            vwap_window: Window size for VWAP calculation (seconds)
            intensity_window: Window for trade intensity calculation (seconds)
        """
        self.logger = logging.getLogger(__name__)
        self.large_order_threshold = large_order_threshold
        self.imbalance_window = imbalance_window
        self.vwap_window = vwap_window
        self.intensity_window = intensity_window
        
        # State tracking
        self.order_history = deque(maxlen=10000)
        self.trade_history = deque(maxlen=10000)
        self.depth_snapshots = deque(maxlen=1000)
        
        # Cumulative metrics
        self.cumulative_buy_volume = 0
        self.cumulative_sell_volume = 0
        self.session_vwap = None
        
    def analyze_tick_data(self, 
                         tick_data: pd.DataFrame,
                         depth_data: Optional[pd.DataFrame] = None) -> List[OrderFlowMetrics]:
        """Analyze tick data for order flow patterns
        
        Args:
            tick_data: DataFrame with tick data (price, volume, timestamp, bid, ask)
            depth_data: Optional market depth snapshots
            
        Returns:
            List of OrderFlowMetrics for each time window
        """
        self.logger.info(f"Analyzing {len(tick_data)} ticks for order flow")
        
        # Classify trades
        tick_data['trade_type'] = self._classify_trades(tick_data)
        
        # Group by time windows
        tick_data['time_bucket'] = pd.to_datetime(tick_data['timestamp']).dt.floor('1min')
        
        metrics_list = []
        
        for time_bucket, group in tick_data.groupby('time_bucket'):
            # Calculate metrics for this time window
            metrics = self._calculate_window_metrics(group, time_bucket)
            metrics_list.append(metrics)
            
            # Update cumulative metrics
            self._update_cumulative_metrics(group)
        
        return metrics_list
    
    def analyze_real_time_order(self, 
                              price: float,
                              volume: int,
                              timestamp: datetime,
                              bid: float,
                              ask: float,
                              bid_size: int,
                              ask_size: int) -> OrderFlowMetrics:
        """Analyze a single order in real-time
        
        Args:
            price: Trade price
            volume: Trade volume
            timestamp: Trade timestamp
            bid: Best bid price
            ask: Best ask price
            bid_size: Bid size
            ask_size: Ask size
            
        Returns:
            Current OrderFlowMetrics
        """
        # Create trade record
        trade = {
            'price': price,
            'volume': volume,
            'timestamp': timestamp,
            'bid': bid,
            'ask': ask,
            'bid_size': bid_size,
            'ask_size': ask_size
        }
        
        # Classify trade
        trade['trade_type'] = self._classify_single_trade(trade)
        
        # Add to history
        self.trade_history.append(trade)
        
        # Calculate current metrics
        recent_trades = [t for t in self.trade_history 
                        if t['timestamp'] > timestamp - timedelta(seconds=self.imbalance_window)]
        
        return self._calculate_current_metrics(recent_trades, timestamp)
    
    def _classify_trades(self, tick_data: pd.DataFrame) -> pd.Series:
        """Classify trades as buyer or seller initiated"""
        classifications = []
        
        for _, trade in tick_data.iterrows():
            classification = self._classify_single_trade(trade.to_dict())
            classifications.append(classification)
        
        return pd.Series(classifications, index=tick_data.index)
    
    def _classify_single_trade(self, trade: Dict[str, Any]) -> TradeClassification:
        """Classify a single trade using tick rule and quote rule"""
        price = trade['price']
        bid = trade.get('bid', price)
        ask = trade.get('ask', price)
        
        # Quote rule
        mid_price = (bid + ask) / 2
        
        if price > mid_price:
            return TradeClassification.BUYER_INITIATED
        elif price < mid_price:
            return TradeClassification.SELLER_INITIATED
        else:
            # Use tick rule as tiebreaker
            if 'prev_price' in trade and trade['prev_price'] is not None:
                if price > trade['prev_price']:
                    return TradeClassification.BUYER_INITIATED
                elif price < trade['prev_price']:
                    return TradeClassification.SELLER_INITIATED
        
        return TradeClassification.NEUTRAL
    
    def _calculate_window_metrics(self, 
                                trades: pd.DataFrame,
                                timestamp: datetime) -> OrderFlowMetrics:
        """Calculate order flow metrics for a time window"""
        # Order imbalance
        buy_volume = trades[trades['trade_type'] == TradeClassification.BUYER_INITIATED]['volume'].sum()
        sell_volume = trades[trades['trade_type'] == TradeClassification.SELLER_INITIATED]['volume'].sum()
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            order_imbalance = (buy_volume - sell_volume) / total_volume
            normalized_imbalance = (buy_volume - sell_volume) / np.sqrt(total_volume)
        else:
            order_imbalance = 0
            normalized_imbalance = 0
        
        # VWAP calculation
        vwap = self._calculate_vwap(trades)
        current_price = trades.iloc[-1]['price'] if len(trades) > 0 else 0
        vwap_deviation = (current_price - vwap) / vwap if vwap > 0 else 0
        
        # Trade intensity
        time_span = (trades['timestamp'].max() - trades['timestamp'].min()).total_seconds()
        trade_intensity = len(trades) / max(time_span, 1)
        
        # Large order ratio
        large_orders = trades[trades['volume'] >= self.large_order_threshold]
        large_order_ratio = large_orders['volume'].sum() / total_volume if total_volume > 0 else 0
        
        # Aggressive order ratio
        aggressive_ratio = self._calculate_aggressive_ratio(trades)
        
        # Bid-ask imbalance
        bid_ask_imbalance = self._calculate_bid_ask_imbalance(trades)
        
        # Cumulative delta
        cumulative_delta = self.cumulative_buy_volume - self.cumulative_sell_volume
        
        # Price impact
        price_impact = self._calculate_price_impact(trades)
        
        return OrderFlowMetrics(
            timestamp=timestamp,
            order_imbalance=order_imbalance,
            normalized_imbalance=normalized_imbalance,
            vwap=vwap,
            vwap_deviation=vwap_deviation,
            trade_intensity=trade_intensity,
            large_order_ratio=large_order_ratio,
            aggressive_ratio=aggressive_ratio,
            bid_ask_imbalance=bid_ask_imbalance,
            cumulative_delta=cumulative_delta,
            price_impact=price_impact
        )
    
    def _calculate_current_metrics(self,
                                 recent_trades: List[Dict],
                                 timestamp: datetime) -> OrderFlowMetrics:
        """Calculate metrics from recent trade history"""
        if not recent_trades:
            return OrderFlowMetrics(
                timestamp=timestamp,
                order_imbalance=0,
                normalized_imbalance=0,
                vwap=0,
                vwap_deviation=0,
                trade_intensity=0,
                large_order_ratio=0,
                aggressive_ratio=0,
                bid_ask_imbalance=0,
                cumulative_delta=0,
                price_impact=0
            )
        
        # Convert to DataFrame for easier calculation
        df = pd.DataFrame(recent_trades)
        return self._calculate_window_metrics(df, timestamp)
    
    def _calculate_vwap(self, trades: pd.DataFrame) -> float:
        """Calculate volume-weighted average price"""
        if trades.empty or trades['volume'].sum() == 0:
            return 0
        
        return (trades['price'] * trades['volume']).sum() / trades['volume'].sum()
    
    def _calculate_aggressive_ratio(self, trades: pd.DataFrame) -> float:
        """Calculate ratio of aggressive orders (market orders at bid/ask)"""
        if trades.empty:
            return 0
        
        # Aggressive orders are those that cross the spread
        spread = trades['ask'] - trades['bid']
        avg_spread = spread.mean()
        
        # Orders at or beyond bid/ask are considered aggressive
        aggressive = trades[
            (trades['price'] >= trades['ask'] - 0.1 * avg_spread) | 
            (trades['price'] <= trades['bid'] + 0.1 * avg_spread)
        ]
        
        return len(aggressive) / len(trades)
    
    def _calculate_bid_ask_imbalance(self, trades: pd.DataFrame) -> float:
        """Calculate bid-ask volume imbalance"""
        if trades.empty or 'bid_size' not in trades.columns:
            return 0
        
        avg_bid_size = trades['bid_size'].mean()
        avg_ask_size = trades['ask_size'].mean()
        
        total_size = avg_bid_size + avg_ask_size
        if total_size > 0:
            return (avg_bid_size - avg_ask_size) / total_size
        
        return 0
    
    def _calculate_price_impact(self, trades: pd.DataFrame) -> float:
        """Calculate average price impact of trades"""
        if len(trades) < 2:
            return 0
        
        # Simple price impact: price change per unit volume
        price_changes = trades['price'].diff().abs()
        volumes = trades['volume']
        
        # Calculate impact for each trade
        impacts = []
        for i in range(1, len(trades)):
            if volumes.iloc[i] > 0:
                impact = price_changes.iloc[i] / volumes.iloc[i]
                impacts.append(impact)
        
        return np.mean(impacts) if impacts else 0
    
    def _update_cumulative_metrics(self, trades: pd.DataFrame):
        """Update cumulative metrics"""
        buy_volume = trades[trades['trade_type'] == TradeClassification.BUYER_INITIATED]['volume'].sum()
        sell_volume = trades[trades['trade_type'] == TradeClassification.SELLER_INITIATED]['volume'].sum()
        
        self.cumulative_buy_volume += buy_volume
        self.cumulative_sell_volume += sell_volume
    
    def analyze_market_depth(self, 
                           bid_levels: List[Tuple[float, int]],
                           ask_levels: List[Tuple[float, int]],
                           timestamp: datetime) -> MarketDepthAnalysis:
        """Analyze market depth and liquidity
        
        Args:
            bid_levels: List of (price, size) tuples for bids
            ask_levels: List of (price, size) tuples for asks
            timestamp: Snapshot timestamp
            
        Returns:
            MarketDepthAnalysis object
        """
        # Calculate total volumes
        total_bid_volume = sum(size for _, size in bid_levels)
        total_ask_volume = sum(size for _, size in ask_levels)
        
        # Bid-ask ratio
        bid_ask_ratio = total_bid_volume / total_ask_volume if total_ask_volume > 0 else 1.0
        
        # Depth imbalance
        total_volume = total_bid_volume + total_ask_volume
        depth_imbalance = (total_bid_volume - total_ask_volume) / total_volume if total_volume > 0 else 0
        
        # Weighted mid price (liquidity-weighted)
        if bid_levels and ask_levels:
            best_bid, best_bid_size = bid_levels[0]
            best_ask, best_ask_size = ask_levels[0]
            
            total_best_size = best_bid_size + best_ask_size
            if total_best_size > 0:
                weighted_mid = (best_bid * best_ask_size + best_ask * best_bid_size) / total_best_size
            else:
                weighted_mid = (best_bid + best_ask) / 2
        else:
            weighted_mid = 0
        
        # Liquidity score (based on depth and spread)
        liquidity_score = self._calculate_liquidity_score(bid_levels, ask_levels)
        
        # Store snapshot
        depth_snapshot = {
            'timestamp': timestamp,
            'bid_levels': bid_levels,
            'ask_levels': ask_levels,
            'analysis': None  # Will be filled below
        }
        
        analysis = MarketDepthAnalysis(
            timestamp=timestamp,
            bid_depth=bid_levels,
            ask_depth=ask_levels,
            total_bid_volume=total_bid_volume,
            total_ask_volume=total_ask_volume,
            bid_ask_ratio=bid_ask_ratio,
            depth_imbalance=depth_imbalance,
            weighted_mid_price=weighted_mid,
            liquidity_score=liquidity_score
        )
        
        depth_snapshot['analysis'] = analysis
        self.depth_snapshots.append(depth_snapshot)
        
        return analysis
    
    def _calculate_liquidity_score(self,
                                 bid_levels: List[Tuple[float, int]],
                                 ask_levels: List[Tuple[float, int]]) -> float:
        """Calculate liquidity score based on depth and spread"""
        if not bid_levels or not ask_levels:
            return 0
        
        # Factors for liquidity score
        # 1. Tight spread
        spread = ask_levels[0][0] - bid_levels[0][0]
        mid_price = (ask_levels[0][0] + bid_levels[0][0]) / 2
        spread_bps = (spread / mid_price) * 10000  # Basis points
        spread_score = max(0, 1 - spread_bps / 100)  # 100bps = 0 score
        
        # 2. Deep book
        total_volume = sum(s for _, s in bid_levels) + sum(s for _, s in ask_levels)
        depth_score = min(1, total_volume / 100000)  # Normalize by 100k
        
        # 3. Balanced book
        bid_volume = sum(s for _, s in bid_levels)
        ask_volume = sum(s for _, s in ask_levels)
        balance = min(bid_volume, ask_volume) / max(bid_volume, ask_volume) if max(bid_volume, ask_volume) > 0 else 0
        
        # Combined score
        liquidity_score = 0.4 * spread_score + 0.4 * depth_score + 0.2 * balance
        
        return liquidity_score
    
    def detect_order_flow_patterns(self, 
                                 metrics_history: List[OrderFlowMetrics]) -> Dict[str, Any]:
        """Detect patterns in order flow history
        
        Args:
            metrics_history: List of historical OrderFlowMetrics
            
        Returns:
            Dictionary of detected patterns
        """
        patterns = {
            'accumulation': False,
            'distribution': False,
            'absorption': False,
            'exhaustion': False,
            'momentum': None,
            'institutional_activity': False
        }
        
        if len(metrics_history) < 10:
            return patterns
        
        # Convert to arrays for analysis
        imbalances = [m.order_imbalance for m in metrics_history]
        intensities = [m.trade_intensity for m in metrics_history]
        large_ratios = [m.large_order_ratio for m in metrics_history]
        cumulative_deltas = [m.cumulative_delta for m in metrics_history]
        
        # Accumulation pattern: persistent positive imbalance with increasing volume
        recent_imbalance = np.mean(imbalances[-5:])
        recent_intensity = np.mean(intensities[-5:])
        prev_intensity = np.mean(intensities[-10:-5])
        
        if recent_imbalance > 0.2 and recent_intensity > prev_intensity * 1.2:
            patterns['accumulation'] = True
        
        # Distribution pattern: persistent negative imbalance with increasing volume
        if recent_imbalance < -0.2 and recent_intensity > prev_intensity * 1.2:
            patterns['distribution'] = True
        
        # Absorption: high volume with minimal price impact
        recent_impacts = [m.price_impact for m in metrics_history[-5:]]
        if recent_intensity > np.mean(intensities) * 1.5 and np.mean(recent_impacts) < np.mean([m.price_impact for m in metrics_history]) * 0.5:
            patterns['absorption'] = True
        
        # Exhaustion: decreasing intensity after extreme move
        if len(metrics_history) > 20:
            peak_intensity = max(intensities[-20:-10])
            current_intensity = np.mean(intensities[-5:])
            if current_intensity < peak_intensity * 0.3:
                patterns['exhaustion'] = True
        
        # Momentum direction
        delta_trend = np.polyfit(range(len(cumulative_deltas[-10:])), cumulative_deltas[-10:], 1)[0]
        if delta_trend > 0:
            patterns['momentum'] = 'bullish'
        elif delta_trend < 0:
            patterns['momentum'] = 'bearish'
        else:
            patterns['momentum'] = 'neutral'
        
        # Institutional activity: large order ratio consistently high
        if np.mean(large_ratios[-10:]) > 0.3:
            patterns['institutional_activity'] = True
        
        return patterns
    
    def generate_order_flow_signal(self, 
                                 current_metrics: OrderFlowMetrics,
                                 metrics_history: List[OrderFlowMetrics],
                                 depth_analysis: Optional[MarketDepthAnalysis] = None) -> OrderFlowSignal:
        """Generate trading signal from order flow analysis
        
        Args:
            current_metrics: Current order flow metrics
            metrics_history: Historical metrics
            depth_analysis: Optional market depth analysis
            
        Returns:
            OrderFlowSignal with trading recommendation
        """
        signal_strength = 0.0
        signal_type = 'neutral'
        reasons = []
        confidence = 0.5
        
        # Check order imbalance
        if current_metrics.order_imbalance > 0.3:
            signal_strength += 0.3
            signal_type = 'buy'
            reasons.append(f"Strong buy imbalance: {current_metrics.order_imbalance:.2f}")
        elif current_metrics.order_imbalance < -0.3:
            signal_strength -= 0.3
            signal_type = 'sell'
            reasons.append(f"Strong sell imbalance: {current_metrics.order_imbalance:.2f}")
        
        # Check VWAP deviation
        if current_metrics.vwap_deviation > 0.01:  # Price above VWAP
            signal_strength += 0.2
            reasons.append("Price above VWAP - bullish")
        elif current_metrics.vwap_deviation < -0.01:  # Price below VWAP
            signal_strength -= 0.2
            reasons.append("Price below VWAP - bearish")
        
        # Check cumulative delta trend
        if len(metrics_history) >= 10:
            deltas = [m.cumulative_delta for m in metrics_history[-10:]]
            delta_trend = np.polyfit(range(len(deltas)), deltas, 1)[0]
            
            if delta_trend > 0:
                signal_strength += 0.2
                reasons.append("Positive cumulative delta trend")
            elif delta_trend < 0:
                signal_strength -= 0.2
                reasons.append("Negative cumulative delta trend")
        
        # Check depth imbalance if available
        if depth_analysis:
            if depth_analysis.depth_imbalance > 0.2:
                signal_strength += 0.15
                reasons.append("Strong bid depth")
            elif depth_analysis.depth_imbalance < -0.2:
                signal_strength -= 0.15
                reasons.append("Strong ask depth")
        
        # Check for institutional activity
        if current_metrics.large_order_ratio > 0.4:
            confidence += 0.1
            reasons.append("High institutional activity")
        
        # Detect patterns
        patterns = self.detect_order_flow_patterns(metrics_history)
        
        if patterns['accumulation']:
            signal_strength += 0.2
            signal_type = 'buy' if signal_strength > 0 else signal_type
            reasons.append("Accumulation pattern detected")
            confidence += 0.1
        elif patterns['distribution']:
            signal_strength -= 0.2
            signal_type = 'sell' if signal_strength < 0 else signal_type
            reasons.append("Distribution pattern detected")
            confidence += 0.1
        
        # Normalize signal strength
        signal_strength = max(-1, min(1, signal_strength))
        
        # Determine final signal type
        if abs(signal_strength) < 0.2:
            signal_type = 'neutral'
        elif signal_strength > 0:
            signal_type = 'buy'
        else:
            signal_type = 'sell'
        
        # Calculate confidence
        confidence = min(1.0, confidence + abs(signal_strength) * 0.3)
        
        return OrderFlowSignal(
            signal_type=signal_type,
            strength=abs(signal_strength),
            confidence=confidence,
            reasons=reasons,
            metrics=current_metrics
        )
    
    def calculate_microstructure_features(self, 
                                        tick_data: pd.DataFrame,
                                        window_size: int = 100) -> pd.DataFrame:
        """Calculate microstructure features for ML models
        
        Args:
            tick_data: Tick data DataFrame
            window_size: Rolling window size
            
        Returns:
            DataFrame with microstructure features
        """
        features = pd.DataFrame(index=tick_data.index)
        
        # Classify trades
        tick_data['trade_type'] = self._classify_trades(tick_data)
        
        # Rolling order imbalance
        tick_data['buy_volume'] = np.where(
            tick_data['trade_type'] == TradeClassification.BUYER_INITIATED,
            tick_data['volume'], 0
        )
        tick_data['sell_volume'] = np.where(
            tick_data['trade_type'] == TradeClassification.SELLER_INITIATED,
            tick_data['volume'], 0
        )
        
        rolling_buy = tick_data['buy_volume'].rolling(window_size).sum()
        rolling_sell = tick_data['sell_volume'].rolling(window_size).sum()
        total_vol = rolling_buy + rolling_sell
        
        features['order_imbalance'] = np.where(
            total_vol > 0,
            (rolling_buy - rolling_sell) / total_vol,
            0
        )
        
        # Normalized imbalance (Easley et al.)
        features['normalized_imbalance'] = np.where(
            total_vol > 0,
            (rolling_buy - rolling_sell) / np.sqrt(total_vol),
            0
        )
        
        # Trade intensity
        features['trade_intensity'] = tick_data['volume'].rolling(window_size).count() / window_size
        
        # Large trade indicator
        features['large_trade_ratio'] = (
            tick_data['volume'].rolling(window_size).apply(
                lambda x: (x >= self.large_order_threshold).sum() / len(x)
            )
        )
        
        # Bid-ask pressure
        if 'bid_size' in tick_data.columns and 'ask_size' in tick_data.columns:
            features['bid_ask_pressure'] = (
                tick_data['bid_size'] - tick_data['ask_size']
            ) / (
                tick_data['bid_size'] + tick_data['ask_size']
            )
        
        # Price impact
        tick_data['price_change'] = tick_data['price'].diff().abs()
        features['avg_price_impact'] = (
            tick_data['price_change'] / tick_data['volume']
        ).rolling(window_size).mean()
        
        # Effective spread
        if 'bid' in tick_data.columns and 'ask' in tick_data.columns:
            tick_data['mid_price'] = (tick_data['bid'] + tick_data['ask']) / 2
            features['effective_spread'] = 2 * abs(tick_data['price'] - tick_data['mid_price'])
            features['relative_spread'] = features['effective_spread'] / tick_data['mid_price']
        
        # Quote asymmetry
        if 'bid' in tick_data.columns and 'ask' in tick_data.columns:
            features['quote_asymmetry'] = (
                (tick_data['ask'] - tick_data['mid_price']) - 
                (tick_data['mid_price'] - tick_data['bid'])
            ) / (tick_data['ask'] - tick_data['bid'])
        
        # Trade direction autocorrelation
        trade_direction = np.where(
            tick_data['trade_type'] == TradeClassification.BUYER_INITIATED, 1,
            np.where(tick_data['trade_type'] == TradeClassification.SELLER_INITIATED, -1, 0)
        )
        features['trade_autocorr'] = pd.Series(trade_direction).rolling(window_size).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
        )
        
        return features
    
    def estimate_pin(self, 
                    tick_data: pd.DataFrame,
                    num_days: int = 60) -> float:
        """Estimate Probability of Informed Trading (PIN)
        
        Simplified PIN estimation based on Easley et al.
        
        Args:
            tick_data: Tick data with trade classification
            num_days: Number of days for estimation
            
        Returns:
            PIN estimate (0 to 1)
        """
        if len(tick_data) < num_days * 100:  # Need sufficient data
            return 0.5  # Default uninformative prior
        
        # Group by day
        tick_data['date'] = pd.to_datetime(tick_data['timestamp']).dt.date
        
        daily_stats = []
        
        for date, day_data in tick_data.groupby('date'):
            buy_count = (day_data['trade_type'] == TradeClassification.BUYER_INITIATED).sum()
            sell_count = (day_data['trade_type'] == TradeClassification.SELLER_INITIATED).sum()
            
            daily_stats.append({
                'date': date,
                'buys': buy_count,
                'sells': sell_count,
                'total': buy_count + sell_count
            })
        
        if not daily_stats:
            return 0.5
        
        df_daily = pd.DataFrame(daily_stats)
        
        # Simple PIN approximation
        # High imbalance days likely have informed trading
        df_daily['imbalance'] = abs(df_daily['buys'] - df_daily['sells']) / df_daily['total']
        
        # Days with high imbalance (top quartile)
        high_imbalance_threshold = df_daily['imbalance'].quantile(0.75)
        informed_days = (df_daily['imbalance'] > high_imbalance_threshold).sum()
        
        # PIN estimate
        pin = informed_days / len(df_daily)
        
        return min(max(pin, 0), 1)  # Ensure in [0, 1]
    
    def reset_state(self):
        """Reset analyzer state"""
        self.order_history.clear()
        self.trade_history.clear()
        self.depth_snapshots.clear()
        self.cumulative_buy_volume = 0
        self.cumulative_sell_volume = 0
        self.session_vwap = None
        self.logger.info("Order flow analyzer state reset")