"""
Enhanced Historical Data Collector for Trading Bot System
Supports tick data, multi-timeframe OHLCV, options chain, and market microstructure data
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, field
import gzip
import json
from collections import deque
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import pytz

from src.integrations.openalgo_client import OpenAlgoClient
from src.core.database import DatabaseManager


@dataclass
class TickData:
    """Container for tick-by-tick market data"""
    timestamp: datetime
    symbol: str
    price: float
    volume: int
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    trade_type: str = "NORMAL"  # NORMAL, BLOCK, SWEEP
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'trade_type': self.trade_type
        }


@dataclass
class MarketMicrostructure:
    """Container for market microstructure data"""
    timestamp: datetime
    symbol: str
    bid_ask_spread: float
    depth_imbalance: float
    order_book_levels: List[Dict[str, float]]
    trade_intensity: float
    volume_weighted_price: float
    
    
@dataclass
class OptionsChainData:
    """Container for historical options chain data"""
    symbol: str
    expiry: str
    timestamp: datetime
    spot_price: float
    strikes: Dict[int, Dict[str, Any]]  # Strike -> {CE: data, PE: data}
    implied_volatility_surface: Optional[pd.DataFrame] = None
    greeks_data: Optional[pd.DataFrame] = None


class EnhancedHistoricalDataCollector:
    """
    Enhanced data collector with support for:
    - Tick data with nanosecond precision
    - Multi-timeframe OHLCV data
    - Options chain historical data
    - Market microstructure data
    - Data compression and efficient storage
    """
    
    def __init__(self, 
                 openalgo_client: Optional[OpenAlgoClient] = None,
                 db_manager: Optional[DatabaseManager] = None,
                 compression_enabled: bool = True):
        """Initialize enhanced data collector
        
        Args:
            openalgo_client: OpenAlgo API client
            db_manager: Database manager for storage
            compression_enabled: Enable data compression
        """
        self.openalgo_client = openalgo_client or OpenAlgoClient()
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.compression_enabled = compression_enabled
        
        # Caching layers
        self._tick_cache = deque(maxlen=100000)  # In-memory tick buffer
        self._ohlcv_cache: Dict[str, pd.DataFrame] = {}
        self._options_cache: Dict[str, OptionsChainData] = {}
        
        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(10)  # Max 10 concurrent requests
        self.request_delay = 0.1  # 100ms between requests
        
        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Supported timeframes
        self.timeframes = ['1min', '5min', '15min', '30min', '1hour', '4hour', '1day']
        
    async def collect_tick_data(self,
                              symbol: str,
                              start_date: datetime,
                              end_date: datetime,
                              store_to_db: bool = True) -> List[TickData]:
        """Collect historical tick data with high precision
        
        Args:
            symbol: Trading symbol
            start_date: Start datetime
            end_date: End datetime  
            store_to_db: Store collected data to database
            
        Returns:
            List of TickData objects
        """
        self.logger.info(f"Collecting tick data for {symbol} from {start_date} to {end_date}")
        
        all_ticks = []
        current_date = start_date
        
        while current_date < end_date:
            # Process day by day for tick data
            day_end = min(current_date + timedelta(days=1), end_date)
            
            async with self.rate_limiter:
                try:
                    # Fetch tick data from API
                    ticks = await self._fetch_tick_data(symbol, current_date, day_end)
                    
                    if ticks:
                        # Process and validate ticks
                        processed_ticks = await self._process_tick_data(ticks, symbol)
                        all_ticks.extend(processed_ticks)
                        
                        # Store to database if requested
                        if store_to_db and self.db_manager:
                            await self._store_tick_data_batch(processed_ticks)
                        
                        self.logger.info(f"Collected {len(processed_ticks)} ticks for {current_date.date()}")
                    
                except Exception as e:
                    self.logger.error(f"Error collecting tick data for {current_date}: {e}")
                
                await asyncio.sleep(self.request_delay)
            
            current_date = day_end
        
        self.logger.info(f"Total ticks collected: {len(all_ticks)}")
        return all_ticks
    
    async def collect_multi_timeframe_data(self,
                                         symbol: str,
                                         start_date: str,
                                         end_date: str,
                                         timeframes: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Collect OHLCV data for multiple timeframes
        
        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframes: List of timeframes (default: all supported)
            
        Returns:
            Dictionary of timeframe -> DataFrame
        """
        timeframes = timeframes or self.timeframes
        results = {}
        
        # Collect data for each timeframe in parallel
        tasks = []
        for tf in timeframes:
            task = self._collect_ohlcv_data(symbol, start_date, end_date, tf)
            tasks.append(task)
        
        # Wait for all tasks to complete
        completed_data = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for tf, data in zip(timeframes, completed_data):
            if isinstance(data, Exception):
                self.logger.error(f"Error collecting {tf} data: {data}")
            else:
                results[tf] = data
                self.logger.info(f"Collected {len(data)} {tf} candles for {symbol}")
        
        return results
    
    async def collect_options_chain_history(self,
                                          symbol: str,
                                          start_date: str,
                                          end_date: str,
                                          strikes: Optional[List[int]] = None,
                                          include_greeks: bool = True) -> List[OptionsChainData]:
        """Collect historical options chain data with Greeks
        
        Args:
            symbol: Underlying symbol
            start_date: Start date
            end_date: End date
            strikes: Specific strikes to collect (None for all)
            include_greeks: Include Greeks calculation
            
        Returns:
            List of OptionsChainData objects
        """
        self.logger.info(f"Collecting options chain history for {symbol}")
        
        options_data = []
        
        # Get all expiries in range
        expiries = await self._get_expiries_in_range(symbol, start_date, end_date)
        
        for expiry in expiries:
            try:
                # Collect daily snapshots for each expiry
                expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
                start_dt = max(datetime.strptime(start_date, "%Y-%m-%d"), 
                             expiry_date - timedelta(days=60))  # Max 60 days before expiry
                
                current_date = start_dt
                while current_date <= expiry_date:
                    async with self.rate_limiter:
                        # Fetch options chain snapshot
                        chain_data = await self._fetch_options_chain_snapshot(
                            symbol, expiry, current_date.strftime("%Y-%m-%d"), strikes
                        )
                        
                        if chain_data:
                            # Calculate Greeks if requested
                            if include_greeks:
                                chain_data.greeks_data = await self._calculate_option_greeks(chain_data)
                            
                            # Build IV surface
                            chain_data.implied_volatility_surface = self._build_iv_surface(chain_data)
                            
                            options_data.append(chain_data)
                        
                        await asyncio.sleep(self.request_delay)
                    
                    current_date += timedelta(days=1)
                    
            except Exception as e:
                self.logger.error(f"Error collecting options data for {expiry}: {e}")
        
        return options_data
    
    async def collect_market_microstructure(self,
                                          symbol: str,
                                          date: str,
                                          interval_seconds: int = 60) -> List[MarketMicrostructure]:
        """Collect market microstructure data
        
        Args:
            symbol: Trading symbol
            date: Date to collect (YYYY-MM-DD)
            interval_seconds: Snapshot interval in seconds
            
        Returns:
            List of MarketMicrostructure objects
        """
        self.logger.info(f"Collecting market microstructure for {symbol} on {date}")
        
        microstructure_data = []
        
        # Get tick data for the day
        start_dt = datetime.strptime(date, "%Y-%m-%d").replace(hour=9, minute=15)
        end_dt = start_dt.replace(hour=15, minute=30)
        
        ticks = await self.collect_tick_data(symbol, start_dt, end_dt, store_to_db=False)
        
        if not ticks:
            return microstructure_data
        
        # Convert to DataFrame for easier processing
        tick_df = pd.DataFrame([t.to_dict() for t in ticks])
        tick_df['timestamp'] = pd.to_datetime(tick_df['timestamp'])
        tick_df.set_index('timestamp', inplace=True)
        
        # Create snapshots at regular intervals
        current_time = start_dt
        while current_time < end_dt:
            window_end = current_time + timedelta(seconds=interval_seconds)
            
            # Get ticks in current window
            window_ticks = tick_df[current_time:window_end]
            
            if not window_ticks.empty:
                # Calculate microstructure metrics
                microstructure = self._calculate_microstructure_metrics(
                    window_ticks, symbol, current_time
                )
                microstructure_data.append(microstructure)
            
            current_time = window_end
        
        return microstructure_data
    
    async def _fetch_tick_data(self, symbol: str, start: datetime, end: datetime) -> List[Dict]:
        """Fetch raw tick data from API"""
        try:
            # This would call the actual API endpoint for tick data
            # For now, simulate with high-frequency data
            response = await self.openalgo_client._make_request(
                "GET",
                f"/historical/ticks/{symbol}",
                params={
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "type": "trades"
                }
            )
            return response.get('data', [])
        except Exception as e:
            self.logger.error(f"Error fetching tick data: {e}")
            return []
    
    async def _process_tick_data(self, raw_ticks: List[Dict], symbol: str) -> List[TickData]:
        """Process and validate raw tick data"""
        processed = []
        
        for tick in raw_ticks:
            try:
                # Validate and clean tick data
                if self._validate_tick(tick):
                    tick_obj = TickData(
                        timestamp=datetime.fromisoformat(tick['timestamp']),
                        symbol=symbol,
                        price=float(tick['price']),
                        volume=int(tick['volume']),
                        bid=float(tick.get('bid', tick['price'])),
                        ask=float(tick.get('ask', tick['price'])),
                        bid_size=int(tick.get('bid_size', 0)),
                        ask_size=int(tick.get('ask_size', 0)),
                        trade_type=self._classify_trade_type(tick)
                    )
                    processed.append(tick_obj)
            except Exception as e:
                self.logger.warning(f"Invalid tick data: {e}")
                
        return processed
    
    def _validate_tick(self, tick: Dict) -> bool:
        """Validate tick data integrity"""
        required_fields = ['timestamp', 'price', 'volume']
        
        # Check required fields
        if not all(field in tick for field in required_fields):
            return False
        
        # Validate price and volume
        if tick['price'] <= 0 or tick['volume'] < 0:
            return False
        
        # Check bid-ask spread validity
        if 'bid' in tick and 'ask' in tick:
            if tick['bid'] > tick['ask']:
                return False
        
        return True
    
    def _classify_trade_type(self, tick: Dict) -> str:
        """Classify trade type based on volume and other factors"""
        volume = tick.get('volume', 0)
        
        # Simple classification based on volume
        if volume > 50000:
            return "BLOCK"
        elif volume > 10000 and tick.get('aggressive', False):
            return "SWEEP"
        else:
            return "NORMAL"
    
    async def _store_tick_data_batch(self, ticks: List[TickData]):
        """Store tick data batch to database"""
        if not self.db_manager:
            return
        
        try:
            # Convert to DataFrame for bulk insert
            tick_records = [t.to_dict() for t in ticks]
            
            # Compress if enabled
            if self.compression_enabled:
                compressed_data = gzip.compress(json.dumps(tick_records).encode())
                # Store compressed data
                # Implementation depends on database schema
            else:
                # Store uncompressed
                # await self.db_manager.insert_tick_data_batch(tick_records)
                pass
                
        except Exception as e:
            self.logger.error(f"Error storing tick data: {e}")
    
    async def _collect_ohlcv_data(self, symbol: str, start: str, end: str, timeframe: str) -> pd.DataFrame:
        """Collect OHLCV data for a specific timeframe"""
        cache_key = f"{symbol}_{start}_{end}_{timeframe}"
        
        # Check cache
        if cache_key in self._ohlcv_cache:
            return self._ohlcv_cache[cache_key]
        
        all_data = []
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        
        current_date = start_dt
        
        while current_date < end_dt:
            # Determine chunk size based on timeframe
            if timeframe in ['1min', '5min', '15min']:
                chunk_days = 7  # 1 week for minute data
            elif timeframe in ['30min', '1hour']:
                chunk_days = 30  # 1 month for hourly data
            else:
                chunk_days = 365  # 1 year for daily data
            
            chunk_end = min(current_date + timedelta(days=chunk_days), end_dt)
            
            async with self.rate_limiter:
                try:
                    data = await self.openalgo_client.get_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=current_date.strftime("%Y-%m-%d"),
                        end_date=chunk_end.strftime("%Y-%m-%d")
                    )
                    
                    if data:
                        df = pd.DataFrame(data)
                        all_data.append(df)
                        
                except Exception as e:
                    self.logger.error(f"Error fetching {timeframe} data: {e}")
                
                await asyncio.sleep(self.request_delay)
            
            current_date = chunk_end
        
        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)
            result_df = self._process_ohlcv_data(result_df)
            
            # Cache the result
            self._ohlcv_cache[cache_key] = result_df
            
            return result_df
        
        return pd.DataFrame()
    
    def _process_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and enhance OHLCV data"""
        df = df.copy()
        
        # Ensure proper column names
        df.columns = df.columns.str.lower()
        
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Sort by time
        df.sort_index(inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Add derived features
        if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            # Price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Volatility features
            df['true_range'] = df[['high', 'low', 'close']].apply(
                lambda x: max(x['high'] - x['low'], 
                            abs(x['high'] - x['close']), 
                            abs(x['low'] - x['close'])), axis=1
            )
            
            # Volume features
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['dollar_volume'] = df['close'] * df['volume']
        
        return df
    
    async def _fetch_options_chain_snapshot(self, 
                                          symbol: str, 
                                          expiry: str, 
                                          date: str,
                                          strikes: Optional[List[int]]) -> Optional[OptionsChainData]:
        """Fetch options chain snapshot for a specific date"""
        try:
            # API call to get historical options chain
            response = await self.openalgo_client._make_request(
                "GET",
                f"/historical/options/{symbol}",
                params={
                    "expiry": expiry,
                    "date": date,
                    "strikes": strikes
                }
            )
            
            if response and 'data' in response:
                return OptionsChainData(
                    symbol=symbol,
                    expiry=expiry,
                    timestamp=datetime.strptime(date, "%Y-%m-%d"),
                    spot_price=response['data'].get('spot_price', 0),
                    strikes=response['data'].get('strikes', {})
                )
        except Exception as e:
            self.logger.error(f"Error fetching options chain: {e}")
            
        return None
    
    async def _calculate_option_greeks(self, chain_data: OptionsChainData) -> pd.DataFrame:
        """Calculate Greeks for options chain"""
        # This would implement Black-Scholes or other models for Greeks calculation
        # For now, return placeholder
        greeks_records = []
        
        for strike, options in chain_data.strikes.items():
            for option_type in ['CE', 'PE']:
                if option_type in options:
                    # Simplified Greeks calculation
                    greeks_records.append({
                        'strike': strike,
                        'option_type': option_type,
                        'delta': options[option_type].get('delta', 0),
                        'gamma': options[option_type].get('gamma', 0),
                        'theta': options[option_type].get('theta', 0),
                        'vega': options[option_type].get('vega', 0),
                        'rho': options[option_type].get('rho', 0)
                    })
        
        return pd.DataFrame(greeks_records)
    
    def _build_iv_surface(self, chain_data: OptionsChainData) -> pd.DataFrame:
        """Build implied volatility surface from options chain"""
        iv_records = []
        
        for strike, options in chain_data.strikes.items():
            for option_type in ['CE', 'PE']:
                if option_type in options and 'iv' in options[option_type]:
                    iv_records.append({
                        'strike': strike,
                        'option_type': option_type,
                        'iv': options[option_type]['iv'],
                        'moneyness': strike / chain_data.spot_price,
                        'time_to_expiry': self._calculate_time_to_expiry(
                            chain_data.timestamp, chain_data.expiry
                        )
                    })
        
        return pd.DataFrame(iv_records)
    
    def _calculate_time_to_expiry(self, current_date: datetime, expiry: str) -> float:
        """Calculate time to expiry in years"""
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
        days_to_expiry = (expiry_date - current_date).days
        return days_to_expiry / 365.0
    
    def _calculate_microstructure_metrics(self, 
                                        ticks: pd.DataFrame, 
                                        symbol: str,
                                        timestamp: datetime) -> MarketMicrostructure:
        """Calculate market microstructure metrics from tick data"""
        # Bid-ask spread
        avg_spread = (ticks['ask'] - ticks['bid']).mean()
        
        # Depth imbalance
        bid_volume = ticks['bid_size'].sum()
        ask_volume = ticks['ask_size'].sum()
        depth_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
        
        # Order book levels (simplified)
        order_book_levels = [
            {'bid_size': ticks['bid_size'].sum(), 'ask_size': ticks['ask_size'].sum()}
        ]
        
        # Trade intensity
        trade_count = len(ticks)
        time_window = (ticks.index[-1] - ticks.index[0]).total_seconds()
        trade_intensity = trade_count / time_window if time_window > 0 else 0
        
        # VWAP
        vwap = (ticks['price'] * ticks['volume']).sum() / ticks['volume'].sum()
        
        return MarketMicrostructure(
            timestamp=timestamp,
            symbol=symbol,
            bid_ask_spread=avg_spread,
            depth_imbalance=depth_imbalance,
            order_book_levels=order_book_levels,
            trade_intensity=trade_intensity,
            volume_weighted_price=vwap
        )
    
    async def _get_expiries_in_range(self, symbol: str, start_date: str, end_date: str) -> List[str]:
        """Get list of option expiries in date range"""
        try:
            response = await self.openalgo_client._make_request(
                "GET",
                f"/options/expiries/{symbol}",
                params={
                    "start_date": start_date,
                    "end_date": end_date
                }
            )
            return response.get('expiries', [])
        except Exception:
            # Fallback to calculating expiries
            expiries = []
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            current = start_dt
            while current <= end_dt:
                # Find last Thursday of the month
                last_day = current.replace(day=28) + timedelta(days=4)
                last_day = last_day - timedelta(days=last_day.day)
                
                while last_day.weekday() != 3:  # Thursday
                    last_day -= timedelta(days=1)
                
                if start_dt <= last_day <= end_dt:
                    expiries.append(last_day.strftime("%Y-%m-%d"))
                
                # Move to next month
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
            
            return expiries
    
    def get_data_quality_report(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Generate data quality report for collected data"""
        report = {
            'symbol': symbol,
            'date_range': f"{start_date} to {end_date}",
            'data_availability': {},
            'quality_metrics': {},
            'issues': []
        }
        
        # Check data availability for each timeframe
        for tf in self.timeframes:
            cache_key = f"{symbol}_{start_date}_{end_date}_{tf}"
            if cache_key in self._ohlcv_cache:
                df = self._ohlcv_cache[cache_key]
                report['data_availability'][tf] = {
                    'records': len(df),
                    'missing_data': df.isnull().sum().to_dict(),
                    'date_gaps': self._find_date_gaps(df)
                }
        
        return report
    
    def _find_date_gaps(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Find gaps in time series data"""
        gaps = []
        
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return gaps
        
        # Expected frequency based on data
        freq = pd.infer_freq(df.index)
        if not freq:
            return gaps
        
        # Create complete date range
        complete_range = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
        
        # Find missing dates
        missing_dates = complete_range.difference(df.index)
        
        if len(missing_dates) > 0:
            # Group consecutive missing dates
            # Implementation simplified for brevity
            gaps.append({
                'start': str(missing_dates[0]),
                'end': str(missing_dates[-1]),
                'count': len(missing_dates)
            })
        
        return gaps