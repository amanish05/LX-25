"""
Historical Data Loader for Trading Bot System
Loads historical market data from OpenAlgo for training and backtesting
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from src.integrations.openalgo_client import OpenAlgoClient


@dataclass
class HistoricalData:
    """Container for historical market data"""
    symbol: str
    timeframe: str
    data: pd.DataFrame
    start_date: str
    end_date: str
    total_records: int


class HistoricalDataLoader:
    """Loads and processes historical market data"""
    
    def __init__(self, openalgo_client: Optional[OpenAlgoClient] = None):
        """Initialize historical data loader
        
        Args:
            openalgo_client: OpenAlgo API client instance
        """
        self.openalgo_client = openalgo_client or OpenAlgoClient()
        self.logger = logging.getLogger(__name__)
        self._cache: Dict[str, HistoricalData] = {}
        
    async def load_training_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        timeframe: str = "5min"
    ) -> HistoricalData:
        """Load historical data for model training
        
        Args:
            symbol: Trading symbol (e.g., "NIFTY", "BANKNIFTY")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Data timeframe (1min, 5min, 15min, 1hour, 1day)
            
        Returns:
            HistoricalData object containing the loaded data
        """
        cache_key = f"{symbol}_{start_date}_{end_date}_{timeframe}"
        
        # Check cache first
        if cache_key in self._cache:
            self.logger.info(f"Using cached data for {cache_key}")
            return self._cache[cache_key]
        
        self.logger.info(f"Loading training data for {symbol} from {start_date} to {end_date}")
        
        # Convert dates to datetime objects
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Load data in chunks (OpenAlgo may have limits)
        all_data = []
        current_date = start_dt
        
        while current_date < end_dt:
            # Load 3 months at a time
            chunk_end = min(current_date + timedelta(days=90), end_dt)
            
            try:
                # Get historical data from OpenAlgo
                data = await self.openalgo_client.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=current_date.strftime("%Y-%m-%d"),
                    end_date=chunk_end.strftime("%Y-%m-%d")
                )
                
                if data:
                    all_data.append(pd.DataFrame(data))
                    self.logger.info(f"Loaded data chunk from {current_date} to {chunk_end}")
                
            except Exception as e:
                self.logger.error(f"Error loading data chunk: {e}")
                
            current_date = chunk_end + timedelta(days=1)
            await asyncio.sleep(0.5)  # Rate limiting
        
        # Combine all chunks
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df = self._process_data(df)
            
            historical_data = HistoricalData(
                symbol=symbol,
                timeframe=timeframe,
                data=df,
                start_date=start_date,
                end_date=end_date,
                total_records=len(df)
            )
            
            # Cache the data
            self._cache[cache_key] = historical_data
            
            self.logger.info(f"Successfully loaded {len(df)} records for {symbol}")
            return historical_data
        else:
            raise ValueError(f"No data found for {symbol} between {start_date} and {end_date}")
    
    async def load_recent_data(
        self, 
        symbol: str, 
        days: int = 30,
        timeframe: str = "5min"
    ) -> HistoricalData:
        """Load recent data for live testing
        
        Args:
            symbol: Trading symbol
            days: Number of days to load (default: 30)
            timeframe: Data timeframe
            
        Returns:
            HistoricalData object containing recent data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return await self.load_training_data(
            symbol=symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            timeframe=timeframe
        )
    
    async def load_options_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        strikes: Optional[List[int]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load historical options data
        
        Args:
            symbol: Underlying symbol
            start_date: Start date
            end_date: End date
            strikes: List of strike prices (None for all)
            
        Returns:
            Dictionary of options data by expiry
        """
        self.logger.info(f"Loading options data for {symbol}")
        
        options_data = {}
        
        # Get list of expiries in the date range
        expiries = await self._get_expiries_in_range(symbol, start_date, end_date)
        
        for expiry in expiries:
            try:
                # Load options chain for each expiry
                chain_data = await self.openalgo_client.get_historical_option_chain(
                    symbol=symbol,
                    expiry=expiry,
                    date=expiry  # Get data on expiry date
                )
                
                if chain_data:
                    # Process and filter strikes
                    df = self._process_options_data(chain_data, strikes)
                    options_data[expiry] = df
                    
            except Exception as e:
                self.logger.error(f"Error loading options data for {expiry}: {e}")
                
            await asyncio.sleep(0.5)  # Rate limiting
        
        return options_data
    
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean historical data
        
        Args:
            df: Raw dataframe
            
        Returns:
            Processed dataframe
        """
        # Ensure proper column names
        df.columns = df.columns.str.lower()
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        # Sort by time
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Fill missing values
        df = df.fillna(method='ffill')
        
        # Add derived features
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Volume features
            if 'volume' in df.columns:
                df['volume_ma'] = df['volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df
    
    def _process_options_data(
        self, 
        chain_data: Dict[str, Any], 
        strikes: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Process options chain data
        
        Args:
            chain_data: Raw options chain data
            strikes: Filter for specific strikes
            
        Returns:
            Processed options dataframe
        """
        records = []
        
        for strike, data in chain_data.get('strikes', {}).items():
            if strikes and int(strike) not in strikes:
                continue
                
            # Process call option
            if 'CE' in data:
                ce_data = data['CE']
                ce_data['strike'] = int(strike)
                ce_data['option_type'] = 'CE'
                records.append(ce_data)
            
            # Process put option
            if 'PE' in data:
                pe_data = data['PE']
                pe_data['strike'] = int(strike)
                pe_data['option_type'] = 'PE'
                records.append(pe_data)
        
        df = pd.DataFrame(records)
        
        # Calculate additional metrics
        if not df.empty:
            df['moneyness'] = df.apply(
                lambda x: 'ITM' if (
                    (x['option_type'] == 'CE' and x['strike'] < chain_data.get('spot_price', 0)) or
                    (x['option_type'] == 'PE' and x['strike'] > chain_data.get('spot_price', 0))
                ) else 'OTM',
                axis=1
            )
        
        return df
    
    async def _get_expiries_in_range(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str
    ) -> List[str]:
        """Get list of option expiries in date range
        
        Args:
            symbol: Underlying symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            List of expiry dates
        """
        # This would typically call OpenAlgo API to get expiry dates
        # For now, return monthly expiries
        expiries = []
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        current = start_dt
        while current <= end_dt:
            # Find last Thursday of the month (typical expiry)
            last_day = current.replace(day=28) + timedelta(days=4)
            last_day = last_day - timedelta(days=last_day.day)
            
            # Find last Thursday
            while last_day.weekday() != 3:  # Thursday is 3
                last_day -= timedelta(days=1)
            
            if start_dt <= last_day <= end_dt:
                expiries.append(last_day.strftime("%Y-%m-%d"))
            
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        return expiries
    
    def get_feature_matrix(
        self, 
        historical_data: HistoricalData,
        feature_config: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """Extract feature matrix for ML training
        
        Args:
            historical_data: Historical data object
            feature_config: Configuration for features to extract
            
        Returns:
            Feature matrix dataframe
        """
        df = historical_data.data.copy()
        
        if feature_config is None:
            # Default feature configuration
            feature_config = {
                "price": ["returns", "log_returns", "high_low_ratio"],
                "volume": ["volume_ratio"],
                "technical": []  # Will be calculated
            }
        
        features = pd.DataFrame(index=df.index)
        
        # Extract configured features
        for category, feature_list in feature_config.items():
            for feature in feature_list:
                if feature in df.columns:
                    features[feature] = df[feature]
        
        # Add technical indicators if requested
        if "technical" in feature_config:
            # These would be calculated using the indicators module
            pass
        
        return features
    
    def create_training_dataset(
        self,
        historical_data: HistoricalData,
        lookback_window: int = 20,
        prediction_horizon: int = 5,
        target_type: str = "returns"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create supervised learning dataset
        
        Args:
            historical_data: Historical data
            lookback_window: Number of periods to look back
            prediction_horizon: Number of periods to predict
            target_type: Type of target variable
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        df = historical_data.data
        
        # Get features
        features = self.get_feature_matrix(historical_data)
        
        # Create sequences
        X, y = [], []
        
        for i in range(lookback_window, len(features) - prediction_horizon):
            # Input sequence
            X.append(features.iloc[i-lookback_window:i].values)
            
            # Target value
            if target_type == "returns":
                target = df['returns'].iloc[i:i+prediction_horizon].sum()
            elif target_type == "direction":
                target = 1 if df['returns'].iloc[i:i+prediction_horizon].sum() > 0 else 0
            else:
                target = df[target_type].iloc[i+prediction_horizon-1]
            
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def clear_cache(self):
        """Clear cached data"""
        self._cache.clear()
        self.logger.info("Cleared historical data cache")