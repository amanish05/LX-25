"""
Iron Condor Bot
Sells OTM Call Spread and Put Spread when expecting low volatility
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import numpy as np

from .base_bot import BaseBot
from ..utils.logger import TradingLogger


class IronCondorBot(BaseBot):
    """
    Iron Condor Strategy Bot with ML ensemble filtering
    - Sells OTM call spread and put spread simultaneously
    - Enters when IV Rank is high but expects low realized volatility
    - Uses ML ensemble to filter market conditions
    - Limited risk, limited reward strategy
    """
    
    def __init__(self, config: Dict[str, Any], db_manager, openalgo_client, logger=None):
        # Set bot type for ML configuration
        config["bot_type"] = "iron_condor"
        super().__init__(config, db_manager, openalgo_client, logger)
        
        # Strategy specific parameters
        self.iv_rank_threshold = self.entry_conditions.get("iv_rank_min", 70)
        self.dte_min = self.entry_conditions.get("dte_min", 30)
        self.dte_max = self.entry_conditions.get("dte_max", 45)
        self.wing_width = self.entry_conditions.get("wing_width", 100)  # Strike spread
        self.short_delta = self.entry_conditions.get("short_delta", 0.15)  # Delta for short strikes
        
        # Exit conditions
        self.profit_target = self.exit_conditions.get("profit_target_pct", 50) / 100
        self.stop_loss_multiplier = self.exit_conditions.get("stop_loss_multiplier", 2)
        self.time_exit_dte = self.exit_conditions.get("time_exit_dte", 21)
        
        # Position tracking
        self.condor_positions = {}  # Track all four legs together
        
        # Symbols to monitor (indices for options)
        self.symbols = {"NIFTY", "BANKNIFTY"}
        
    async def initialize(self):
        """Initialize the Iron Condor bot with ML ensemble integration"""
        self.logger.info(f"Initializing Iron Condor Bot: {self.name}")
        
        # Set symbols to monitor
        default_symbols = ["NIFTY", "BANKNIFTY"]
        config_symbols = self.config.get('symbols', default_symbols)
        self.symbols.update(config_symbols)
        
        # Initialize ML ensemble system
        await self._initialize_ml_ensemble()
        
        # Subscribe to market data
        for symbol in self.symbols:
            await self.openalgo_client.subscribe_market_data(symbol, self._on_market_data)
        
        self.is_initialized = True
        self.logger.info(f"Iron Condor Bot {self.name} initialized successfully")
        
    async def on_market_data(self, symbol: str, data: Dict[str, Any]):
        """Process market data and check for Iron Condor opportunities"""
        if not self.is_active or not self.is_initialized:
            return
        
        try:
            # Convert data to DataFrame for ML processing
            df_data = self._prepare_market_data(data)
            if df_data is None or len(df_data) < self.min_data_points:
                return
            
            # Generate ML ensemble signal
            ensemble_signal = self.indicators.generate_ensemble_signal(df_data)
            
            # Check for Iron Condor entry signals
            await self._check_entry_signals(symbol, data, ensemble_signal)
            
            # Check existing positions for exit signals
            await self._check_exit_signals(symbol, data, ensemble_signal)
            
        except Exception as e:
            self.logger.error(f"Error processing market data for {symbol}: {e}")
    
    async def _check_entry_signals(self, symbol: str, data: Dict[str, Any], ensemble_signal: Dict):
        """Check for Iron Condor entry conditions"""
        if symbol in self.condor_positions:
            return  # Already have position in this symbol
        
        current_price = data.get('last_price', 0)
        iv_rank = data.get('iv_rank', 0)
        
        # Basic entry conditions
        if iv_rank < self.iv_rank_threshold:
            return
        
        # Check ML ensemble signal (looking for low volatility/range-bound conditions)
        if not self._validate_ml_signal(ensemble_signal, "range_bound"):
            return
        
        # Get option chain
        option_chain = await self.openalgo_client.get_option_chain(symbol)
        if not option_chain:
            return
        
        # Find suitable strikes for Iron Condor
        strikes = self._find_condor_strikes(option_chain, current_price)
        if not strikes:
            return
        
        # Execute Iron Condor
        await self._execute_iron_condor(symbol, strikes, current_price)
    
    def _find_condor_strikes(self, option_chain: Dict, current_price: float) -> Optional[Dict]:
        """Find suitable strikes for Iron Condor"""
        try:
            # Find short put strike (below current price at target delta)
            short_put_strike = None
            short_call_strike = None
            
            for strike, options in option_chain.items():
                strike_price = float(strike)
                
                # Short put: below current price
                if strike_price < current_price:
                    put_delta = abs(options.get('put', {}).get('delta', 0))
                    if abs(put_delta - self.short_delta) < 0.05:
                        short_put_strike = strike_price
                
                # Short call: above current price  
                if strike_price > current_price:
                    call_delta = options.get('call', {}).get('delta', 0)
                    if abs(call_delta - self.short_delta) < 0.05:
                        short_call_strike = strike_price
            
            if not short_put_strike or not short_call_strike:
                return None
            
            return {
                'short_put': short_put_strike,
                'long_put': short_put_strike - self.wing_width,
                'short_call': short_call_strike,
                'long_call': short_call_strike + self.wing_width
            }
            
        except Exception as e:
            self.logger.error(f"Error finding condor strikes: {e}")
            return None
    
    async def _execute_iron_condor(self, symbol: str, strikes: Dict, current_price: float):
        """Execute Iron Condor strategy"""
        try:
            position_size = self._calculate_position_size(symbol, current_price)
            if position_size <= 0:
                return
            
            condor_legs = []
            
            # Leg 1: Sell Put (short put)
            short_put_order = await self._place_option_order(
                symbol=symbol,
                strike=strikes['short_put'],
                option_type='PUT',
                action='SELL',
                quantity=position_size
            )
            if short_put_order:
                condor_legs.append(short_put_order)
            
            # Leg 2: Buy Put (long put) 
            long_put_order = await self._place_option_order(
                symbol=symbol,
                strike=strikes['long_put'],
                option_type='PUT',
                action='BUY',
                quantity=position_size
            )
            if long_put_order:
                condor_legs.append(long_put_order)
            
            # Leg 3: Sell Call (short call)
            short_call_order = await self._place_option_order(
                symbol=symbol,
                strike=strikes['short_call'],
                option_type='CALL',
                action='SELL',
                quantity=position_size
            )
            if short_call_order:
                condor_legs.append(short_call_order)
            
            # Leg 4: Buy Call (long call)
            long_call_order = await self._place_option_order(
                symbol=symbol,
                strike=strikes['long_call'],
                option_type='CALL',
                action='BUY', 
                quantity=position_size
            )
            if long_call_order:
                condor_legs.append(long_call_order)
            
            # Store position if all legs executed
            if len(condor_legs) == 4:
                self.condor_positions[symbol] = {
                    'entry_time': datetime.now(),
                    'entry_price': current_price,
                    'strikes': strikes,
                    'legs': condor_legs,
                    'position_size': position_size,
                    'net_credit': sum(leg.get('fill_price', 0) * leg.get('action_multiplier', 1) for leg in condor_legs)
                }
                
                self.logger.info(f"Iron Condor executed for {symbol}: {strikes}")
            else:
                # Close partial fills
                for leg in condor_legs:
                    await self._close_position(leg['order_id'])
                    
        except Exception as e:
            self.logger.error(f"Error executing Iron Condor for {symbol}: {e}")
    
    async def _check_exit_signals(self, symbol: str, data: Dict[str, Any], ensemble_signal: Dict):
        """Check for Iron Condor exit conditions"""
        if symbol not in self.condor_positions:
            return
        
        position = self.condor_positions[symbol]
        current_price = data.get('last_price', 0)
        
        # Calculate current P&L
        current_value = await self._calculate_condor_value(symbol, position)
        if current_value is None:
            return
        
        net_credit = position['net_credit']
        unrealized_pnl = net_credit - current_value
        pnl_pct = unrealized_pnl / abs(net_credit) if net_credit != 0 else 0
        
        # Exit conditions
        should_exit = False
        exit_reason = ""
        
        # Profit target
        if pnl_pct >= self.profit_target:
            should_exit = True
            exit_reason = "profit_target"
        
        # Stop loss
        elif pnl_pct <= -self.stop_loss_multiplier:
            should_exit = True
            exit_reason = "stop_loss"
        
        # Time exit (close to expiration)
        elif self._get_days_to_expiry(position) <= self.time_exit_dte:
            should_exit = True
            exit_reason = "time_exit"
        
        # ML signal suggests high volatility (bad for Iron Condor)
        elif self._should_exit_on_ml_signal(ensemble_signal):
            should_exit = True
            exit_reason = "ml_signal"
        
        if should_exit:
            await self._close_iron_condor(symbol, exit_reason)
    
    async def _close_iron_condor(self, symbol: str, exit_reason: str):
        """Close Iron Condor position"""
        if symbol not in self.condor_positions:
            return
        
        try:
            position = self.condor_positions[symbol]
            
            # Close all legs
            for leg in position['legs']:
                # Reverse the original action
                action = 'BUY' if leg['action'] == 'SELL' else 'SELL'
                await self._place_option_order(
                    symbol=symbol,
                    strike=leg['strike'],
                    option_type=leg['option_type'],
                    action=action,
                    quantity=leg['quantity']
                )
            
            # Calculate final P&L
            final_pnl = await self._calculate_final_pnl(symbol, position)
            
            # Log the exit
            self.logger.info(f"Iron Condor closed for {symbol}: {exit_reason}, P&L: {final_pnl}")
            
            # Remove position
            del self.condor_positions[symbol]
            
        except Exception as e:
            self.logger.error(f"Error closing Iron Condor for {symbol}: {e}")
    
    def _validate_ml_signal(self, ensemble_signal: Dict, expected_regime: str) -> bool:
        """Validate ML ensemble signal for Iron Condor entry"""
        if not ensemble_signal:
            return False
        
        # Iron Condor works best in low volatility, range-bound markets
        regime = ensemble_signal.get('market_regime', '')
        volatility_score = ensemble_signal.get('volatility_score', 0.5)
        confidence = ensemble_signal.get('confidence', 0)
        
        return (regime == expected_regime and 
                volatility_score < 0.3 and  # Low volatility expected
                confidence > 0.6)
    
    def _should_exit_on_ml_signal(self, ensemble_signal: Dict) -> bool:
        """Check if ML signal suggests exiting Iron Condor"""
        if not ensemble_signal:
            return False
        
        # Exit if high volatility expected
        volatility_score = ensemble_signal.get('volatility_score', 0.5)
        regime = ensemble_signal.get('market_regime', '')
        
        return volatility_score > 0.7 or regime in ['trending_up', 'trending_down']
    
    async def _calculate_condor_value(self, symbol: str, position: Dict) -> Optional[float]:
        """Calculate current value of Iron Condor position"""
        try:
            # Get current option prices
            total_value = 0
            for leg in position['legs']:
                current_price = await self._get_option_price(
                    symbol, leg['strike'], leg['option_type']
                )
                if current_price is not None:
                    multiplier = -1 if leg['action'] == 'SELL' else 1
                    total_value += current_price * leg['quantity'] * multiplier
            
            return total_value
            
        except Exception as e:
            self.logger.error(f"Error calculating condor value: {e}")
            return None
    
    def get_strategy_description(self) -> str:
        """Return strategy description"""
        return (f"Iron Condor Bot - Sells OTM call and put spreads when IV > {self.iv_rank_threshold}%. "
                f"Profit target: {self.profit_target*100}%, Stop loss: {self.stop_loss_multiplier*100}%")