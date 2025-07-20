"""
Volatility Expander Bot
Buys ATM straddle/strangle when expecting volatility expansion
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import numpy as np

from .base_bot import BaseBot
from ..utils.logger import TradingLogger


class VolatilityExpanderBot(BaseBot):
    """
    Volatility Expander Strategy Bot with ML ensemble filtering
    - Buys ATM straddle/strangle when expecting volatility expansion
    - Enters when IV Rank is low but expecting volatility increase
    - Uses ML ensemble to identify volatility expansion opportunities
    - Benefits from large price movements in either direction
    """
    
    def __init__(self, config: Dict[str, Any], db_manager, openalgo_client, logger=None):
        # Set bot type for ML configuration
        config["bot_type"] = "volatility_expander"
        super().__init__(config, db_manager, openalgo_client, logger)
        
        # Strategy specific parameters
        self.iv_rank_threshold_max = self.entry_conditions.get("iv_rank_max", 40)  # Low IV
        self.iv_rank_threshold_min = self.entry_conditions.get("iv_rank_min", 10)  # Not too low
        self.dte_min = self.entry_conditions.get("dte_min", 20)
        self.dte_max = self.entry_conditions.get("dte_max", 40)
        self.volatility_expansion_threshold = self.entry_conditions.get("vol_expansion_threshold", 0.7)
        
        # Exit conditions
        self.profit_target = self.exit_conditions.get("profit_target_pct", 100) / 100  # 100% gain
        self.stop_loss_pct = self.exit_conditions.get("stop_loss_pct", 50) / 100  # 50% loss
        self.time_exit_dte = self.exit_conditions.get("time_exit_dte", 7)
        
        # Position tracking
        self.expander_positions = {}  # Track straddle/strangle legs together
        
        # Symbols to monitor (indices for options)
        self.symbols = {"NIFTY", "BANKNIFTY"}
        
    async def initialize(self):
        """Initialize the Volatility Expander bot with ML ensemble integration"""
        self.logger.info(f"Initializing Volatility Expander Bot: {self.name}")
        
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
        self.logger.info(f"Volatility Expander Bot {self.name} initialized successfully")
        
    async def on_market_data(self, symbol: str, data: Dict[str, Any]):
        """Process market data and check for volatility expansion opportunities"""
        if not self.is_active or not self.is_initialized:
            return
        
        try:
            # Convert data to DataFrame for ML processing
            df_data = self._prepare_market_data(data)
            if df_data is None or len(df_data) < self.min_data_points:
                return
            
            # Generate ML ensemble signal
            ensemble_signal = self.indicators.generate_ensemble_signal(df_data)
            
            # Check for volatility expansion entry signals
            await self._check_entry_signals(symbol, data, ensemble_signal)
            
            # Check existing positions for exit signals
            await self._check_exit_signals(symbol, data, ensemble_signal)
            
        except Exception as e:
            self.logger.error(f"Error processing market data for {symbol}: {e}")
    
    async def _check_entry_signals(self, symbol: str, data: Dict[str, Any], ensemble_signal: Dict):
        """Check for volatility expansion entry conditions"""
        if symbol in self.expander_positions:
            return  # Already have position in this symbol
        
        current_price = data.get('last_price', 0)
        iv_rank = data.get('iv_rank', 50)  # Mock IV rank
        
        # Basic entry conditions - look for low IV
        if not (self.iv_rank_threshold_min <= iv_rank <= self.iv_rank_threshold_max):
            return
        
        # Check ML ensemble signal for volatility expansion
        if not self._validate_ml_signal(ensemble_signal, "volatility_expansion"):
            return
        
        # Get option chain
        option_chain = await self.openalgo_client.get_option_chain(symbol)
        if not option_chain:
            return
        
        # Find suitable strikes for long straddle/strangle
        strikes = self._find_expansion_strikes(option_chain, current_price)
        if not strikes:
            return
        
        # Execute volatility expansion strategy
        await self._execute_volatility_expansion(symbol, strikes, current_price)
    
    def _find_expansion_strikes(self, option_chain: Dict, current_price: float) -> Optional[Dict]:
        """Find suitable strikes for volatility expansion strategy"""
        try:
            # For ATM straddle
            atm_strike = None
            
            # Find closest ATM strike
            min_diff = float('inf')
            for strike, options in option_chain.items():
                strike_price = float(strike)
                diff = abs(strike_price - current_price)
                if diff < min_diff:
                    min_diff = diff
                    atm_strike = strike_price
            
            if not atm_strike:
                return None
            
            # For strangle, we can use slightly OTM strikes
            call_strike = atm_strike + 100  # OTM call
            put_strike = atm_strike - 100   # OTM put
            
            return {
                'atm_call': atm_strike,
                'atm_put': atm_strike,
                'otm_call': call_strike,
                'otm_put': put_strike,
                'strategy': 'straddle'  # or 'strangle'
            }
            
        except Exception as e:
            self.logger.error(f"Error finding expansion strikes: {e}")
            return None
    
    async def _execute_volatility_expansion(self, symbol: str, strikes: Dict, current_price: float):
        """Execute volatility expansion strategy (buy straddle/strangle)"""
        try:
            position_size = self._calculate_position_size(symbol, current_price)
            if position_size <= 0:
                return
            
            expansion_legs = []
            strategy = strikes.get('strategy', 'straddle')
            
            if strategy == 'straddle':
                # Leg 1: Buy ATM Call
                call_order = await self._place_option_order(
                    symbol=symbol,
                    strike=strikes['atm_call'],
                    option_type='CALL',
                    action='BUY',
                    quantity=position_size
                )
                if call_order:
                    expansion_legs.append(call_order)
                
                # Leg 2: Buy ATM Put
                put_order = await self._place_option_order(
                    symbol=symbol,
                    strike=strikes['atm_put'],
                    option_type='PUT',
                    action='BUY',
                    quantity=position_size
                )
                if put_order:
                    expansion_legs.append(put_order)
            
            else:  # strangle
                # Leg 1: Buy OTM Call
                call_order = await self._place_option_order(
                    symbol=symbol,
                    strike=strikes['otm_call'],
                    option_type='CALL',
                    action='BUY',
                    quantity=position_size
                )
                if call_order:
                    expansion_legs.append(call_order)
                
                # Leg 2: Buy OTM Put
                put_order = await self._place_option_order(
                    symbol=symbol,
                    strike=strikes['otm_put'],
                    option_type='PUT',
                    action='BUY',
                    quantity=position_size
                )
                if put_order:
                    expansion_legs.append(put_order)
            
            # Store position if legs executed
            if len(expansion_legs) >= 2:
                net_debit = sum(leg.get('fill_price', 0) for leg in expansion_legs)
                
                self.expander_positions[symbol] = {
                    'entry_time': datetime.now(),
                    'entry_price': current_price,
                    'strikes': strikes,
                    'legs': expansion_legs,
                    'position_size': position_size,
                    'net_debit': net_debit,
                    'strategy': strategy
                }
                
                self.logger.info(f"Volatility expansion executed for {symbol}: {strategy} @ {strikes}")
            else:
                # Close partial fills
                for leg in expansion_legs:
                    await self._close_position(leg['order_id'])
                    
        except Exception as e:
            self.logger.error(f"Error executing volatility expansion for {symbol}: {e}")
    
    async def _check_exit_signals(self, symbol: str, data: Dict[str, Any], ensemble_signal: Dict):
        """Check for volatility expansion exit conditions"""
        if symbol not in self.expander_positions:
            return
        
        position = self.expander_positions[symbol]
        current_price = data.get('last_price', 0)
        
        # Calculate current P&L
        current_value = await self._calculate_expansion_value(symbol, position)
        if current_value is None:
            return
        
        net_debit = position['net_debit']
        unrealized_pnl = current_value - net_debit
        pnl_pct = unrealized_pnl / net_debit if net_debit > 0 else 0
        
        # Exit conditions
        should_exit = False
        exit_reason = ""
        
        # Profit target
        if pnl_pct >= self.profit_target:
            should_exit = True
            exit_reason = "profit_target"
        
        # Stop loss
        elif pnl_pct <= -self.stop_loss_pct:
            should_exit = True
            exit_reason = "stop_loss"
        
        # Time exit (close to expiration)
        elif self._get_days_to_expiry(position) <= self.time_exit_dte:
            should_exit = True
            exit_reason = "time_exit"
        
        # ML signal suggests volatility contraction
        elif self._should_exit_on_ml_signal(ensemble_signal):
            should_exit = True
            exit_reason = "ml_signal"
        
        if should_exit:
            await self._close_volatility_expansion(symbol, exit_reason)
    
    async def _close_volatility_expansion(self, symbol: str, exit_reason: str):
        """Close volatility expansion position"""
        if symbol not in self.expander_positions:
            return
        
        try:
            position = self.expander_positions[symbol]
            
            # Close all legs
            for leg in position['legs']:
                # Sell the options we bought
                await self._place_option_order(
                    symbol=symbol,
                    strike=leg['strike'],
                    option_type=leg['option_type'],
                    action='SELL',
                    quantity=leg['quantity']
                )
            
            # Calculate final P&L
            final_pnl = await self._calculate_final_pnl(symbol, position)
            
            # Log the exit
            self.logger.info(f"Volatility expansion closed for {symbol}: {exit_reason}, P&L: {final_pnl}")
            
            # Remove position
            del self.expander_positions[symbol]
            
        except Exception as e:
            self.logger.error(f"Error closing volatility expansion for {symbol}: {e}")
    
    def _validate_ml_signal(self, ensemble_signal: Dict, expected_signal: str) -> bool:
        """Validate ML ensemble signal for volatility expansion entry"""
        if not ensemble_signal:
            return False
        
        # Volatility expansion works best when expecting high volatility
        volatility_score = ensemble_signal.get('volatility_score', 0.5)
        confidence = ensemble_signal.get('confidence', 0)
        signal_type = ensemble_signal.get('signal_type', 'hold')
        
        # Look for signals suggesting upcoming volatility
        if expected_signal == "volatility_expansion":
            return (volatility_score > self.volatility_expansion_threshold and 
                   confidence > 0.6 and
                   signal_type in ['volatility_breakout', 'trend_change', 'high_volatility'])
        
        return False
    
    def _should_exit_on_ml_signal(self, ensemble_signal: Dict) -> bool:
        """Check if ML signal suggests exiting volatility expansion"""
        if not ensemble_signal:
            return False
        
        # Exit if low volatility expected (bad for long volatility positions)
        volatility_score = ensemble_signal.get('volatility_score', 0.5)
        regime = ensemble_signal.get('market_regime', '')
        
        return volatility_score < 0.3 or regime == 'low_volatility'
    
    async def _calculate_expansion_value(self, symbol: str, position: Dict) -> Optional[float]:
        """Calculate current value of volatility expansion position"""
        try:
            # Get current option prices
            total_value = 0
            for leg in position['legs']:
                current_price = await self._get_option_price(
                    symbol, leg['strike'], leg['option_type']
                )
                if current_price is not None:
                    # Long positions - we own the options
                    total_value += current_price * leg['quantity']
            
            return total_value
            
        except Exception as e:
            self.logger.error(f"Error calculating expansion value: {e}")
            return None
    
    async def _place_option_order(self, symbol: str, strike: float, option_type: str, 
                                 action: str, quantity: int) -> Optional[Dict]:
        """Place option order"""
        try:
            # Generate option symbol (simplified)
            option_symbol = f"{symbol}{datetime.now().strftime('%y%m%d')}{int(strike)}{option_type[0]}E"
            
            order = await self.openalgo_client.place_order(
                symbol=option_symbol,
                exchange="NFO",
                action=action,
                quantity=quantity,
                product="NRML",
                price_type="MARKET"
            )
            
            if order and order.get('status') == 'success':
                return {
                    'order_id': order.get('orderid'),
                    'symbol': option_symbol,
                    'strike': strike,
                    'option_type': option_type,
                    'action': action,
                    'quantity': quantity,
                    'fill_price': order.get('price', 0)
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error placing option order: {e}")
            return None
    
    async def _get_option_price(self, symbol: str, strike: float, option_type: str) -> Optional[float]:
        """Get current option price"""
        try:
            # Generate option symbol
            option_symbol = f"{symbol}{datetime.now().strftime('%y%m%d')}{int(strike)}{option_type[0]}E"
            quote = await self.openalgo_client.get_quote(option_symbol, "NFO")
            return quote.get('ltp')
        except:
            return None
    
    async def _calculate_final_pnl(self, symbol: str, position: Dict) -> float:
        """Calculate final P&L for closed position"""
        try:
            current_value = await self._calculate_expansion_value(symbol, position)
            if current_value is not None:
                return current_value - position['net_debit']
            return 0
        except:
            return 0
    
    def _get_days_to_expiry(self, position: Dict) -> int:
        """Calculate days to expiry"""
        # Simplified - would use actual expiry from position
        entry_time = position.get('entry_time', datetime.now())
        # Assume weekly expiry (next Thursday)
        days = 7 - entry_time.weekday() if entry_time.weekday() < 3 else 14 - entry_time.weekday()
        return max(days, 0)
    
    def _calculate_position_size(self, symbol: str, current_price: float) -> int:
        """Calculate position size for volatility expansion"""
        try:
            # Conservative position sizing for long options
            max_risk = self.available_capital * 0.1  # Max 10% risk per trade
            
            # Estimate premium cost (simplified)
            estimated_premium_per_lot = current_price * 0.02  # 2% of underlying
            lot_size = 50 if symbol == "NIFTY" else 25
            
            # Calculate maximum lots we can afford
            max_lots = int(max_risk / (estimated_premium_per_lot * lot_size * 2))  # 2 legs
            
            # Apply configured limits
            configured_lots = self.config.get("lots_per_trade", 1)
            final_lots = min(max_lots, configured_lots)
            
            return max(final_lots * lot_size, lot_size)  # At least one lot
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
    
    def get_strategy_description(self) -> str:
        """Return strategy description"""
        return (f"Volatility Expander Bot - Buys ATM straddle/strangle when IV < {self.iv_rank_threshold_max}% "
                f"and ML signals volatility expansion. Profit target: {self.profit_target*100}%, "
                f"Stop loss: {self.stop_loss_pct*100}%")