"""
Short Straddle Bot
Sells ATM Call and Put options when IV is high
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import numpy as np

from .base_bot import BaseBot
from ..utils.logger import TradingLogger


class ShortStraddleBot(BaseBot):
    """
    Short Straddle Strategy Bot
    - Sells ATM Call and Put simultaneously
    - Enters when IV Rank > threshold
    - Manages risk through stop loss and profit targets
    """
    
    def __init__(self, config: Dict[str, Any], db_manager, openalgo_client, logger=None):
        super().__init__(config, db_manager, openalgo_client, logger)
        
        # Strategy specific parameters
        self.iv_rank_threshold = self.entry_conditions.get("iv_rank_min", 75)
        self.dte_min = self.entry_conditions.get("dte_min", 30)
        self.dte_max = self.entry_conditions.get("dte_max", 45)
        self.profit_target = self.exit_conditions.get("profit_target_pct", 50) / 100
        self.stop_loss_multiplier = self.exit_conditions.get("stop_loss_multiplier", 2)
        self.time_exit_dte = self.exit_conditions.get("time_exit_dte", 21)
        
        # Position tracking
        self.straddle_positions = {}  # Track call and put legs together
        
        # Symbols to monitor (indices for options)
        self.symbols = {"NIFTY", "BANKNIFTY"}
        
    async def initialize(self):
        """Initialize the Short Straddle bot"""
        self.logger.info(f"Initializing Short Straddle Bot: {self.name}")
        
        # Load historical IV data for rank calculation
        self.iv_history = {}
        
        # Initialize option chain data
        self.option_chains = {}
        
        # Schedule option chain updates
        self._tasks.append(asyncio.create_task(self._update_option_chains()))
        
        self.logger.info("Short Straddle Bot initialized")
    
    async def generate_signals(self, symbol: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate short straddle signals based on IV and market conditions"""
        try:
            # Only generate signals for index options
            if symbol not in ["NIFTY", "BANKNIFTY"]:
                return None
            
            # Check if we already have a position
            if symbol in self.straddle_positions:
                return None
            
            # Get current option chain
            option_chain = self.option_chains.get(symbol)
            if not option_chain:
                return None
            
            # Calculate IV rank
            iv_rank = await self._calculate_iv_rank(symbol, option_chain)
            
            # Check entry conditions
            if iv_rank < self.iv_rank_threshold:
                return None
            
            # Find suitable expiry
            expiry = self._find_suitable_expiry(option_chain)
            if not expiry:
                return None
            
            # Get ATM strike
            spot_price = data.get("ltp", 0)
            atm_strike = self._get_atm_strike(spot_price, option_chain, expiry)
            
            # Get option prices
            call_data = self._get_option_data(option_chain, expiry, atm_strike, "CE")
            put_data = self._get_option_data(option_chain, expiry, atm_strike, "PE")
            
            if not call_data or not put_data:
                return None
            
            # Check liquidity
            if not self._check_liquidity(call_data, put_data):
                return None
            
            # Generate signal
            total_premium = call_data["ltp"] + put_data["ltp"]
            
            signal = {
                "symbol": symbol,
                "type": "SHORT_STRADDLE",
                "expiry": expiry,
                "strike": atm_strike,
                "call_symbol": call_data["symbol"],
                "put_symbol": put_data["symbol"],
                "call_price": call_data["ltp"],
                "put_price": put_data["ltp"],
                "total_premium": total_premium,
                "iv_rank": iv_rank,
                "strength": min(iv_rank / 100, 1.0),
                "metadata": {
                    "spot_price": spot_price,
                    "call_iv": call_data.get("iv", 0),
                    "put_iv": put_data.get("iv", 0),
                    "call_oi": call_data.get("oi", 0),
                    "put_oi": put_data.get("oi", 0),
                    "profit_target": total_premium * self.profit_target,
                    "stop_loss": total_premium * self.stop_loss_multiplier
                }
            }
            
            self.logger.info(f"Generated short straddle signal for {symbol}: "
                           f"Strike={atm_strike}, Premium={total_premium:.2f}, "
                           f"IV Rank={iv_rank:.1f}%")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return None
    
    async def calculate_position_size(self, signal: Dict[str, Any]) -> int:
        """Calculate position size based on risk and available capital"""
        # For straddle, we need margin for both legs
        margin_required = self._calculate_margin_requirement(signal)
        
        # Check available capital
        if margin_required > self.available_capital:
            self.logger.warning(f"Insufficient capital for straddle. "
                              f"Required: {margin_required}, Available: {self.available_capital}")
            return 0
        
        # Calculate lots based on risk parameters
        max_risk = self.available_capital * 0.2  # Max 20% risk per trade
        stop_loss_amount = signal["total_premium"] * self.stop_loss_multiplier
        
        # Calculate lot size (assuming lot size of 50 for NIFTY, 25 for BANKNIFTY)
        lot_size = 50 if signal["symbol"] == "NIFTY" else 25
        max_lots = int(max_risk / (stop_loss_amount * lot_size))
        
        # Apply position size limits
        configured_lots = self.config.get("lots_per_trade", 1)
        final_lots = min(max_lots, configured_lots)
        
        return final_lots * lot_size
    
    async def should_enter_position(self, signal: Dict[str, Any]) -> bool:
        """Check if all entry conditions are met"""
        # Check market conditions
        if not await self._check_market_conditions(signal["symbol"]):
            return False
        
        # Check for upcoming events
        if await self._has_major_event_soon(signal["symbol"]):
            self.logger.warning("Major event detected, skipping entry")
            return False
        
        # Check realized volatility
        realized_vol = await self._calculate_realized_volatility(signal["symbol"])
        if realized_vol > signal["metadata"]["call_iv"] * 0.8:
            self.logger.warning("Realized volatility too high relative to IV")
            return False
        
        return True
    
    async def should_exit_position(self, position: Dict[str, Any], 
                                  current_data: Dict[str, Any]) -> bool:
        """Check if exit conditions are met for the straddle"""
        straddle = self.straddle_positions.get(position["symbol"])
        if not straddle:
            return False
        
        # Get current option prices
        call_price = await self._get_current_option_price(straddle["call_symbol"])
        put_price = await self._get_current_option_price(straddle["put_symbol"])
        
        if call_price is None or put_price is None:
            return False
        
        current_premium = call_price + put_price
        entry_premium = straddle["entry_premium"]
        
        # Calculate P&L
        pnl_pct = (entry_premium - current_premium) / entry_premium
        
        # Check profit target
        if pnl_pct >= self.profit_target:
            self.logger.info(f"Profit target reached: {pnl_pct:.1%}")
            return True
        
        # Check stop loss
        if pnl_pct <= -self.stop_loss_multiplier:
            self.logger.warning(f"Stop loss hit: {pnl_pct:.1%}")
            return True
        
        # Check time-based exit
        days_to_expiry = await self._get_days_to_expiry(straddle["expiry"])
        if days_to_expiry <= self.time_exit_dte:
            self.logger.info(f"Time-based exit: {days_to_expiry} DTE")
            return True
        
        # Check for IV crush
        current_iv_rank = await self._calculate_iv_rank(position["symbol"], None)
        entry_iv_rank = straddle.get("entry_iv_rank", 0)
        
        if current_iv_rank < entry_iv_rank - 20:
            self.logger.info(f"IV crush detected: {current_iv_rank:.1f}% vs {entry_iv_rank:.1f}%")
            return True
        
        return False
    
    async def _place_order(self, signal: Dict[str, Any], quantity: int) -> bool:
        """Override to place both legs of the straddle"""
        try:
            # Place SELL orders for both Call and Put
            call_order = await self.openalgo_client.place_order(
                symbol=signal["call_symbol"],
                exchange="NFO",
                action="SELL",
                quantity=quantity,
                product="NRML",  # Positional for options
                price_type="LIMIT",
                price=signal["call_price"]
            )
            
            if call_order.get("status") != "success":
                self.logger.error(f"Failed to place call order: {call_order}")
                return False
            
            put_order = await self.openalgo_client.place_order(
                symbol=signal["put_symbol"],
                exchange="NFO",
                action="SELL",
                quantity=quantity,
                product="NRML",
                price_type="LIMIT",
                price=signal["put_price"]
            )
            
            if put_order.get("status") != "success":
                # Cancel call order if put fails
                await self.openalgo_client.cancel_order(call_order["orderid"])
                self.logger.error(f"Failed to place put order: {put_order}")
                return False
            
            # Track straddle position
            self.straddle_positions[signal["symbol"]] = {
                "call_symbol": signal["call_symbol"],
                "put_symbol": signal["put_symbol"],
                "call_order_id": call_order["orderid"],
                "put_order_id": put_order["orderid"],
                "strike": signal["strike"],
                "expiry": signal["expiry"],
                "quantity": quantity,
                "entry_premium": signal["total_premium"],
                "entry_iv_rank": signal["iv_rank"],
                "entry_time": datetime.now()
            }
            
            # Create position records
            await self.db_manager.create_position(
                self.name, signal["symbol"], "NFO", "SHORT_STRADDLE",
                quantity, signal["total_premium"],
                {"straddle": self.straddle_positions[signal["symbol"]], "signal": signal}
            )
            
            self.logger.info(f"Short straddle executed: {signal['symbol']} "
                           f"{signal['strike']} @ {signal['total_premium']:.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error placing straddle order: {e}")
            return False
    
    async def _close_position(self, position: Dict[str, Any]):
        """Override to close both legs of the straddle"""
        straddle = self.straddle_positions.get(position["symbol"])
        if not straddle:
            return
        
        try:
            # Place BUY orders to close both legs
            call_close = await self.openalgo_client.place_order(
                symbol=straddle["call_symbol"],
                exchange="NFO",
                action="BUY",
                quantity=straddle["quantity"],
                product="NRML",
                price_type="MARKET"
            )
            
            put_close = await self.openalgo_client.place_order(
                symbol=straddle["put_symbol"],
                exchange="NFO",
                action="BUY",
                quantity=straddle["quantity"],
                product="NRML",
                price_type="MARKET"
            )
            
            # Remove from tracking
            del self.straddle_positions[position["symbol"]]
            
            # Update position in database
            await self.db_manager.update_position(
                position["id"],
                status="CLOSED",
                exit_time=datetime.now()
            )
            
            self.logger.info(f"Closed short straddle position: {position['symbol']}")
            
        except Exception as e:
            self.logger.error(f"Error closing straddle: {e}")
    
    async def _update_option_chains(self):
        """Periodically update option chain data"""
        while self.state.value in ["running", "paused"]:
            try:
                for symbol in self.symbols:
                    # Get current expiries
                    expiries = await self._get_expiries(symbol)
                    
                    # Update option chain for nearest expiries
                    for expiry in expiries[:3]:  # First 3 expiries
                        chain = await self.openalgo_client.get_option_chain(symbol, expiry)
                        
                        if symbol not in self.option_chains:
                            self.option_chains[symbol] = {}
                        
                        self.option_chains[symbol][expiry] = chain
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error updating option chains: {e}")
                await asyncio.sleep(60)
    
    async def _calculate_iv_rank(self, symbol: str, option_chain: Dict[str, Any]) -> float:
        """Calculate IV Rank for the symbol"""
        # This would need historical IV data
        # For now, return a mock value
        current_iv = 20  # Would calculate from option chain
        
        # Mock IV rank calculation
        iv_rank = np.random.uniform(60, 90)  # Replace with actual calculation
        
        return iv_rank
    
    async def _get_expiries(self, symbol: str) -> List[str]:
        """Get available expiries for the symbol"""
        # This would fetch from OpenAlgo
        # Mock implementation
        today = datetime.now()
        expiries = []
        
        # Generate weekly expiries
        for i in range(8):
            expiry = today + timedelta(days=(3 - today.weekday() + 7 * i) % 7 + 7 * i)
            expiries.append(expiry.strftime("%Y-%m-%d"))
        
        return expiries
    
    def _find_suitable_expiry(self, option_chain: Dict[str, Any]) -> Optional[str]:
        """Find expiry within DTE range"""
        if not option_chain:
            return None
        
        today = datetime.now()
        
        for expiry_str in sorted(option_chain.keys()):
            expiry = datetime.strptime(expiry_str, "%Y-%m-%d")
            dte = (expiry - today).days
            
            if self.dte_min <= dte <= self.dte_max:
                return expiry_str
        
        return None
    
    def _get_atm_strike(self, spot_price: float, option_chain: Dict[str, Any], 
                       expiry: str) -> float:
        """Get ATM strike price"""
        # Round to nearest strike
        if "NIFTY" in str(option_chain):
            strike_gap = 50
        else:  # BANKNIFTY
            strike_gap = 100
        
        atm_strike = round(spot_price / strike_gap) * strike_gap
        return atm_strike
    
    def _get_option_data(self, option_chain: Dict[str, Any], expiry: str, 
                        strike: float, option_type: str) -> Optional[Dict[str, Any]]:
        """Get option data for specific strike and type"""
        # Mock implementation - would fetch from actual option chain
        return {
            "symbol": f"NIFTY{expiry}{int(strike)}{option_type}",
            "ltp": np.random.uniform(100, 200),
            "iv": np.random.uniform(15, 25),
            "oi": np.random.randint(10000, 100000),
            "volume": np.random.randint(1000, 10000)
        }
    
    def _check_liquidity(self, call_data: Dict[str, Any], 
                        put_data: Dict[str, Any]) -> bool:
        """Check if options have sufficient liquidity"""
        min_oi = 1000
        min_volume = 100
        
        return (call_data.get("oi", 0) > min_oi and 
                put_data.get("oi", 0) > min_oi and
                call_data.get("volume", 0) > min_volume and
                put_data.get("volume", 0) > min_volume)
    
    def _calculate_margin_requirement(self, signal: Dict[str, Any]) -> float:
        """Calculate margin required for short straddle"""
        # Simplified margin calculation
        # Actual calculation would be based on exchange rules
        spot_price = signal["metadata"]["spot_price"]
        
        # Approximate margin: 15% of notional for each leg
        margin_per_leg = spot_price * 0.15
        total_margin = margin_per_leg * 2  # Both legs
        
        return total_margin
    
    async def _check_market_conditions(self, symbol: str) -> bool:
        """Check if market conditions are suitable"""
        # Would check for trending vs ranging market
        # Mock implementation
        return True
    
    async def _has_major_event_soon(self, symbol: str) -> bool:
        """Check for upcoming major events"""
        # Would check economic calendar
        # Mock implementation
        return False
    
    async def _calculate_realized_volatility(self, symbol: str) -> float:
        """Calculate realized volatility"""
        # Would calculate from historical data
        # Mock implementation
        return np.random.uniform(10, 20)
    
    async def _get_current_option_price(self, option_symbol: str) -> Optional[float]:
        """Get current price of an option"""
        try:
            quote = await self.openalgo_client.get_quote(option_symbol, "NFO")
            return quote.get("ltp")
        except:
            return None
    
    async def _get_days_to_expiry(self, expiry: str) -> int:
        """Calculate days to expiry"""
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
        return (expiry_date - datetime.now()).days