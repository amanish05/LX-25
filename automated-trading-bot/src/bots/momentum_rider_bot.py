"""
Momentum Rider Bot - Enhanced with Advanced Confirmation System
Option-buying bot that trades momentum with multi-layer validation
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

from .base_bot import BaseBot, BotState
from ..indicators.momentum import MomentumIndicators
from ..indicators.advanced_confirmation import AdvancedConfirmationSystem, ConfirmationSignal
from ..indicators.signal_validator import SignalValidator
from ..config import BOT_CONSTANTS


class MomentumRiderBot(BaseBot):
    """
    Enhanced momentum-based option buying bot
    
    Features:
    - Multi-layer confirmation system
    - False positive filtering
    - Adaptive parameter adjustment
    - Real-time performance tracking
    """
    
    def __init__(self, config: Dict[str, Any], db_manager, openalgo_client):
        super().__init__(config, db_manager, openalgo_client)
        
        # Initialize indicators
        self.momentum_indicator = MomentumIndicators()
        self.confirmation_system = AdvancedConfirmationSystem(config.get('confirmation_config', {}))
        self.signal_validator = SignalValidator(config.get('validator_config', {}))
        
        # Enhanced parameters
        self.momentum_threshold = config.get('momentum_threshold', 0.45)
        self.volume_spike_multiplier = config.get('volume_spike_multiplier', 2.0)
        self.min_confluence_score = config.get('min_confluence_score', 0.65)
        self.max_false_positive_rate = config.get('max_false_positive_rate', 0.30)
        
        # Position management
        self.max_hold_minutes = config.get('max_hold_minutes', 30)
        self.trailing_stop_activate = config.get('trailing_stop_activate', 50)  # 50% profit
        self.trailing_stop_percent = config.get('trailing_stop_percent', 20)  # 20% trailing
        
        # Performance tracking
        self.signal_performance = []
        self.false_positive_count = 0
        self.true_positive_count = 0
        
    async def generate_signals(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate momentum signals with advanced confirmation
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            
        Returns:
            Confirmed signal or None
        """
        try:
            # Get historical data
            historical_data = await self._get_historical_data(symbol, periods=100)
            if historical_data is None or len(historical_data) < 50:
                return None
            
            # Step 1: Check primary momentum signal
            momentum_signal = await self._check_momentum_signal(symbol, historical_data, market_data)
            if not momentum_signal:
                return None
            
            # Step 2: Apply advanced confirmation system
            confirmed_signal = self.confirmation_system.validate_signal(
                momentum_signal,
                historical_data,
                await self._get_option_chain(symbol)
            )
            
            if not confirmed_signal:
                self.logger.info(f"Signal rejected by confirmation system for {symbol}")
                return None
            
            # Step 3: Validate for false positives
            market_context = {
                'active_positions': await self._get_active_positions(),
                'vix_level': await self._get_vix_level(),
                'market_breadth': await self._get_market_breadth()
            }
            
            is_valid, validation_details = self.signal_validator.validate(
                self._convert_to_validation_format(confirmed_signal),
                historical_data,
                market_context
            )
            
            if not is_valid:
                self.logger.info(f"Signal failed validation: {validation_details['recommendation']}")
                self.false_positive_count += 1
                return None
            
            # Step 4: Select appropriate option
            option_details = await self._select_option(symbol, confirmed_signal, historical_data)
            if not option_details:
                return None
            
            # Create final signal
            final_signal = {
                'symbol': symbol,
                'type': confirmed_signal.signal_type,
                'action': 'BUY',
                'option_type': 'CE' if confirmed_signal.signal_type == 'BUY' else 'PE',
                'strike': option_details['strike'],
                'expiry': option_details['expiry'],
                'option_price': option_details['ltp'],
                'underlying_price': market_data['ltp'],
                'stop_loss': confirmed_signal.stop_loss,
                'target': confirmed_signal.target_price,
                'risk_reward': confirmed_signal.risk_reward,
                'confluence_score': confirmed_signal.confluence_score,
                'confirmations': confirmed_signal.confirmations,
                'false_positive_probability': confirmed_signal.false_positive_probability,
                'entry_time': datetime.now(),
                'metadata': {
                    'momentum_value': momentum_signal.get('momentum_value'),
                    'volume_ratio': momentum_signal.get('volume_ratio'),
                    'strength': confirmed_signal.strength.name,
                    'validation_score': validation_details.get('validation_results', {})
                }
            }
            
            self.logger.info(f"Generated confirmed momentum signal for {symbol}: {final_signal['type']}")
            self.true_positive_count += 1
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
            return None
    
    async def _check_momentum_signal(self, symbol: str, data: pd.DataFrame, 
                                    market_data: Dict) -> Optional[Dict]:
        """Check for primary momentum signal"""
        # Calculate momentum
        close_prices = data['close'].values
        volumes = data['volume'].values
        
        # Price momentum (rate of change)
        momentum = self.momentum_indicator.rate_of_change(close_prices, period=5)
        current_momentum = momentum[-1] if len(momentum) > 0 else 0
        
        # Volume confirmation
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
        current_volume = market_data.get('volume', volumes[-1])
        volume_spike = current_volume > avg_volume * self.volume_spike_multiplier
        
        # Check thresholds
        if abs(current_momentum) < self.momentum_threshold:
            return None
        
        if not volume_spike:
            return None
        
        # Determine direction
        signal_type = 'BUY' if current_momentum > 0 else 'SELL'
        
        # Additional momentum checks
        rsi = self.momentum_indicator.rsi(close_prices)
        current_rsi = rsi[-1] if len(rsi) > 0 else 50
        
        # Avoid extremes
        if signal_type == 'BUY' and current_rsi > 75:
            return None
        if signal_type == 'SELL' and current_rsi < 25:
            return None
        
        return {
            'symbol': symbol,
            'type': signal_type,
            'indicator': 'momentum',
            'momentum_value': current_momentum,
            'volume_ratio': current_volume / avg_volume,
            'rsi': current_rsi,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _select_option(self, symbol: str, signal: ConfirmationSignal,
                           data: pd.DataFrame) -> Optional[Dict]:
        """Select appropriate option based on signal strength"""
        option_chain = await self._get_option_chain(symbol)
        if not option_chain:
            return None
        
        current_price = data['close'].iloc[-1]
        
        # Get valid expiries (15-30 days)
        valid_expiries = self._get_valid_expiries(option_chain)
        if not valid_expiries:
            return None
        
        # Select expiry based on signal strength
        if signal.strength.value >= 3:  # STRONG or VERY_STRONG
            # Use monthly expiry for strong signals
            selected_expiry = valid_expiries[0]
        else:
            # Use weekly for moderate signals
            selected_expiry = valid_expiries[-1] if len(valid_expiries) > 1 else valid_expiries[0]
        
        # Strike selection based on momentum
        strikes = sorted([int(s) for s in option_chain[selected_expiry]['strikes'].keys()])
        
        if signal.signal_type == 'BUY':
            # For calls: ATM to 2 strikes OTM based on strength
            atm_strike = min(strikes, key=lambda x: abs(x - current_price))
            strike_index = strikes.index(atm_strike)
            
            if signal.strength.value >= 3:
                # Strong signal: ATM or 1 strike OTM
                otm_offset = min(1, len(strikes) - strike_index - 1)
            else:
                # Moderate signal: 1-2 strikes OTM
                otm_offset = min(2, len(strikes) - strike_index - 1)
            
            selected_strike = strikes[strike_index + otm_offset]
            option_type = 'CE'
            
        else:  # SELL signal
            # For puts: ATM to 2 strikes OTM based on strength
            atm_strike = min(strikes, key=lambda x: abs(x - current_price))
            strike_index = strikes.index(atm_strike)
            
            if signal.strength.value >= 3:
                # Strong signal: ATM or 1 strike OTM
                otm_offset = max(-1, -strike_index)
            else:
                # Moderate signal: 1-2 strikes OTM
                otm_offset = max(-2, -strike_index)
            
            selected_strike = strikes[strike_index + otm_offset]
            option_type = 'PE'
        
        # Get option details
        option = option_chain[selected_expiry]['strikes'][str(selected_strike)].get(option_type)
        
        if not option:
            return None
        
        # Validate option liquidity
        if option.get('oi', 0) < 1000 or option.get('volume', 0) < 100:
            return None
        
        # Check option price limits
        max_premium = self.available_capital * 0.02  # Max 2% per trade
        if option.get('ltp', 0) > max_premium:
            return None
        
        return {
            'strike': selected_strike,
            'expiry': selected_expiry,
            'option_type': option_type,
            'ltp': option['ltp'],
            'iv': option.get('iv', 0),
            'oi': option.get('oi', 0),
            'volume': option.get('volume', 0)
        }
    
    async def calculate_position_size(self, signal: Dict[str, Any]) -> int:
        """Calculate position size with dynamic adjustment"""
        base_risk = self.available_capital * 0.01  # 1% base risk
        
        # Adjust based on signal strength
        strength_multipliers = {
            'VERY_STRONG': 1.5,
            'STRONG': 1.2,
            'MODERATE': 1.0,
            'WEAK': 0.5
        }
        
        strength = signal.get('metadata', {}).get('strength', 'MODERATE')
        multiplier = strength_multipliers.get(strength, 1.0)
        
        # Adjust based on false positive probability
        fp_prob = signal.get('false_positive_probability', 0.3)
        if fp_prob < 0.2:
            multiplier *= 1.1
        elif fp_prob > 0.4:
            multiplier *= 0.8
        
        # Calculate final position size
        adjusted_risk = base_risk * multiplier
        option_price = signal['option_price']
        lot_size = self._get_lot_size(signal['symbol'])
        
        if option_price > 0:
            max_lots = int(adjusted_risk / (option_price * lot_size))
            return max(1, min(max_lots, 5))  # 1-5 lots
        
        return 1
    
    async def should_exit_position(self, position: Dict[str, Any], 
                                 current_data: Dict[str, Any]) -> bool:
        """Enhanced exit logic with trailing stops"""
        symbol = position['symbol']
        entry_time = position['entry_time']
        entry_price = position['entry_price']
        
        # Get current option price
        option_quote = await self._get_option_quote(
            position['option_symbol'],
            position['exchange']
        )
        
        if not option_quote:
            return False
        
        current_price = option_quote.get('ltp', 0)
        
        # Calculate P&L
        pnl_percent = ((current_price - entry_price) / entry_price) * 100
        
        # 1. Stop loss check
        stop_loss = position.get('stop_loss_percent', -50)
        if pnl_percent <= stop_loss:
            self.logger.info(f"Stop loss hit for {symbol}: {pnl_percent:.2f}%")
            await self._record_exit_reason(position, 'STOP_LOSS', pnl_percent)
            return True
        
        # 2. Target check
        target = position.get('target_percent', 100)
        if pnl_percent >= target:
            self.logger.info(f"Target hit for {symbol}: {pnl_percent:.2f}%")
            await self._record_exit_reason(position, 'TARGET_HIT', pnl_percent)
            return True
        
        # 3. Trailing stop logic
        if pnl_percent >= self.trailing_stop_activate:
            # Activate trailing stop
            if 'trailing_stop_activated' not in position:
                position['trailing_stop_activated'] = True
                position['highest_pnl'] = pnl_percent
                self.logger.info(f"Trailing stop activated for {symbol} at {pnl_percent:.2f}%")
            
            # Update highest P&L
            if pnl_percent > position.get('highest_pnl', 0):
                position['highest_pnl'] = pnl_percent
            
            # Check trailing stop
            trailing_level = position['highest_pnl'] * (1 - self.trailing_stop_percent / 100)
            if pnl_percent <= trailing_level:
                self.logger.info(f"Trailing stop hit for {symbol}: {pnl_percent:.2f}%")
                await self._record_exit_reason(position, 'TRAILING_STOP', pnl_percent)
                return True
        
        # 4. Time-based exit
        hold_time = (datetime.now() - entry_time).total_seconds() / 60
        if hold_time > self.max_hold_minutes:
            self.logger.info(f"Time exit for {symbol}: {hold_time:.0f} minutes")
            await self._record_exit_reason(position, 'TIME_EXIT', pnl_percent)
            return True
        
        # 5. Signal reversal check
        if await self._check_signal_reversal(symbol, current_data):
            self.logger.info(f"Signal reversal for {symbol}")
            await self._record_exit_reason(position, 'SIGNAL_REVERSAL', pnl_percent)
            return True
        
        # 6. Underlying stop/target check
        underlying_price = current_data.get('ltp', 0)
        if position.get('underlying_stop') and underlying_price <= position['underlying_stop']:
            self.logger.info(f"Underlying stop hit for {symbol}")
            await self._record_exit_reason(position, 'UNDERLYING_STOP', pnl_percent)
            return True
        
        if position.get('underlying_target') and underlying_price >= position['underlying_target']:
            self.logger.info(f"Underlying target hit for {symbol}")
            await self._record_exit_reason(position, 'UNDERLYING_TARGET', pnl_percent)
            return True
        
        return False
    
    def _convert_to_validation_format(self, signal: ConfirmationSignal) -> Dict:
        """Convert ConfirmationSignal to validator format"""
        return {
            'symbol': signal.symbol,
            'type': signal.signal_type,
            'timestamp': signal.timestamp.isoformat(),
            'confluence_score': signal.confluence_score,
            'confirmations': signal.confirmations,
            'strength': signal.strength.value,
            'entry_price': signal.entry_price
        }
    
    async def _record_exit_reason(self, position: Dict, reason: str, pnl: float):
        """Record exit reason for performance analysis"""
        exit_record = {
            'position_id': position.get('id'),
            'symbol': position['symbol'],
            'exit_reason': reason,
            'pnl_percent': pnl,
            'hold_time_minutes': (datetime.now() - position['entry_time']).total_seconds() / 60,
            'confirmations': position.get('confirmations', []),
            'confluence_score': position.get('confluence_score', 0)
        }
        
        self.signal_performance.append(exit_record)
        
        # Update signal validator with outcome
        if pnl > 0:
            outcome = {'profitable': True, 'reason': reason}
        else:
            outcome = {'profitable': False, 'reason': reason}
        
        self.signal_validator.update_signal_outcome(
            str(position.get('id')),
            outcome
        )
    
    async def _check_signal_reversal(self, symbol: str, current_data: Dict) -> bool:
        """Check if momentum has reversed"""
        historical_data = await self._get_historical_data(symbol, periods=20)
        if not historical_data or len(historical_data) < 10:
            return False
        
        # Check current momentum
        close_prices = historical_data['close'].values
        momentum = self.momentum_indicator.rate_of_change(close_prices, period=5)
        
        if len(momentum) < 2:
            return False
        
        # Check for momentum reversal
        current_momentum = momentum[-1]
        previous_momentum = momentum[-2]
        
        # Reversal if momentum changes sign significantly
        if abs(current_momentum) > self.momentum_threshold * 0.5:
            if current_momentum * previous_momentum < 0:  # Sign change
                return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get enhanced bot status"""
        base_status = super().get_status()
        
        # Add momentum-specific metrics
        base_status['momentum_metrics'] = {
            'true_positives': self.true_positive_count,
            'false_positives': self.false_positive_count,
            'signal_accuracy': self.true_positive_count / (self.true_positive_count + self.false_positive_count + 1),
            'avg_hold_time': self._calculate_avg_hold_time(),
            'win_rate_by_strength': self._calculate_win_rate_by_strength()
        }
        
        # Add confirmation system metrics
        base_status['confirmation_metrics'] = self.confirmation_system.get_performance_metrics()
        
        # Add validator stats
        base_status['validator_stats'] = self.signal_validator.get_validation_stats()
        
        return base_status
    
    def _calculate_avg_hold_time(self) -> float:
        """Calculate average position hold time"""
        if not self.signal_performance:
            return 0
        
        hold_times = [p.get('hold_time_minutes', 0) for p in self.signal_performance]
        return np.mean(hold_times) if hold_times else 0
    
    def _calculate_win_rate_by_strength(self) -> Dict[str, float]:
        """Calculate win rate by signal strength"""
        strength_performance = {}
        
        for record in self.signal_performance:
            strength = record.get('metadata', {}).get('strength', 'UNKNOWN')
            if strength not in strength_performance:
                strength_performance[strength] = {'wins': 0, 'total': 0}
            
            strength_performance[strength]['total'] += 1
            if record.get('pnl_percent', 0) > 0:
                strength_performance[strength]['wins'] += 1
        
        win_rates = {}
        for strength, perf in strength_performance.items():
            if perf['total'] > 0:
                win_rates[strength] = perf['wins'] / perf['total']
        
        return win_rates