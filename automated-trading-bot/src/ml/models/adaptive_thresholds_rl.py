"""
Reinforcement Learning for Adaptive Indicator Thresholds
Uses RL to dynamically adjust indicator thresholds based on market conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
import pickle
from pathlib import Path
import json
import gym
from gym import spaces

try:
    from stable_baselines3 import PPO, DQN, SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Stable-baselines3 not available. RL model will use fallback implementation.")

from dataclasses import dataclass


@dataclass
class ThresholdAction:
    """Threshold adjustment action"""
    indicator: str  # 'rsi', 'macd', 'bollinger', etc.
    threshold_type: str  # 'overbought', 'oversold', 'signal', 'noise'
    adjustment: float  # -1.0 to 1.0 (relative adjustment)
    new_value: float  # Absolute new threshold value
    confidence: float  # Action confidence 0-1


@dataclass
class MarketState:
    """Current market state for RL environment"""
    volatility: float
    trend_strength: float
    volume_ratio: float
    volume_price_correlation: float
    obv_momentum: float
    volume_flow_strength: float
    mfi: float
    time_of_day: float  # 0-1 normalized
    market_regime: str  # 'trending', 'ranging', 'volatile'
    recent_performance: float  # Recent trading performance
    current_thresholds: Dict[str, float]


@dataclass
class RLConfig:
    """Configuration for RL model"""
    algorithm: str = 'PPO'  # 'PPO', 'DQN', 'SAC'
    learning_rate: float = 3e-4
    total_timesteps: int = 100000
    eval_frequency: int = 5000
    threshold_bounds: Dict[str, Tuple[float, float]] = None
    reward_weights: Dict[str, float] = None


class TradingEnvironment(gym.Env):
    """
    Custom trading environment for threshold optimization
    """
    
    def __init__(self, market_data: pd.DataFrame, initial_thresholds: Dict[str, float],
                 reward_weights: Optional[Dict[str, float]] = None):
        """Initialize trading environment"""
        super(TradingEnvironment, self).__init__()
        
        self.market_data = market_data
        self.initial_thresholds = initial_thresholds
        self.current_thresholds = initial_thresholds.copy()
        
        # Reward weights
        self.reward_weights = reward_weights or {
            'profit': 1.0,
            'sharpe': 0.5,
            'drawdown': -0.5,
            'trade_frequency': -0.1
        }
        
        # Define action and observation spaces
        self.threshold_names = list(initial_thresholds.keys())
        self.num_thresholds = len(self.threshold_names)
        
        # Action space: adjustment for each threshold (-1 to 1)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_thresholds,), dtype=np.float32
        )
        
        # Observation space: market state + current thresholds
        obs_dim = 12 + self.num_thresholds  # enhanced market features + thresholds
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # State tracking
        self.current_step = 0
        self.episode_length = min(1000, len(market_data) - 50)
        self.performance_history = []
        self.trade_history = []
        
        # Initialize indicators for market state
        self._calculate_market_features()
    
    def _calculate_market_features(self):
        """Calculate market features for state representation"""
        data = self.market_data.copy()
        
        # Volatility (rolling std of returns)
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(20).std()
        
        # Trend strength (price vs moving averages)
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        data['trend_strength'] = (data['close'] - data['sma_20']) / data['sma_20']
        
        # Enhanced Volume Features for RL State Representation
        # These features help the RL agent understand volume patterns and money flow
        # Basic volume ratio
        data['volume_avg'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_avg']
        
        # Volume-price relationship
        data['price_change'] = data['close'].pct_change()
        data['volume_price_correlation'] = data['volume_ratio'].rolling(10).corr(data['price_change'])
        
        # On Balance Volume (OBV) momentum
        data['obv'] = (data['volume'] * np.sign(data['price_change'])).cumsum()
        data['obv_ma'] = data['obv'].rolling(14).mean()
        data['obv_momentum'] = (data['obv'] - data['obv_ma']) / data['obv_ma']
        
        # Volume flow strength
        data['volume_flow'] = np.where(data['price_change'] > 0, data['volume'], -data['volume'])
        data['volume_flow_ma'] = data['volume_flow'].rolling(10).mean()
        data['volume_flow_strength'] = data['volume_flow_ma'] / data['volume_avg']
        
        # Money Flow Index (approximation)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        positive_flow = money_flow.where(data['price_change'] > 0, 0).rolling(14).sum()
        negative_flow = money_flow.where(data['price_change'] < 0, 0).rolling(14).sum()
        money_flow_ratio = positive_flow / (negative_flow + 1e-10)  # Avoid division by zero
        data['mfi'] = 100 - (100 / (1 + money_flow_ratio))
        
        # RSI for additional market context
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['close'].ewm(span=12).mean()
        exp2 = data['close'].ewm(span=26).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        
        self.market_features = data.fillna(method='bfill').fillna(method='ffill')
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.current_thresholds = self.initial_thresholds.copy()
        self.performance_history = []
        self.trade_history = []
        
        return self._get_observation()
    
    def step(self, action):
        """Execute one time step"""
        # Apply threshold adjustments
        self._apply_threshold_adjustments(action)
        
        # Generate trading signals with new thresholds
        signals = self._generate_signals()
        
        # Calculate performance metrics
        reward = self._calculate_reward(signals)
        
        # Update state
        self.current_step += 1
        done = self.current_step >= self.episode_length
        
        # Store performance
        self.performance_history.append(reward)
        
        obs = self._get_observation()
        info = {
            'thresholds': self.current_thresholds.copy(),
            'signals_generated': len(signals),
            'current_performance': reward
        }
        
        return obs, reward, done, info
    
    def _apply_threshold_adjustments(self, action):
        """Apply RL agent's threshold adjustments"""
        for i, threshold_name in enumerate(self.threshold_names):
            adjustment = action[i]
            current_value = self.current_thresholds[threshold_name]
            
            # Define adjustment bounds based on threshold type
            if 'rsi' in threshold_name.lower():
                if 'overbought' in threshold_name.lower():
                    bounds = (60, 90)
                else:  # oversold
                    bounds = (10, 40)
            elif 'macd' in threshold_name.lower():
                bounds = (-0.1, 0.1)
            elif 'bollinger' in threshold_name.lower():
                bounds = (1.5, 3.0)
            else:
                # Generic bounds
                bounds = (current_value * 0.5, current_value * 1.5)
            
            # Apply adjustment with bounds
            adjustment_magnitude = (bounds[1] - bounds[0]) * 0.1  # 10% max adjustment
            new_value = current_value + (adjustment * adjustment_magnitude)
            new_value = max(bounds[0], min(bounds[1], new_value))
            
            self.current_thresholds[threshold_name] = new_value
    
    def _generate_signals(self):
        """Generate trading signals using current thresholds"""
        signals = []
        start_idx = max(50, self.current_step)
        end_idx = min(start_idx + 50, len(self.market_features))
        
        data_slice = self.market_features.iloc[start_idx:end_idx]
        
        for i, (_, row) in enumerate(data_slice.iterrows()):
            signal_strength = 0
            signal_count = 0
            
            # RSI signals
            if 'rsi_overbought' in self.current_thresholds:
                if row['rsi'] > self.current_thresholds['rsi_overbought']:
                    signal_strength -= 1
                    signal_count += 1
            
            if 'rsi_oversold' in self.current_thresholds:
                if row['rsi'] < self.current_thresholds['rsi_oversold']:
                    signal_strength += 1
                    signal_count += 1
            
            # MACD signals
            if 'macd_threshold' in self.current_thresholds:
                macd_signal = row['macd'] - row['macd_signal']
                if abs(macd_signal) > self.current_thresholds['macd_threshold']:
                    signal_strength += 1 if macd_signal > 0 else -1
                    signal_count += 1
            
            # Only generate signal if we have indicator agreement
            if signal_count > 0:
                signals.append({
                    'timestamp': row.name,
                    'signal': signal_strength / signal_count,
                    'strength': abs(signal_strength / signal_count),
                    'price': row['close']
                })
        
        return signals
    
    def _calculate_reward(self, signals):
        """Calculate reward based on trading signals performance"""
        if len(signals) == 0:
            return -0.1  # Penalty for no signals
        
        # Simulate trading performance
        total_return = 0
        trade_count = 0
        returns = []
        
        for signal in signals:
            if abs(signal['signal']) > 0.5:  # Strong enough signal
                # Simple forward-looking return calculation
                current_price = signal['price']
                future_idx = min(len(self.market_features) - 1, 
                               self.market_features.index.get_loc(signal['timestamp']) + 5)
                future_price = self.market_features.iloc[future_idx]['close']
                
                # Calculate return based on signal direction
                if signal['signal'] > 0:  # Buy signal
                    trade_return = (future_price - current_price) / current_price
                else:  # Sell signal
                    trade_return = (current_price - future_price) / current_price
                
                total_return += trade_return
                returns.append(trade_return)
                trade_count += 1
        
        if trade_count == 0:
            return -0.05
        
        # Calculate reward components
        avg_return = total_return / trade_count
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) if len(returns) > 1 else 0
        
        # Penalty for too many trades
        trade_frequency_penalty = max(0, (trade_count - 10) * 0.01)
        
        # Combined reward
        reward = (
            avg_return * self.reward_weights['profit'] +
            sharpe * self.reward_weights['sharpe'] +
            (-trade_frequency_penalty) * self.reward_weights['trade_frequency']
        )
        
        return reward
    
    def _get_observation(self):
        """Get current observation state"""
        if self.current_step >= len(self.market_features):
            # Return last available observation
            row = self.market_features.iloc[-1]
        else:
            row = self.market_features.iloc[self.current_step]
        
        # Enhanced Market state features with volume analysis
        market_obs = [
            row['volatility'] if pd.notna(row['volatility']) else 0,
            row['trend_strength'] if pd.notna(row['trend_strength']) else 0,
            row['volume_ratio'] if pd.notna(row['volume_ratio']) else 1,
            (self.current_step % 390) / 390,  # Time of day (normalized)
            row['rsi'] / 100 if pd.notna(row['rsi']) else 0.5,
            row['macd'] if pd.notna(row['macd']) else 0,
            # Enhanced volume features
            row['volume_price_correlation'] if pd.notna(row['volume_price_correlation']) else 0,
            np.tanh(row['obv_momentum']) if pd.notna(row['obv_momentum']) else 0,  # Bound between -1,1
            np.tanh(row['volume_flow_strength']) if pd.notna(row['volume_flow_strength']) else 0,  # Bound between -1,1
            row['mfi'] / 100 if pd.notna(row['mfi']) else 0.5,  # Normalize MFI to 0-1
            np.mean(self.performance_history[-10:]) if self.performance_history else 0,
            # Volume trend strength (helps RL understand volume patterns)
            np.tanh(row['volume_ratio'] - 1) if pd.notna(row['volume_ratio']) else 0  # Normalized volume deviation
        ]
        
        # Current thresholds (normalized)
        threshold_obs = []
        for threshold_name in self.threshold_names:
            value = self.current_thresholds[threshold_name]
            # Normalize based on typical ranges
            if 'rsi' in threshold_name.lower():
                normalized = value / 100
            elif 'macd' in threshold_name.lower():
                normalized = np.tanh(value * 10)  # Bound between -1 and 1
            else:
                normalized = np.tanh(value)
            threshold_obs.append(normalized)
        
        observation = np.array(market_obs + threshold_obs, dtype=np.float32)
        return observation
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Thresholds: {self.current_thresholds}")
            if self.performance_history:
                print(f"Recent Performance: {np.mean(self.performance_history[-10:]):.4f}")


class AdaptiveThresholdsRL:
    """
    Reinforcement Learning model for adaptive indicator thresholds
    
    Features:
    - Learns optimal thresholds for different market conditions
    - Adapts to changing market regimes
    - Balances signal frequency with accuracy
    - Uses PPO/DQN/SAC algorithms
    """
    
    def __init__(self, config: Optional[RLConfig] = None):
        """Initialize RL model"""
        self.config = config or RLConfig()
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.model = None
        self.env = None
        self.is_trained = False
        
        # Default thresholds
        self.default_thresholds = {
            'rsi_overbought': 70.0,
            'rsi_oversold': 30.0,
            'macd_threshold': 0.01,
            'bollinger_threshold': 2.0
        }
        
        # Current optimized thresholds
        self.current_thresholds = self.default_thresholds.copy()
        
        # Training history
        self.training_history = []
        self.performance_metrics = {}
        
        if not SB3_AVAILABLE:
            self.logger.warning("Stable-baselines3 not available. Using fallback threshold adaptation.")
    
    def create_environment(self, market_data: pd.DataFrame, 
                          initial_thresholds: Optional[Dict[str, float]] = None) -> TradingEnvironment:
        """Create trading environment for RL training"""
        thresholds = initial_thresholds or self.default_thresholds
        
        env = TradingEnvironment(
            market_data=market_data,
            initial_thresholds=thresholds,
            reward_weights=self.config.reward_weights
        )
        
        return env
    
    def train(self, market_data: pd.DataFrame, 
             initial_thresholds: Optional[Dict[str, float]] = None,
             validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train the RL model
        
        Args:
            market_data: Training market data
            initial_thresholds: Starting threshold values
            validation_data: Optional validation data
            
        Returns:
            Training results and metrics
        """
        if not SB3_AVAILABLE:
            return self._fallback_training(market_data, initial_thresholds)
        
        self.logger.info(f"Training RL model with {len(market_data)} samples")
        
        # Create environment
        self.env = self.create_environment(market_data, initial_thresholds)
        
        # Create validation environment if data provided
        eval_env = None
        if validation_data is not None:
            eval_env = self.create_environment(validation_data, initial_thresholds)
            eval_env = Monitor(eval_env)
        
        # Initialize RL algorithm
        if self.config.algorithm == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                self.env,
                learning_rate=self.config.learning_rate,
                verbose=1
            )
        elif self.config.algorithm == 'DQN':
            # Discretize action space for DQN
            from gym.wrappers import FlattenObservation
            discrete_env = FlattenObservation(self.env)
            self.model = DQN(
                'MlpPolicy',
                discrete_env,
                learning_rate=self.config.learning_rate,
                verbose=1
            )
        elif self.config.algorithm == 'SAC':
            self.model = SAC(
                'MlpPolicy',
                self.env,
                learning_rate=self.config.learning_rate,
                verbose=1
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        # Setup callbacks
        callbacks = []
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path='./models/rl_thresholds/',
                log_path='./logs/rl_training/',
                eval_freq=self.config.eval_frequency,
                deterministic=True
            )
            callbacks.append(eval_callback)
        
        # Train model
        start_time = datetime.now()
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks
        )
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.is_trained = True
        
        # Evaluate trained model
        final_thresholds = self._extract_learned_thresholds(market_data)
        
        # Calculate performance metrics
        evaluation_results = self._evaluate_thresholds(market_data, final_thresholds)
        
        self.performance_metrics = {
            'training_time': training_time,
            'final_thresholds': final_thresholds,
            'evaluation_results': evaluation_results,
            'algorithm': self.config.algorithm,
            'total_timesteps': self.config.total_timesteps
        }
        
        self.current_thresholds = final_thresholds
        
        self.logger.info(f"RL training completed in {training_time:.1f}s")
        self.logger.info(f"Learned thresholds: {final_thresholds}")
        
        return {
            'metrics': self.performance_metrics,
            'thresholds': final_thresholds
        }
    
    def _extract_learned_thresholds(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract learned thresholds from trained model"""
        if not self.model or not self.env:
            return self.default_thresholds.copy()
        
        # Run model on representative data to extract typical threshold values
        obs = self.env.reset()
        threshold_samples = []
        
        for _ in range(min(100, len(market_data) // 10)):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, info = self.env.step(action)
            
            if 'thresholds' in info:
                threshold_samples.append(info['thresholds'].copy())
            
            if done:
                obs = self.env.reset()
        
        # Average the threshold values
        if threshold_samples:
            learned_thresholds = {}
            for key in self.default_thresholds.keys():
                values = [sample[key] for sample in threshold_samples if key in sample]
                learned_thresholds[key] = np.mean(values) if values else self.default_thresholds[key]
            return learned_thresholds
        
        return self.default_thresholds.copy()
    
    def _evaluate_thresholds(self, market_data: pd.DataFrame, 
                           thresholds: Dict[str, float]) -> Dict[str, float]:
        """Evaluate threshold performance"""
        # Create temporary environment for evaluation
        eval_env = self.create_environment(market_data, thresholds)
        
        total_reward = 0
        episode_count = 5
        
        for _ in range(episode_count):
            obs = eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                if self.model:
                    action, _ = self.model.predict(obs, deterministic=True)
                else:
                    action = np.zeros(len(thresholds))
                
                obs, reward, done, _ = eval_env.step(action)
                episode_reward += reward
            
            total_reward += episode_reward
        
        avg_reward = total_reward / episode_count
        
        return {
            'average_reward': avg_reward,
            'evaluation_episodes': episode_count
        }
    
    def adapt_thresholds(self, market_data: pd.DataFrame, 
                        current_state: MarketState) -> Dict[str, ThresholdAction]:
        """
        Adapt thresholds based on current market conditions
        
        Args:
            market_data: Recent market data
            current_state: Current market state
            
        Returns:
            Dictionary of threshold actions
        """
        if not self.is_trained:
            return self._fallback_adaptation(current_state)
        
        # Create temporary environment for prediction
        temp_env = self.create_environment(market_data, self.current_thresholds)
        obs = temp_env._get_observation()
        
        # Get RL model's recommended action
        if SB3_AVAILABLE and self.model:
            action, _ = self.model.predict(obs, deterministic=True)
        else:
            return self._fallback_adaptation(current_state)
        
        # Convert actions to threshold adjustments
        threshold_actions = {}
        threshold_names = list(self.current_thresholds.keys())
        
        for i, threshold_name in enumerate(threshold_names):
            adjustment = action[i]
            current_value = self.current_thresholds[threshold_name]
            
            # Calculate new value (similar to environment logic)
            if 'rsi' in threshold_name.lower():
                if 'overbought' in threshold_name.lower():
                    bounds = (60, 90)
                else:
                    bounds = (10, 40)
            elif 'macd' in threshold_name.lower():
                bounds = (-0.1, 0.1)
            else:
                bounds = (current_value * 0.5, current_value * 1.5)
            
            adjustment_magnitude = (bounds[1] - bounds[0]) * 0.1
            new_value = current_value + (adjustment * adjustment_magnitude)
            new_value = max(bounds[0], min(bounds[1], new_value))
            
            # Update current thresholds
            self.current_thresholds[threshold_name] = new_value
            
            # Create threshold action
            threshold_actions[threshold_name] = ThresholdAction(
                indicator=threshold_name.split('_')[0],
                threshold_type=threshold_name.split('_')[1] if '_' in threshold_name else 'general',
                adjustment=adjustment,
                new_value=new_value,
                confidence=0.8  # Model confidence
            )
        
        return threshold_actions
    
    def _fallback_adaptation(self, current_state: MarketState) -> Dict[str, ThresholdAction]:
        """Fallback threshold adaptation when RL is not available"""
        threshold_actions = {}
        
        # Simple rule-based adaptation
        volatility_factor = current_state.volatility
        
        # Adjust RSI thresholds based on volatility
        if volatility_factor > 0.03:  # High volatility
            rsi_overbought = 75
            rsi_oversold = 25
        elif volatility_factor < 0.01:  # Low volatility
            rsi_overbought = 65
            rsi_oversold = 35
        else:  # Normal volatility
            rsi_overbought = 70
            rsi_oversold = 30
        
        # Create threshold actions
        if 'rsi_overbought' in self.current_thresholds:
            old_value = self.current_thresholds['rsi_overbought']
            self.current_thresholds['rsi_overbought'] = rsi_overbought
            
            threshold_actions['rsi_overbought'] = ThresholdAction(
                indicator='rsi',
                threshold_type='overbought',
                adjustment=(rsi_overbought - old_value) / 30,  # Normalized
                new_value=rsi_overbought,
                confidence=0.6
            )
        
        if 'rsi_oversold' in self.current_thresholds:
            old_value = self.current_thresholds['rsi_oversold']
            self.current_thresholds['rsi_oversold'] = rsi_oversold
            
            threshold_actions['rsi_oversold'] = ThresholdAction(
                indicator='rsi',
                threshold_type='oversold',
                adjustment=(rsi_oversold - old_value) / 30,
                new_value=rsi_oversold,
                confidence=0.6
            )
        
        return threshold_actions
    
    def _fallback_training(self, market_data: pd.DataFrame, 
                          initial_thresholds: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """Fallback training when stable-baselines3 is not available"""
        self.logger.info("Using fallback training (rule-based threshold adaptation)")
        self.is_trained = True
        
        # Simple optimization based on historical performance
        thresholds = initial_thresholds or self.default_thresholds
        self.current_thresholds = thresholds.copy()
        
        # Calculate basic performance metrics
        volatility = market_data['close'].pct_change().std()
        
        self.performance_metrics = {
            'fallback_mode': True,
            'market_volatility': volatility,
            'total_samples': len(market_data),
            'optimized_thresholds': self.current_thresholds
        }
        
        return {'metrics': self.performance_metrics}
    
    def get_current_thresholds(self) -> Dict[str, float]:
        """Get current optimized thresholds"""
        return self.current_thresholds.copy()
    
    def save_model(self, filepath: str):
        """Save trained model and components"""
        model_dir = Path(filepath).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save RL model
        if SB3_AVAILABLE and self.model:
            self.model.save(f"{filepath}_rl_model")
        
        # Save thresholds and configuration
        model_info = {
            'config': {
                'algorithm': self.config.algorithm,
                'learning_rate': self.config.learning_rate,
                'total_timesteps': self.config.total_timesteps
            },
            'current_thresholds': self.current_thresholds,
            'default_thresholds': self.default_thresholds,
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained,
            'sb3_available': SB3_AVAILABLE
        }
        
        with open(f"{filepath}_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        self.logger.info(f"RL threshold model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and components"""
        try:
            # Load RL model
            if SB3_AVAILABLE and Path(f"{filepath}_rl_model.zip").exists():
                if self.config.algorithm == 'PPO':
                    self.model = PPO.load(f"{filepath}_rl_model")
                elif self.config.algorithm == 'DQN':
                    self.model = DQN.load(f"{filepath}_rl_model")
                elif self.config.algorithm == 'SAC':
                    self.model = SAC.load(f"{filepath}_rl_model")
            
            # Load configuration
            with open(f"{filepath}_info.json", 'r') as f:
                model_info = json.load(f)
                self.current_thresholds = model_info['current_thresholds']
                self.default_thresholds = model_info['default_thresholds']
                self.performance_metrics = model_info['performance_metrics']
                self.is_trained = model_info['is_trained']
            
            self.logger.info(f"RL threshold model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary and performance metrics"""
        summary = {
            'model_type': 'Adaptive_Thresholds_RL',
            'is_trained': self.is_trained,
            'sb3_available': SB3_AVAILABLE,
            'algorithm': self.config.algorithm,
            'current_thresholds': self.current_thresholds,
            'default_thresholds': self.default_thresholds,
            'performance': self.performance_metrics
        }
        
        return summary


# Example usage
if __name__ == "__main__":
    # Generate sample market data for testing
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=2000, freq='5min')
    
    # Generate realistic market data
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
    
    market_data = pd.DataFrame({
        'open': prices + np.random.randn(len(prices)) * 0.5,
        'high': prices + np.abs(np.random.randn(len(prices))) * 1.0,
        'low': prices - np.abs(np.random.randn(len(prices))) * 1.0,
        'close': prices,
        'volume': np.random.randint(50000, 200000, len(prices))
    }, index=dates)
    
    # Ensure OHLC consistency
    market_data['high'] = market_data[['open', 'high', 'close']].max(axis=1)
    market_data['low'] = market_data[['open', 'low', 'close']].min(axis=1)
    
    # Initialize RL model
    config = RLConfig(total_timesteps=10000)  # Reduced for testing
    model = AdaptiveThresholdsRL(config)
    
    # Split data
    train_data = market_data.iloc[:1500]
    val_data = market_data.iloc[1500:]
    
    print("Training RL Threshold Adaptation Model...")
    results = model.train(train_data, validation_data=val_data)
    print(f"Training Results: {results['metrics']}")
    
    # Test threshold adaptation
    current_state = MarketState(
        volatility=0.02,
        trend_strength=0.1,
        volume_ratio=1.2,
        time_of_day=0.5,
        market_regime='ranging',
        recent_performance=0.05,
        current_thresholds=model.get_current_thresholds()
    )
    
    threshold_actions = model.adapt_thresholds(val_data.tail(50), current_state)
    
    print(f"\nAdaptive Threshold Actions:")
    for name, action in threshold_actions.items():
        print(f"{name}: {action.new_value:.2f} (adjustment: {action.adjustment:.3f}, confidence: {action.confidence:.3f})")
    
    # Model summary
    summary = model.get_model_summary()
    print(f"\nModel Summary: {summary}")