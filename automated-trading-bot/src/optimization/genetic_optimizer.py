"""
Genetic Algorithm Optimizer for Trading Indicators and Strategies
Uses DEAP library for evolutionary optimization of trading parameters
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# DEAP imports for genetic algorithms
from deap import algorithms, base, creator, tools
from deap.benchmarks.tools import diversity, convergence, hypervolume
import random
import multiprocessing

# Sklearn for validation
from sklearn.metrics import sharpe_ratio
from sklearn.model_selection import TimeSeriesSplit

# Import Individual Indicator Intelligence components
try:
    from ..ml.indicator_ensemble import IndicatorEnsemble, EnsembleConfig
    from ..ml.models.rsi_lstm_model import RSILSTMModel
    from ..ml.models.pattern_cnn_model import PatternCNNModel
    from ..ml.models.adaptive_thresholds_rl import AdaptiveThresholdsRL
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False


@dataclass
class OptimizationResult:
    """Results from genetic algorithm optimization"""
    best_individual: List[float]
    best_fitness: float
    best_parameters: Dict[str, Any]
    generation_stats: List[Dict[str, float]]
    total_generations: int
    population_size: int
    optimization_time: float
    validation_metrics: Dict[str, float]


@dataclass
class ParameterSpace:
    """Definition of parameter search space"""
    name: str
    min_value: float
    max_value: float
    param_type: str = 'float'  # 'float', 'int', 'choice'
    choices: Optional[List] = None
    step: Optional[float] = None


class GeneticOptimizer:
    """
    Genetic Algorithm Optimizer for trading strategies
    
    Features:
    - Multi-objective optimization (return, risk, drawdown)
    - Parallel evaluation using multiprocessing
    - Time series cross-validation
    - Parameter space constraints
    - Elite preservation
    - Adaptive mutation rates
    """
    
    def __init__(self, 
                 population_size: int = 100,
                 generations: int = 50,
                 crossover_prob: float = 0.7,
                 mutation_prob: float = 0.2,
                 elite_size: int = 10,
                 n_jobs: int = -1,
                 random_seed: int = 42):
        """Initialize genetic optimizer"""
        
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elite_size = elite_size
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        self.random_seed = random_seed
        
        self.logger = logging.getLogger(__name__)
        
        # Parameter space definition
        self.parameter_spaces: List[ParameterSpace] = []
        
        # DEAP setup
        self.toolbox = base.Toolbox()
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.logbook = tools.Logbook()
        
        # Results storage
        self.optimization_history = []
        self.best_results = {}
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        self.logger.info(f"Initialized genetic optimizer with {self.n_jobs} processes")
    
    def add_parameter(self, name: str, min_val: float, max_val: float, 
                     param_type: str = 'float', choices: Optional[List] = None,
                     step: Optional[float] = None):
        """Add parameter to optimization space"""
        param_space = ParameterSpace(
            name=name,
            min_value=min_val,
            max_value=max_val,
            param_type=param_type,
            choices=choices,
            step=step
        )
        self.parameter_spaces.append(param_space)
        self.logger.info(f"Added parameter: {name} [{min_val}, {max_val}] ({param_type})")
    
    def setup_deap_framework(self):
        """Setup DEAP genetic algorithm framework"""
        
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize fitness
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Register genetic operators
        self.toolbox.register("attr_float", self._generate_parameter_value)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                             self.toolbox.attr_float, len(self.parameter_spaces))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register evolution operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._custom_mutation)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Statistics
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        
        self.logger.info("DEAP framework configured")
    
    def _generate_parameter_value(self) -> float:
        """Generate random parameter value within constraints"""
        # This will be called for each parameter in the individual
        # We need to track which parameter we're generating
        if not hasattr(self, '_param_index'):
            self._param_index = 0
        
        param_space = self.parameter_spaces[self._param_index % len(self.parameter_spaces)]
        self._param_index += 1
        
        if param_space.param_type == 'float':
            return random.uniform(param_space.min_value, param_space.max_value)
        elif param_space.param_type == 'int':
            return random.randint(int(param_space.min_value), int(param_space.max_value))
        elif param_space.param_type == 'choice':
            return random.choice(param_space.choices)
        else:
            return random.uniform(param_space.min_value, param_space.max_value)
    
    def _custom_mutation(self, individual, indpb=0.1):
        """Custom mutation operator with parameter-specific constraints"""
        for i, param_space in enumerate(self.parameter_spaces):
            if random.random() < indpb:
                if param_space.param_type == 'float':
                    # Gaussian mutation with bounds
                    sigma = (param_space.max_value - param_space.min_value) * 0.1
                    individual[i] += random.gauss(0, sigma)
                    individual[i] = max(param_space.min_value, 
                                      min(param_space.max_value, individual[i]))
                elif param_space.param_type == 'int':
                    # Integer mutation
                    individual[i] = random.randint(int(param_space.min_value),
                                                 int(param_space.max_value))
                elif param_space.param_type == 'choice':
                    # Choice mutation
                    individual[i] = random.choice(param_space.choices)
        
        return individual,
    
    def _evaluate_individual(self, individual: List[float]) -> Tuple[float]:
        """Evaluate fitness of individual (to be implemented by subclass)"""
        raise NotImplementedError("Subclass must implement _evaluate_individual method")
    
    def _decode_individual(self, individual: List[float]) -> Dict[str, Any]:
        """Convert individual genes to parameter dictionary"""
        parameters = {}
        for i, param_space in enumerate(self.parameter_spaces):
            value = individual[i]
            
            if param_space.param_type == 'int':
                value = int(round(value))
            elif param_space.param_type == 'choice':
                # Find closest choice
                if isinstance(value, (int, float)):
                    idx = int(round(value)) % len(param_space.choices)
                    value = param_space.choices[idx]
            
            parameters[param_space.name] = value
        
        return parameters
    
    def optimize(self, 
                 evaluation_function: callable,
                 data: pd.DataFrame,
                 validation_split: float = 0.2,
                 early_stopping: int = 10,
                 verbose: bool = True) -> OptimizationResult:
        """
        Run genetic algorithm optimization
        
        Args:
            evaluation_function: Function to evaluate strategy performance
            data: Historical data for optimization
            validation_split: Fraction for validation
            early_stopping: Generations without improvement to stop
            verbose: Print progress
            
        Returns:
            OptimizationResult with best parameters and statistics
        """
        self.logger.info(f"Starting optimization with {self.population_size} individuals, {self.generations} generations")
        
        # Store evaluation function and data
        self.evaluation_function = evaluation_function
        self.data = data
        
        # Split data for validation
        split_idx = int(len(data) * (1 - validation_split))
        self.train_data = data.iloc[:split_idx].copy()
        self.validation_data = data.iloc[split_idx:].copy()
        
        start_time = datetime.now()
        
        # Setup DEAP framework
        self.setup_deap_framework()
        
        # Initialize population
        population = self.toolbox.population(n=self.population_size)
        
        # Track best fitness
        best_fitness_history = []
        best_individual = None
        best_fitness = float('-inf')
        generations_without_improvement = 0
        
        # Evolution loop
        for generation in range(self.generations):
            gen_start = datetime.now()
            
            # Evaluate population
            if self.n_jobs == 1:
                # Single process evaluation
                fitnesses = list(map(self.toolbox.evaluate, population))
            else:
                # Parallel evaluation
                with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                    fitnesses = list(executor.map(self.toolbox.evaluate, population))
            
            # Assign fitness to individuals
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            
            # Track best individual
            current_best = tools.selBest(population, 1)[0]
            current_best_fitness = current_best.fitness.values[0]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best[:]
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            best_fitness_history.append(best_fitness)
            
            # Record statistics
            record = self.stats.compile(population)
            self.logbook.record(gen=generation, **record)
            
            if verbose:
                gen_time = (datetime.now() - gen_start).total_seconds()
                print(f"Gen {generation:3d}: Best={best_fitness:.4f}, "
                      f"Avg={record['avg']:.4f}, Std={record['std']:.4f}, "
                      f"Time={gen_time:.1f}s")
            
            # Early stopping
            if generations_without_improvement >= early_stopping:
                self.logger.info(f"Early stopping at generation {generation}")
                break
            
            # Selection
            offspring = algorithms.varAnd(population, self.toolbox, 
                                        self.crossover_prob, self.mutation_prob)
            
            # Elite preservation
            elite = tools.selBest(population, self.elite_size)
            
            # Select next generation
            population = self.toolbox.select(offspring + elite, self.population_size)
        
        # Calculate optimization time
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Get best parameters
        best_parameters = self._decode_individual(best_individual)
        
        # Validate best solution
        validation_metrics = self._validate_solution(best_parameters)
        
        # Create result object
        result = OptimizationResult(
            best_individual=best_individual,
            best_fitness=best_fitness,
            best_parameters=best_parameters,
            generation_stats=[dict(record) for record in self.logbook],
            total_generations=len(self.logbook),
            population_size=self.population_size,
            optimization_time=optimization_time,
            validation_metrics=validation_metrics
        )
        
        # Store results
        self.optimization_history.append(result)
        self.best_results[datetime.now().isoformat()] = result
        
        self.logger.info(f"Optimization completed in {optimization_time:.1f}s")
        self.logger.info(f"Best fitness: {best_fitness:.4f}")
        self.logger.info(f"Best parameters: {best_parameters}")
        
        return result
    
    def _validate_solution(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Validate solution on out-of-sample data"""
        try:
            # Evaluate on validation data
            validation_fitness = self.evaluation_function(parameters, self.validation_data)
            
            # Calculate additional metrics
            metrics = {
                'validation_fitness': validation_fitness,
                'overfitting_ratio': validation_fitness / self.best_results.get('train_fitness', 1.0) if self.best_results else 1.0
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return {'validation_fitness': 0.0, 'overfitting_ratio': 0.0}
    
    def save_results(self, filepath: str):
        """Save optimization results"""
        results_data = {
            'parameter_spaces': [
                {
                    'name': ps.name,
                    'min_value': ps.min_value,
                    'max_value': ps.max_value,
                    'param_type': ps.param_type,
                    'choices': ps.choices
                }
                for ps in self.parameter_spaces
            ],
            'optimization_history': self.optimization_history,
            'best_results': self.best_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load optimization results"""
        with open(filepath, 'r') as f:
            results_data = json.load(f)
        
        # Reconstruct parameter spaces
        self.parameter_spaces = []
        for ps_data in results_data['parameter_spaces']:
            self.add_parameter(
                ps_data['name'],
                ps_data['min_value'],
                ps_data['max_value'],
                ps_data['param_type'],
                ps_data.get('choices')
            )
        
        self.optimization_history = results_data['optimization_history']
        self.best_results = results_data['best_results']
        
        self.logger.info(f"Results loaded from {filepath}")


class IndicatorOptimizer(GeneticOptimizer):
    """
    Specialized genetic optimizer for trading indicators
    """
    
    def __init__(self, indicator_class, **kwargs):
        super().__init__(**kwargs)
        self.indicator_class = indicator_class
        
        # Define common indicator parameters
        self._setup_indicator_parameters()
    
    def _setup_indicator_parameters(self):
        """Setup common indicator parameters for optimization"""
        
        # RSI parameters
        self.add_parameter('rsi_period', 5, 50, 'int')
        self.add_parameter('rsi_overbought', 60, 90, 'int')
        self.add_parameter('rsi_oversold', 10, 40, 'int')
        
        # Moving average parameters
        self.add_parameter('ma_fast', 5, 30, 'int')
        self.add_parameter('ma_slow', 20, 100, 'int')
        
        # MACD parameters
        self.add_parameter('macd_fast', 8, 20, 'int')
        self.add_parameter('macd_slow', 20, 40, 'int')
        self.add_parameter('macd_signal', 5, 15, 'int')
        
        # Bollinger Bands
        self.add_parameter('bb_period', 10, 30, 'int')
        self.add_parameter('bb_std', 1.5, 3.0, 'float')
        
        # Volume parameters
        self.add_parameter('volume_period', 10, 50, 'int')
        self.add_parameter('volume_multiplier', 1.2, 3.0, 'float')
        
        # Stop loss and take profit
        self.add_parameter('stop_loss', 0.5, 5.0, 'float')
        self.add_parameter('take_profit', 1.0, 10.0, 'float')
    
    def _evaluate_individual(self, individual: List[float]) -> Tuple[float]:
        """Evaluate indicator strategy performance"""
        try:
            parameters = self._decode_individual(individual)
            
            # Create indicator instance with parameters
            indicator = self.indicator_class(**parameters)
            
            # Generate signals
            signals = indicator.generate_signals(self.train_data)
            
            # Calculate returns
            returns = self._calculate_strategy_returns(signals, self.train_data)
            
            # Calculate fitness (Sharpe ratio with constraints)
            fitness = self._calculate_fitness(returns, parameters)
            
            return (fitness,)
            
        except Exception as e:
            self.logger.debug(f"Evaluation error: {e}")
            return (0.0,)
    
    def _calculate_strategy_returns(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Calculate strategy returns from signals"""
        # Align signals with price data
        aligned_signals = signals.reindex(data.index, method='ffill').fillna(0)
        
        # Calculate price returns
        price_returns = data['close'].pct_change().fillna(0)
        
        # Calculate strategy returns (signal * return)
        strategy_returns = aligned_signals.shift(1) * price_returns
        
        return strategy_returns.fillna(0)
    
    def _calculate_fitness(self, returns: pd.Series, parameters: Dict[str, Any]) -> float:
        """Calculate fitness score with multiple objectives"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        # Base metrics
        total_return = returns.sum()
        sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        max_dd = self._calculate_max_drawdown(returns.cumsum())
        
        # Penalty for too many trades
        trade_count = (returns != 0).sum()
        trade_penalty = max(0, (trade_count - 100) * 0.001)  # Penalty if > 100 trades
        
        # Penalty for extreme parameters
        param_penalty = 0
        if parameters.get('stop_loss', 0) > 4.0:
            param_penalty += 0.1
        if parameters.get('take_profit', 0) > 8.0:
            param_penalty += 0.1
        
        # Combined fitness
        fitness = (
            sharpe * 0.4 +  # Sharpe ratio weight
            total_return * 0.3 +  # Total return weight
            (1 / (1 + abs(max_dd))) * 0.2 +  # Max drawdown weight (inversed)
            (1 / (1 + trade_count/100)) * 0.1 -  # Trade frequency weight
            trade_penalty - param_penalty
        )
        
        return max(0, fitness)  # Ensure non-negative
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(cumulative_returns) == 0:
            return 0.0
        
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / (running_max + 1e-8)
        return abs(drawdown.min())


class StrategyOptimizer(GeneticOptimizer):
    """
    Genetic optimizer for complete trading strategies
    """
    
    def __init__(self, strategy_class, **kwargs):
        super().__init__(**kwargs)
        self.strategy_class = strategy_class
        
        # Setup strategy-specific parameters
        self._setup_strategy_parameters()
    
    def _setup_strategy_parameters(self):
        """Setup strategy-specific parameters"""
        
        # Position sizing
        self.add_parameter('position_size', 0.1, 1.0, 'float')
        self.add_parameter('max_positions', 1, 5, 'int')
        
        # Risk management
        self.add_parameter('portfolio_risk', 0.01, 0.05, 'float')
        self.add_parameter('max_drawdown_limit', 0.05, 0.20, 'float')
        
        # Entry/Exit timing
        self.add_parameter('entry_threshold', 0.6, 0.9, 'float')
        self.add_parameter('exit_threshold', 0.3, 0.7, 'float')
        
        # Market regime filters
        self.add_parameter('trending_filter', 0.0, 1.0, 'float')
        self.add_parameter('volatile_filter', 0.0, 1.0, 'float')
    
    def _evaluate_individual(self, individual: List[float]) -> Tuple[float]:
        """Evaluate complete strategy performance"""
        try:
            parameters = self._decode_individual(individual)
            
            # Create strategy instance
            strategy = self.strategy_class(**parameters)
            
            # Run backtest
            results = strategy.backtest(self.train_data)
            
            # Calculate comprehensive fitness
            fitness = self._calculate_strategy_fitness(results, parameters)
            
            return (fitness,)
            
        except Exception as e:
            self.logger.debug(f"Strategy evaluation error: {e}")
            return (0.0,)
    
    def _calculate_strategy_fitness(self, results: Dict[str, Any], parameters: Dict[str, Any]) -> float:
        """Calculate strategy fitness with multiple objectives"""
        
        # Extract key metrics
        total_return = results.get('total_return', 0)
        sharpe_ratio = results.get('sharpe_ratio', 0)
        max_drawdown = results.get('max_drawdown', 1)
        win_rate = results.get('win_rate', 0)
        profit_factor = results.get('profit_factor', 0)
        
        # Normalize metrics
        normalized_return = min(total_return / 0.5, 2.0)  # Cap at 200%
        normalized_sharpe = min(sharpe_ratio / 2.0, 2.0)  # Cap at 2.0
        normalized_dd = 1 / (1 + max_drawdown)
        normalized_win_rate = win_rate
        normalized_pf = min(profit_factor / 2.0, 1.0)  # Cap at 2.0
        
        # Multi-objective fitness
        fitness = (
            normalized_return * 0.25 +
            normalized_sharpe * 0.25 +
            normalized_dd * 0.20 +
            normalized_win_rate * 0.15 +
            normalized_pf * 0.15
        )
        
        # Penalty for unrealistic parameters
        if parameters.get('position_size', 0) > 0.8:
            fitness *= 0.9  # Too aggressive position sizing
        
        return max(0, fitness)


class IndividualIndicatorOptimizer(GeneticOptimizer):
    """
    Genetic optimizer specifically for Individual Indicator Intelligence
    Optimizes ML model parameters and ensemble weights
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Setup ML model parameters
        self._setup_ml_parameters()
        
        # Storage for models
        self.ml_models = {}
        self.ensemble = None
    
    def _setup_ml_parameters(self):
        """Setup parameters for ML models and ensemble"""
        
        # RSI LSTM parameters
        self.add_parameter('rsi_lstm_units', 20, 100, 'int')
        self.add_parameter('rsi_lstm_dropout', 0.1, 0.5, 'float')
        self.add_parameter('rsi_lstm_sequence_length', 10, 40, 'int')
        self.add_parameter('rsi_lstm_learning_rate', 0.0001, 0.01, 'float')
        
        # Pattern CNN parameters
        self.add_parameter('cnn_image_size', 32, 128, 'int')
        self.add_parameter('cnn_lookback_periods', 30, 80, 'int')
        self.add_parameter('cnn_learning_rate', 0.0001, 0.01, 'float')
        
        # Adaptive Thresholds RL parameters
        self.add_parameter('rl_learning_rate', 0.0001, 0.01, 'float')
        self.add_parameter('rl_total_timesteps', 5000, 50000, 'int')
        
        # Ensemble weights
        self.add_parameter('weight_rsi_lstm', 0.05, 0.25, 'float')
        self.add_parameter('weight_pattern_cnn', 0.05, 0.25, 'float')
        self.add_parameter('weight_adaptive_thresholds', 0.05, 0.20, 'float')
        self.add_parameter('weight_technical_indicators', 0.20, 0.40, 'float')
        self.add_parameter('weight_price_action', 0.15, 0.35, 'float')
        
        # Ensemble consensus parameters
        self.add_parameter('min_consensus_ratio', 0.4, 0.8, 'float')
        self.add_parameter('min_confidence', 0.3, 0.7, 'float')
    
    def _evaluate_individual(self, individual: List[float]) -> Tuple[float]:
        """Evaluate Individual Indicator Intelligence configuration"""
        try:
            parameters = self._decode_individual(individual)
            
            if not ML_MODELS_AVAILABLE:
                # Fallback evaluation
                return self._fallback_evaluation(parameters)
            
            # Create and configure ML models
            models = self._create_ml_models(parameters)
            
            # Create ensemble
            ensemble_config = EnsembleConfig(
                min_consensus_ratio=parameters['min_consensus_ratio'],
                min_confidence=parameters['min_confidence'],
                indicator_weights={
                    'rsi_lstm': parameters['weight_rsi_lstm'],
                    'pattern_cnn': parameters['weight_pattern_cnn'],
                    'adaptive_thresholds': parameters['weight_adaptive_thresholds']
                }
            )
            
            ensemble = IndicatorEnsemble(ensemble_config)
            
            # Add models to ensemble
            for name, model in models.items():
                ensemble.add_ml_model(name, model)
            
            # Evaluate ensemble performance
            fitness = self._evaluate_ensemble_performance(ensemble, self.train_data)
            
            return (fitness,)
            
        except Exception as e:
            self.logger.debug(f"Evaluation error: {e}")
            return (0.0,)
    
    def _create_ml_models(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create ML models with optimized parameters"""
        models = {}
        
        try:
            # RSI LSTM Model
            from ..ml.models.rsi_lstm_model import RSILSTMModel, RSIModelConfig
            
            rsi_config = RSIModelConfig(
                lstm_units=int(parameters['rsi_lstm_units']),
                dropout_rate=parameters['rsi_lstm_dropout'],
                sequence_length=int(parameters['rsi_lstm_sequence_length']),
                learning_rate=parameters['rsi_lstm_learning_rate']
            )
            
            models['rsi_lstm'] = RSILSTMModel(rsi_config)
            
        except Exception as e:
            self.logger.warning(f"Failed to create RSI LSTM model: {e}")
        
        try:
            # Pattern CNN Model
            from ..ml.models.pattern_cnn_model import PatternCNNModel, CNNModelConfig
            
            size = int(parameters['cnn_image_size'])
            cnn_config = CNNModelConfig(
                image_size=(size, size),
                lookback_periods=int(parameters['cnn_lookback_periods']),
                learning_rate=parameters['cnn_learning_rate']
            )
            
            models['pattern_cnn'] = PatternCNNModel(cnn_config)
            
        except Exception as e:
            self.logger.warning(f"Failed to create Pattern CNN model: {e}")
        
        try:
            # Adaptive Thresholds RL Model
            from ..ml.models.adaptive_thresholds_rl import AdaptiveThresholdsRL, RLConfig
            
            rl_config = RLConfig(
                learning_rate=parameters['rl_learning_rate'],
                total_timesteps=int(parameters['rl_total_timesteps'])
            )
            
            models['adaptive_thresholds'] = AdaptiveThresholdsRL(rl_config)
            
        except Exception as e:
            self.logger.warning(f"Failed to create RL model: {e}")
        
        return models
    
    def _evaluate_ensemble_performance(self, ensemble: 'IndicatorEnsemble', data: pd.DataFrame) -> float:
        """Evaluate ensemble performance on data"""
        if len(data) < 100:
            return 0.0
        
        # Simple performance evaluation
        signals_generated = 0
        successful_signals = 0
        total_return = 0
        
        # Sliding window evaluation
        window_size = 50
        step_size = 10
        
        for i in range(window_size, len(data) - step_size, step_size):
            window_data = data.iloc[i-window_size:i]
            
            try:
                # Generate ensemble signal
                signal = ensemble.generate_ensemble_signal(window_data)
                
                if signal and signal.signal_type != 'hold':
                    signals_generated += 1
                    
                    # Calculate forward return
                    current_price = data.iloc[i]['close']
                    future_price = data.iloc[min(i + step_size, len(data) - 1)]['close']
                    
                    if signal.signal_type == 'buy':
                        signal_return = (future_price - current_price) / current_price
                    else:  # sell
                        signal_return = (current_price - future_price) / current_price
                    
                    # Weight by signal strength and confidence
                    weighted_return = signal_return * signal.strength * signal.confidence
                    total_return += weighted_return
                    
                    if signal_return > 0:
                        successful_signals += 1
                        
            except Exception as e:
                self.logger.debug(f"Signal evaluation error: {e}")
                continue
        
        if signals_generated == 0:
            return 0.0
        
        # Calculate fitness metrics
        win_rate = successful_signals / signals_generated
        avg_return = total_return / signals_generated
        signal_frequency = signals_generated / (len(data) / step_size)
        
        # Combined fitness score
        fitness = (
            win_rate * 0.4 +  # Win rate importance
            min(avg_return * 10, 1.0) * 0.4 +  # Average return (capped)
            min(signal_frequency, 0.5) * 0.2  # Signal frequency (not too many)
        )
        
        return max(0, fitness)
    
    def _fallback_evaluation(self, parameters: Dict[str, Any]) -> Tuple[float]:
        """Fallback evaluation when ML models are not available"""
        # Simple evaluation based on parameter reasonableness
        score = 0.5
        
        # Check if ensemble weights sum to reasonable total
        total_weight = (
            parameters['weight_rsi_lstm'] +
            parameters['weight_pattern_cnn'] +
            parameters['weight_adaptive_thresholds'] +
            parameters['weight_technical_indicators'] +
            parameters['weight_price_action']
        )
        
        if 0.8 <= total_weight <= 1.2:
            score += 0.2
        
        # Check consensus parameters
        if 0.5 <= parameters['min_consensus_ratio'] <= 0.7:
            score += 0.1
            
        if 0.4 <= parameters['min_confidence'] <= 0.6:
            score += 0.1
        
        return (score,)


class EnsembleOptimizer(GeneticOptimizer):
    """
    Genetic optimizer for ensemble weights and parameters
    """
    
    def __init__(self, indicator_ensemble: 'IndicatorEnsemble', **kwargs):
        super().__init__(**kwargs)
        self.ensemble = indicator_ensemble
        
        # Setup ensemble-specific parameters
        self._setup_ensemble_parameters()
    
    def _setup_ensemble_parameters(self):
        """Setup parameters for ensemble optimization"""
        
        # Individual indicator weights
        for indicator_name in self.ensemble.indicators.keys():
            self.add_parameter(f'weight_{indicator_name}', 0.01, 0.30, 'float')
        
        # Ensemble consensus parameters
        self.add_parameter('min_consensus_ratio', 0.3, 0.8, 'float')
        self.add_parameter('min_confidence', 0.2, 0.8, 'float')
        
        # Performance window for adaptive weights
        self.add_parameter('performance_window', 20, 200, 'int')
    
    def _evaluate_individual(self, individual: List[float]) -> Tuple[float]:
        """Evaluate ensemble configuration"""
        try:
            parameters = self._decode_individual(individual)
            
            # Update ensemble weights
            total_weight = 0
            for indicator_name in self.ensemble.indicators.keys():
                weight_param = f'weight_{indicator_name}'
                if weight_param in parameters:
                    self.ensemble.indicators[indicator_name].weight = parameters[weight_param]
                    total_weight += parameters[weight_param]
            
            # Normalize weights
            if total_weight > 0:
                for indicator in self.ensemble.indicators.values():
                    indicator.weight /= total_weight
            
            # Update ensemble config
            self.ensemble.config.min_consensus_ratio = parameters['min_consensus_ratio']
            self.ensemble.config.min_confidence = parameters['min_confidence']
            
            if 'performance_window' in parameters:
                self.ensemble.config.performance_window = int(parameters['performance_window'])
            
            # Evaluate ensemble performance
            fitness = self._evaluate_ensemble_on_data(self.train_data)
            
            return (fitness,)
            
        except Exception as e:
            self.logger.debug(f"Ensemble evaluation error: {e}")
            return (0.0,)
    
    def _evaluate_ensemble_on_data(self, data: pd.DataFrame) -> float:
        """Evaluate ensemble performance on historical data"""
        if len(data) < 50:
            return 0.0
        
        signals_generated = []
        returns = []
        
        # Evaluate in chunks
        chunk_size = 25
        for i in range(chunk_size, len(data) - 5, 5):
            chunk_data = data.iloc[i-chunk_size:i]
            
            try:
                signal = self.ensemble.generate_ensemble_signal(chunk_data)
                
                if signal and signal.signal_type != 'hold':
                    signals_generated.append(signal)
                    
                    # Calculate return
                    current_price = data.iloc[i]['close']
                    future_price = data.iloc[i + 5]['close']
                    
                    if signal.signal_type == 'buy':
                        ret = (future_price - current_price) / current_price
                    else:
                        ret = (current_price - future_price) / current_price
                    
                    # Weight by signal strength
                    weighted_return = ret * signal.strength * signal.confidence
                    returns.append(weighted_return)
                    
            except Exception:
                continue
        
        if len(returns) == 0:
            return 0.0
        
        # Calculate performance metrics
        avg_return = np.mean(returns)
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        sharpe = avg_return / (np.std(returns) + 1e-8)
        
        # Penalize too many or too few signals
        signal_frequency = len(signals_generated) / (len(data) / 5)
        frequency_penalty = abs(signal_frequency - 0.2)  # Target ~20% signal frequency
        
        fitness = (
            avg_return * 5 +  # Return importance
            win_rate * 0.3 +  # Win rate
            sharpe * 0.2 -    # Risk-adjusted return
            frequency_penalty * 0.1  # Signal frequency penalty
        )
        
        return max(0, fitness)


# Example usage functions
def optimize_rsi_strategy(data: pd.DataFrame) -> OptimizationResult:
    """Example: Optimize RSI-based strategy"""
    
    class SimpleRSIStrategy:
        def __init__(self, rsi_period=14, rsi_overbought=70, rsi_oversold=30, **kwargs):
            self.rsi_period = int(rsi_period)
            self.rsi_overbought = rsi_overbought
            self.rsi_oversold = rsi_oversold
        
        def generate_signals(self, data):
            # Simple RSI strategy
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            signals = pd.Series(0, index=data.index)
            signals[rsi < self.rsi_oversold] = 1  # Buy
            signals[rsi > self.rsi_overbought] = -1  # Sell
            
            return signals
    
    # Create optimizer
    optimizer = IndicatorOptimizer(SimpleRSIStrategy, population_size=50, generations=20)
    
    # Define evaluation function
    def evaluate_rsi(params, data):
        strategy = SimpleRSIStrategy(**params)
        signals = strategy.generate_signals(data)
        returns = signals.shift(1) * data['close'].pct_change()
        return returns.mean() / returns.std() if returns.std() > 0 else 0
    
    # Run optimization
    result = optimizer.optimize(evaluate_rsi, data)
    
    return result


def optimize_individual_indicator_intelligence(data: pd.DataFrame) -> OptimizationResult:
    """
    Example: Optimize Individual Indicator Intelligence system
    
    Args:
        data: Historical market data for optimization
        
    Returns:
        OptimizationResult with optimized ML model and ensemble parameters
    """
    print("🧬 Optimizing Individual Indicator Intelligence System")
    print("=" * 60)
    
    # Create optimizer
    optimizer = IndividualIndicatorOptimizer(
        population_size=30,  # Smaller population for complex optimization
        generations=20,      # Fewer generations due to complexity
        n_jobs=2            # Limit parallel processes
    )
    
    # Define evaluation function
    def evaluate_ml_ensemble(params, data):
        """Evaluation function for ML ensemble"""
        try:
            if not ML_MODELS_AVAILABLE:
                # Simple fallback scoring
                return 0.5
            
            # Create ensemble with parameters
            ensemble_config = EnsembleConfig(
                min_consensus_ratio=params.get('min_consensus_ratio', 0.6),
                min_confidence=params.get('min_confidence', 0.5),
                indicator_weights={
                    'rsi_lstm': params.get('weight_rsi_lstm', 0.15),
                    'pattern_cnn': params.get('weight_pattern_cnn', 0.15),
                    'adaptive_thresholds': params.get('weight_adaptive_thresholds', 0.10)
                }
            )
            
            ensemble = IndicatorEnsemble(ensemble_config)
            
            # Simplified evaluation for example
            signals_count = 0
            returns = []
            
            # Sliding window evaluation
            window_size = 50
            for i in range(window_size, len(data) - 5, 10):
                window_data = data.iloc[i-window_size:i]
                
                # Simulate signal generation (simplified)
                if len(window_data) >= window_size:
                    # Simple momentum signal for demonstration
                    momentum = window_data['close'].iloc[-1] / window_data['close'].iloc[-10] - 1
                    
                    if abs(momentum) > 0.02:  # 2% threshold
                        signals_count += 1
                        
                        # Forward return
                        current_price = data.iloc[i]['close']
                        future_price = data.iloc[i + 5]['close']
                        ret = (future_price - current_price) / current_price
                        
                        if momentum > 0:
                            returns.append(ret)
                        else:
                            returns.append(-ret)
            
            if len(returns) == 0:
                return 0.0
            
            # Calculate performance metrics
            avg_return = np.mean(returns)
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
            sharpe = avg_return / (np.std(returns) + 1e-8)
            
            return avg_return * 2 + win_rate * 0.5 + sharpe * 0.3
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0.0
    
    # Run optimization
    print(f"Optimizing with {len(data)} data points...")
    result = optimizer.optimize(evaluate_ml_ensemble, data, verbose=True)
    
    print(f"\n🎯 Individual Indicator Intelligence Optimization Results:")
    print(f"Best Fitness: {result.best_fitness:.4f}")
    print(f"Best Parameters:")
    for param, value in result.best_parameters.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.4f}")
        else:
            print(f"  {param}: {value}")
    
    print(f"Optimization Time: {result.optimization_time:.1f}s")
    print(f"Generations: {result.total_generations}")
    
    return result


if __name__ == "__main__":
    # Example usage
    print("🧬 Genetic Algorithm Optimizer for Trading Strategies")
    print("=" * 60)
    
    # Generate sample data
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
    
    sample_data = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
        'low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
        'volume': np.random.uniform(1000, 10000, len(dates))
    }, index=dates)
    
    print(f"Sample data shape: {sample_data.shape}")
    
    # Run optimization
    result = optimize_rsi_strategy(sample_data)
    
    print(f"\n🎯 Optimization Results:")
    print(f"Best Fitness: {result.best_fitness:.4f}")
    print(f"Best Parameters: {result.best_parameters}")
    print(f"Generations: {result.total_generations}")
    print(f"Time: {result.optimization_time:.1f}s")
    print(f"Validation Fitness: {result.validation_metrics.get('validation_fitness', 0):.4f}")