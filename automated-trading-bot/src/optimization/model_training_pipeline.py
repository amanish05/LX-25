"""
Model Training Pipeline for Automated Trading Bot
Trains ML models using historical data and generates performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('src')

from indicators.price_action_composite import PriceActionComposite
from indicators.advanced_confirmation import AdvancedConfirmationSystem
from indicators.signal_validator import SignalValidator
from indicators.rsi_advanced import AdvancedRSI
from indicators.oscillator_matrix import OscillatorMatrix


class ModelTrainingPipeline:
    """
    Trains ML models for signal prediction and generates performance metrics
    """
    
    def __init__(self, config_path: str = 'config/price_action_fine_tuned.json'):
        """Initialize training pipeline with configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize indicators
        self.price_action = PriceActionComposite(
            weights=self.config['price_action']['weights'],
            min_signal_strength=self.config['price_action']['min_strength']
        )
        self.confirmation_system = AdvancedConfirmationSystem()
        self.signal_validator = SignalValidator()
        self.rsi_advanced = AdvancedRSI()
        self.oscillator_matrix = OscillatorMatrix()
        
        # Initialize models
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        }
        
        # Scaler for feature normalization
        self.scaler = StandardScaler()
        
        # Training results
        self.training_results = {}
        self.feature_importance = {}
        
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features from price data"""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Technical indicators
        features['rsi'] = self.rsi_advanced.calculate(data['close'])
        oscillator_data = self.oscillator_matrix.calculate_all_oscillators(data)
        features['oscillator_score'] = oscillator_data['composite_score']
        
        # Price action features
        pa_data = self.price_action.calculate(data)
        if len(pa_data) > 0:
            features['pa_signal'] = pa_data['signal']
            features['pa_strength'] = pa_data['signal_strength']
            features['pa_confidence'] = pa_data['confidence'].map({'low': 0, 'medium': 0.5, 'high': 1})
        else:
            features['pa_signal'] = 0
            features['pa_strength'] = 0
            features['pa_confidence'] = 0
        
        # Market microstructure
        features['spread'] = (data['high'] - data['low']) / data['close']
        features['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Volatility features
        features['volatility_20'] = data['close'].pct_change().rolling(20).std()
        features['volatility_ratio'] = features['volatility_20'] / features['volatility_20'].rolling(60).mean()
        
        # Trend features
        features['sma_20'] = data['close'].rolling(20).mean()
        features['sma_50'] = data['close'].rolling(50).mean()
        features['trend_strength'] = (features['sma_20'] - features['sma_50']) / features['sma_50']
        
        # Remove NaN values
        features = features.dropna()
        
        return features
    
    def create_labels(self, data: pd.DataFrame, features: pd.DataFrame, 
                     lookahead: int = 5, profit_threshold: float = 0.005) -> pd.Series:
        """
        Create labels for training
        1: Profitable trade, 0: Non-profitable trade
        """
        # Calculate future returns
        future_returns = data['close'].shift(-lookahead) / data['close'] - 1
        
        # Create labels based on profit threshold
        labels = (future_returns > profit_threshold).astype(int)
        
        # Align with features
        labels = labels.loc[features.index]
        
        return labels.dropna()
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Train all models and evaluate performance"""
        results = {}
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train_scaled)
            y_val_pred = model.predict(X_val_scaled)
            
            # Calculate metrics
            train_metrics = {
                'accuracy': accuracy_score(y_train, y_train_pred),
                'precision': precision_score(y_train, y_train_pred, zero_division=0),
                'recall': recall_score(y_train, y_train_pred, zero_division=0),
                'f1': f1_score(y_train, y_train_pred, zero_division=0)
            }
            
            val_metrics = {
                'accuracy': accuracy_score(y_val, y_val_pred),
                'precision': precision_score(y_val, y_val_pred, zero_division=0),
                'recall': recall_score(y_val, y_val_pred, zero_division=0),
                'f1': f1_score(y_val, y_val_pred, zero_division=0)
            }
            
            # Calculate feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = dict(zip(
                    X_train.columns, 
                    model.feature_importances_
                ))
            
            results[model_name] = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'predictions': {
                    'train': y_train_pred,
                    'val': y_val_pred
                }
            }
            
            print(f"{model_name} - Val Accuracy: {val_metrics['accuracy']:.3f}, "
                  f"Precision: {val_metrics['precision']:.3f}, "
                  f"Recall: {val_metrics['recall']:.3f}")
        
        return results
    
    def generate_trading_signals(self, data: pd.DataFrame, model_name: str = 'random_forest') -> pd.DataFrame:
        """Generate trading signals using trained model"""
        # Generate features
        features = self.generate_features(data)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get model predictions
        model = self.models[model_name]
        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)[:, 1]
        
        # Create signals dataframe
        signals = pd.DataFrame(index=features.index)
        signals['ml_signal'] = predictions
        signals['ml_probability'] = probabilities
        
        # Combine with price action signals
        pa_data = self.price_action.calculate(data)
        if len(pa_data) > 0 and len(signals) > 0:
            # Find overlapping indices
            common_idx = signals.index.intersection(pa_data.index)
            signals.loc[common_idx, 'pa_signal'] = pa_data.loc[common_idx, 'signal']
            signals.loc[common_idx, 'pa_strength'] = pa_data.loc[common_idx, 'signal_strength']
        else:
            signals['pa_signal'] = 0
            signals['pa_strength'] = 0
        
        # Create composite signal
        signals['composite_signal'] = (
            (signals['ml_probability'] > 0.6) & 
            (signals['pa_signal'] == 1)
        ).astype(int)
        
        return signals
    
    def backtest_strategy(self, data: pd.DataFrame, signals: pd.DataFrame,
                         initial_capital: float = 100000) -> Dict:
        """Backtest trading strategy with ML signals"""
        results = []
        position = 0
        capital = initial_capital
        
        for i in range(len(signals)):
            date = signals.index[i]
            
            if signals['composite_signal'].iloc[i] == 1 and position == 0:
                # Enter position
                position = capital * 0.02  # 2% position size
                entry_price = data.loc[date, 'close']
                entry_date = date
                
            elif position > 0:
                current_price = data.loc[date, 'close']
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Exit conditions
                if pnl_pct > 0.02 or pnl_pct < -0.01:  # 2% profit or 1% loss
                    pnl = position * pnl_pct
                    capital += pnl
                    
                    results.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'capital': capital
                    })
                    
                    position = 0
        
        # Calculate performance metrics
        if results:
            results_df = pd.DataFrame(results)
            
            metrics = {
                'total_trades': len(results_df),
                'winning_trades': len(results_df[results_df['pnl'] > 0]),
                'losing_trades': len(results_df[results_df['pnl'] < 0]),
                'win_rate': len(results_df[results_df['pnl'] > 0]) / len(results_df),
                'total_pnl': results_df['pnl'].sum(),
                'total_return': (capital - initial_capital) / initial_capital,
                'avg_win': results_df[results_df['pnl'] > 0]['pnl_pct'].mean() if len(results_df[results_df['pnl'] > 0]) > 0 else 0,
                'avg_loss': results_df[results_df['pnl'] < 0]['pnl_pct'].mean() if len(results_df[results_df['pnl'] < 0]) > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(results_df),
                'sharpe_ratio': self._calculate_sharpe_ratio(results_df)
            }
            
            return {'metrics': metrics, 'trades': results_df}
        
        return {'metrics': {}, 'trades': pd.DataFrame()}
    
    def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        cumulative_returns = (1 + trades_df['pnl_pct']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, trades_df: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        returns = trades_df['pnl_pct']
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        if returns.std() > 0:
            return np.sqrt(252) * excess_returns.mean() / returns.std()
        return 0
    
    def save_models(self, path: str = 'models/'):
        """Save trained models and scaler"""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            with open(f'{path}{model_name}_model.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # Save scaler
        with open(f'{path}scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature importance
        with open(f'{path}feature_importance.json', 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
        
        print(f"Models saved to {path}")
    
    def generate_performance_report(self, train_results: Dict, test_results: Dict) -> Dict:
        """Generate comprehensive performance report"""
        report = {
            'training_date': datetime.now().isoformat(),
            'model_performance': {},
            'feature_importance': self.feature_importance,
            'backtest_results': {
                'train': train_results,
                'test': test_results
            }
        }
        
        # Add model performance metrics
        for model_name, results in self.training_results.items():
            report['model_performance'][model_name] = {
                'train_metrics': results['train_metrics'],
                'validation_metrics': results['val_metrics']
            }
        
        # Save report
        with open('reports/model_training_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


def main():
    """Run model training pipeline"""
    print("="*60)
    print("MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Initialize pipeline
    pipeline = ModelTrainingPipeline()
    
    # Generate synthetic training data (replace with real data)
    print("\n1. Generating training data...")
    dates = pd.date_range(end=datetime.now(), periods=2000, freq='5min')
    
    # Create realistic market data
    np.random.seed(42)
    prices = []
    price = 20000
    
    for i in range(len(dates)):
        # Add trend and volatility
        trend = np.sin(i / 100) * 0.001
        volatility = 0.002 * (1 + 0.5 * np.sin(i / 50))
        change = np.random.normal(trend, volatility)
        price *= (1 + change)
        prices.append(price)
    
    data = pd.DataFrame({
        'open': prices + np.random.randn(len(prices)) * 10,
        'high': prices + np.abs(np.random.randn(len(prices))) * 20,
        'low': prices - np.abs(np.random.randn(len(prices))) * 20,
        'close': prices,
        'volume': np.random.randint(50000, 200000, len(prices))
    }, index=dates)
    
    # Ensure OHLC consistency
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    print(f"Generated {len(data)} data points")
    
    # Generate features
    print("\n2. Generating features...")
    features = pipeline.generate_features(data)
    labels = pipeline.create_labels(data, features)
    
    # Align features and labels
    common_index = features.index.intersection(labels.index)
    features = features.loc[common_index]
    labels = labels.loc[common_index]
    
    print(f"Generated {len(features)} feature vectors with {features.shape[1]} features")
    
    # Split data
    print("\n3. Splitting data...")
    split_idx = int(len(features) * 0.7)
    X_train = features.iloc[:split_idx]
    y_train = labels.iloc[:split_idx]
    X_val = features.iloc[split_idx:]
    y_val = labels.iloc[split_idx:]
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Train models
    print("\n4. Training models...")
    pipeline.training_results = pipeline.train_models(X_train, y_train, X_val, y_val)
    
    # Generate signals for backtesting
    print("\n5. Generating trading signals...")
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    train_signals = pipeline.generate_trading_signals(train_data)
    test_signals = pipeline.generate_trading_signals(test_data)
    
    # Backtest strategy
    print("\n6. Backtesting strategy...")
    train_backtest = pipeline.backtest_strategy(train_data, train_signals)
    test_backtest = pipeline.backtest_strategy(test_data, test_signals)
    
    print("\nTraining Period Performance:")
    if train_backtest['metrics']:
        for metric, value in train_backtest['metrics'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
    
    print("\nTest Period Performance:")
    if test_backtest['metrics']:
        for metric, value in test_backtest['metrics'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
    
    # Save models
    print("\n7. Saving models...")
    pipeline.save_models()
    
    # Generate report
    print("\n8. Generating performance report...")
    report = pipeline.generate_performance_report(
        train_backtest['metrics'],
        test_backtest['metrics']
    )
    
    print("\nTraining complete! Report saved to reports/model_training_report.json")
    
    return pipeline, report


if __name__ == "__main__":
    pipeline, report = main()