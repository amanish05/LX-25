"""
LSTM Neural Network for RSI Pattern Recognition
Uses LSTM to learn RSI patterns and predict future RSI movements and divergences
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import pickle
from pathlib import Path
import json

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # Define dummy classes for type hints when TensorFlow is not available
    class Model:
        pass
    class Sequential:
        pass
    print("TensorFlow not available. RSI LSTM model will use fallback implementation.")

from dataclasses import dataclass


@dataclass
class RSIPattern:
    """RSI pattern detection result"""
    pattern_type: str  # 'bullish_divergence', 'bearish_divergence', 'oversold_reversal', 'overbought_reversal'
    confidence: float  # 0-1
    rsi_prediction: float  # Predicted RSI value
    price_direction: int  # -1 (down), 0 (sideways), 1 (up)
    strength: float  # Signal strength 0-1


@dataclass 
class RSIModelConfig:
    """Configuration for RSI LSTM model"""
    sequence_length: int = 20  # Number of RSI values to look back
    prediction_horizon: int = 5  # Number of periods to predict ahead
    lstm_units: int = 50
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2


class RSILSTMModel:
    """
    LSTM model for RSI pattern recognition and prediction
    
    Features:
    - Learns RSI sequences to predict future RSI values
    - Detects bullish/bearish divergences
    - Identifies oversold/overbought reversal patterns
    - Provides confidence scores for predictions
    """
    
    def __init__(self, config: Optional[RSIModelConfig] = None):
        """Initialize RSI LSTM model"""
        self.config = config or RSIModelConfig()
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.model = None
        self.rsi_scaler = MinMaxScaler()
        self.price_scaler = MinMaxScaler()
        self.is_trained = False
        
        # Training history
        self.training_history = []
        self.performance_metrics = {}
        
        # Pattern detection thresholds
        self.divergence_threshold = 0.7
        self.reversal_threshold = 0.6
        
        if not TF_AVAILABLE:
            self.logger.warning("TensorFlow not available. Using fallback RSI analysis.")
    
    def _build_model(self) -> Model:
        """Build LSTM model architecture"""
        if not TF_AVAILABLE:
            return None
            
        model = Sequential([
            Input(shape=(self.config.sequence_length, 2)),  # RSI + Price features
            
            LSTM(self.config.lstm_units, return_sequences=True),
            BatchNormalization(),
            Dropout(self.config.dropout_rate),
            
            LSTM(self.config.lstm_units // 2, return_sequences=False),
            BatchNormalization(), 
            Dropout(self.config.dropout_rate),
            
            Dense(32, activation='relu'),
            Dropout(self.config.dropout_rate),
            
            Dense(16, activation='relu'),
            
            # Multi-output: RSI prediction + pattern classification
            Dense(4, activation='linear', name='rsi_prediction'),  # RSI values
            Dense(4, activation='softmax', name='pattern_classification')  # Pattern types
        ])
        
        # Compile with multiple outputs
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss={
                'rsi_prediction': 'mse',
                'pattern_classification': 'categorical_crossentropy'
            },
            loss_weights={
                'rsi_prediction': 0.7,
                'pattern_classification': 0.3
            },
            metrics={
                'rsi_prediction': 'mae',
                'pattern_classification': 'accuracy'
            }
        )
        
        return model
    
    def prepare_training_data(self, rsi_data: pd.Series, price_data: pd.Series) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare training data from RSI and price series
        
        Args:
            rsi_data: RSI values time series
            price_data: Price data time series
            
        Returns:
            Tuple of (X, y) where X is input sequences and y is targets
        """
        # Align data
        common_index = rsi_data.index.intersection(price_data.index)
        rsi_aligned = rsi_data.loc[common_index].values
        price_aligned = price_data.loc[common_index].values
        
        # Scale data
        rsi_scaled = self.rsi_scaler.fit_transform(rsi_aligned.reshape(-1, 1)).flatten()
        price_scaled = self.price_scaler.fit_transform(price_aligned.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y_rsi, y_patterns = [], [], []
        
        for i in range(self.config.sequence_length, len(rsi_scaled) - self.config.prediction_horizon):
            # Input sequence: RSI + price features
            rsi_seq = rsi_scaled[i-self.config.sequence_length:i]
            price_seq = price_scaled[i-self.config.sequence_length:i]
            X.append(np.column_stack([rsi_seq, price_seq]))
            
            # Target RSI values
            future_rsi = rsi_scaled[i:i+self.config.prediction_horizon]
            y_rsi.append(future_rsi)
            
            # Pattern classification
            pattern = self._identify_pattern(
                rsi_aligned[i-self.config.sequence_length:i],
                price_aligned[i-self.config.sequence_length:i],
                rsi_aligned[i:i+self.config.prediction_horizon],
                price_aligned[i:i+self.config.prediction_horizon]
            )
            y_patterns.append(pattern)
        
        X = np.array(X)
        y_targets = {
            'rsi_prediction': np.array(y_rsi),
            'pattern_classification': np.array(y_patterns)
        }
        
        return X, y_targets
    
    def _identify_pattern(self, historical_rsi: np.ndarray, historical_price: np.ndarray,
                         future_rsi: np.ndarray, future_price: np.ndarray) -> np.ndarray:
        """Identify RSI pattern for training labels"""
        pattern = np.zeros(4)  # [bullish_div, bearish_div, oversold_reversal, overbought_reversal]
        
        # Check for divergences
        if self._check_bullish_divergence(historical_rsi, historical_price, future_price):
            pattern[0] = 1
        elif self._check_bearish_divergence(historical_rsi, historical_price, future_price):
            pattern[1] = 1
        # Check for reversal patterns
        elif historical_rsi[-1] < 30 and np.mean(future_price) > historical_price[-1]:
            pattern[2] = 1  # Oversold reversal
        elif historical_rsi[-1] > 70 and np.mean(future_price) < historical_price[-1]:
            pattern[3] = 1  # Overbought reversal
        
        return pattern
    
    def _check_bullish_divergence(self, rsi: np.ndarray, price: np.ndarray, future_price: np.ndarray) -> bool:
        """Check for bullish divergence pattern"""
        # Price making lower lows, RSI making higher lows
        if len(rsi) < 10:
            return False
        
        price_trend = np.polyfit(range(len(price)), price, 1)[0]
        rsi_trend = np.polyfit(range(len(rsi)), rsi, 1)[0]
        
        # Bullish divergence: price declining, RSI rising, future price up
        return (price_trend < -0.001 and rsi_trend > 0.1 and 
                np.mean(future_price) > price[-1])
    
    def _check_bearish_divergence(self, rsi: np.ndarray, price: np.ndarray, future_price: np.ndarray) -> bool:
        """Check for bearish divergence pattern"""
        # Price making higher highs, RSI making lower highs
        if len(rsi) < 10:
            return False
        
        price_trend = np.polyfit(range(len(price)), price, 1)[0]
        rsi_trend = np.polyfit(range(len(rsi)), rsi, 1)[0]
        
        # Bearish divergence: price rising, RSI falling, future price down
        return (price_trend > 0.001 and rsi_trend < -0.1 and 
                np.mean(future_price) < price[-1])
    
    def train(self, rsi_data: pd.Series, price_data: pd.Series, 
             validation_data: Optional[Tuple[pd.Series, pd.Series]] = None) -> Dict[str, Any]:
        """
        Train the LSTM model
        
        Args:
            rsi_data: Training RSI data
            price_data: Training price data
            validation_data: Optional validation data tuple (rsi, price)
            
        Returns:
            Training results and metrics
        """
        if not TF_AVAILABLE:
            return self._fallback_training(rsi_data, price_data)
        
        self.logger.info(f"Training RSI LSTM model with {len(rsi_data)} samples")
        
        # Prepare training data
        X_train, y_train = self.prepare_training_data(rsi_data, price_data)
        
        # Prepare validation data if provided
        X_val, y_val = None, None
        if validation_data:
            X_val, y_val = self.prepare_training_data(validation_data[0], validation_data[1])
        
        # Build model
        self.model = self._build_model()
        
        # Check if model was built successfully
        if self.model is None:
            self.logger.error("Failed to build LSTM model - TensorFlow may not be available")
            return self._fallback_training(rsi_data, price_data)
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        validation_data_tuple = (X_val, y_val) if validation_data else None
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=validation_data_tuple,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        self.training_history.append(history.history)
        
        # Calculate performance metrics
        train_predictions = self.model.predict(X_train)
        rsi_mae = np.mean(np.abs(train_predictions[0] - y_train['rsi_prediction']))
        pattern_accuracy = accuracy_score(
            np.argmax(y_train['pattern_classification'], axis=1),
            np.argmax(train_predictions[1], axis=1)
        )
        
        self.performance_metrics = {
            'rsi_mae': float(rsi_mae),
            'pattern_accuracy': float(pattern_accuracy),
            'final_loss': float(history.history['loss'][-1]),
            'epochs_trained': len(history.history['loss'])
        }
        
        self.logger.info(f"Training completed. RSI MAE: {rsi_mae:.4f}, Pattern Accuracy: {pattern_accuracy:.4f}")
        
        return {
            'metrics': self.performance_metrics,
            'history': history.history
        }
    
    def predict_pattern(self, rsi_sequence: pd.Series, price_sequence: pd.Series) -> RSIPattern:
        """
        Predict RSI pattern from recent data
        
        Args:
            rsi_sequence: Recent RSI values (at least sequence_length)
            price_sequence: Recent price values
            
        Returns:
            RSIPattern with prediction results
        """
        if not self.is_trained:
            return self._fallback_prediction(rsi_sequence, price_sequence)
        
        # Prepare input data
        if len(rsi_sequence) < self.config.sequence_length:
            self.logger.warning(f"Insufficient data for prediction. Need {self.config.sequence_length}, got {len(rsi_sequence)}")
            return RSIPattern("insufficient_data", 0.0, rsi_sequence.iloc[-1], 0, 0.0)
        
        # Get most recent sequence
        rsi_recent = rsi_sequence.iloc[-self.config.sequence_length:].values
        price_recent = price_sequence.iloc[-self.config.sequence_length:].values
        
        # Scale input
        rsi_scaled = self.rsi_scaler.transform(rsi_recent.reshape(-1, 1)).flatten()
        price_scaled = self.price_scaler.transform(price_recent.reshape(-1, 1)).flatten()
        
        # Create input sequence
        X = np.column_stack([rsi_scaled, price_scaled]).reshape(1, self.config.sequence_length, 2)
        
        # Make prediction
        if TF_AVAILABLE and self.model:
            predictions = self.model.predict(X, verbose=0)
            rsi_pred = predictions[0][0]  # Future RSI values
            pattern_probs = predictions[1][0]  # Pattern probabilities
        else:
            return self._fallback_prediction(rsi_sequence, price_sequence)
        
        # Interpret results
        pattern_types = ['bullish_divergence', 'bearish_divergence', 'oversold_reversal', 'overbought_reversal']
        max_prob_idx = np.argmax(pattern_probs)
        pattern_type = pattern_types[max_prob_idx]
        confidence = float(pattern_probs[max_prob_idx])
        
        # Predict price direction
        predicted_rsi = float(self.rsi_scaler.inverse_transform(rsi_pred.reshape(-1, 1)).mean())
        current_rsi = rsi_sequence.iloc[-1]
        
        if predicted_rsi > current_rsi + 2:
            price_direction = 1
        elif predicted_rsi < current_rsi - 2:
            price_direction = -1
        else:
            price_direction = 0
        
        # Calculate overall signal strength
        strength = confidence * (0.7 if confidence > self.divergence_threshold else 0.4)
        
        return RSIPattern(
            pattern_type=pattern_type,
            confidence=confidence,
            rsi_prediction=predicted_rsi,
            price_direction=price_direction,
            strength=strength
        )
    
    def _fallback_prediction(self, rsi_sequence: pd.Series, price_sequence: pd.Series) -> RSIPattern:
        """Fallback prediction when TensorFlow is not available"""
        current_rsi = rsi_sequence.iloc[-1]
        
        # Simple rule-based pattern detection
        if current_rsi < 30:
            pattern_type = "oversold_reversal"
            confidence = 0.6
            price_direction = 1
        elif current_rsi > 70:
            pattern_type = "overbought_reversal"
            confidence = 0.6
            price_direction = -1
        else:
            pattern_type = "neutral"
            confidence = 0.3
            price_direction = 0
        
        return RSIPattern(
            pattern_type=pattern_type,
            confidence=confidence,
            rsi_prediction=current_rsi,
            price_direction=price_direction,
            strength=confidence * 0.5
        )
    
    def _fallback_training(self, rsi_data: pd.Series, price_data: pd.Series) -> Dict[str, Any]:
        """Fallback training when TensorFlow is not available"""
        self.logger.info("Using fallback training (rule-based RSI analysis)")
        self.is_trained = True
        
        # Calculate basic statistics
        oversold_count = (rsi_data < 30).sum()
        overbought_count = (rsi_data > 70).sum()
        
        self.performance_metrics = {
            'oversold_signals': int(oversold_count),
            'overbought_signals': int(overbought_count),
            'total_samples': len(rsi_data),
            'fallback_mode': True
        }
        
        return {'metrics': self.performance_metrics}
    
    def save_model(self, filepath: str):
        """Save trained model and components"""
        model_dir = Path(filepath).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model architecture and weights
        if TF_AVAILABLE and self.model:
            self.model.save(f"{filepath}_model.h5")
        
        # Save scalers and config
        with open(f"{filepath}_scalers.pkl", 'wb') as f:
            pickle.dump({
                'rsi_scaler': self.rsi_scaler,
                'price_scaler': self.price_scaler
            }, f)
        
        # Save configuration and metrics
        model_info = {
            'config': {
                'sequence_length': self.config.sequence_length,
                'prediction_horizon': self.config.prediction_horizon,
                'lstm_units': self.config.lstm_units,
                'dropout_rate': self.config.dropout_rate,
                'learning_rate': self.config.learning_rate
            },
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained,
            'tf_available': TF_AVAILABLE
        }
        
        with open(f"{filepath}_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        self.logger.info(f"RSI LSTM model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and components"""
        try:
            # Load model
            if TF_AVAILABLE and Path(f"{filepath}_model.h5").exists():
                self.model = tf.keras.models.load_model(f"{filepath}_model.h5")
            
            # Load scalers
            with open(f"{filepath}_scalers.pkl", 'rb') as f:
                scalers = pickle.load(f)
                self.rsi_scaler = scalers['rsi_scaler']
                self.price_scaler = scalers['price_scaler']
            
            # Load configuration
            with open(f"{filepath}_info.json", 'r') as f:
                model_info = json.load(f)
                self.performance_metrics = model_info['performance_metrics']
                self.is_trained = model_info['is_trained']
            
            self.logger.info(f"RSI LSTM model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary and performance metrics"""
        summary = {
            'model_type': 'RSI_LSTM',
            'is_trained': self.is_trained,
            'tensorflow_available': TF_AVAILABLE,
            'config': {
                'sequence_length': self.config.sequence_length,
                'prediction_horizon': self.config.prediction_horizon,
                'lstm_units': self.config.lstm_units
            },
            'performance': self.performance_metrics
        }
        
        if TF_AVAILABLE and self.model:
            summary['model_parameters'] = self.model.count_params()
        
        return summary


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range(end=datetime.now(), periods=1000, freq='5min')
    
    # Generate realistic RSI and price data
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.02, len(dates))))
    
    # Generate RSI that correlates with price patterns
    rsi_values = []
    for i in range(len(prices)):
        if i == 0:
            rsi_values.append(50)
        else:
            price_change = (prices[i] - prices[i-1]) / prices[i-1]
            rsi_momentum = 50 + price_change * 500 + np.random.normal(0, 5)
            rsi_values.append(max(0, min(100, rsi_momentum)))
    
    rsi_data = pd.Series(rsi_values, index=dates)
    price_data = pd.Series(prices, index=dates)
    
    # Initialize and train model
    model = RSILSTMModel()
    
    # Split data for training and validation
    split_idx = int(len(rsi_data) * 0.8)
    train_rsi = rsi_data.iloc[:split_idx]
    train_price = price_data.iloc[:split_idx]
    val_rsi = rsi_data.iloc[split_idx:]
    val_price = price_data.iloc[split_idx:]
    
    print("Training RSI LSTM Model...")
    results = model.train(train_rsi, train_price, (val_rsi, val_price))
    print(f"Training Results: {results['metrics']}")
    
    # Test prediction
    recent_rsi = rsi_data.tail(25)
    recent_price = price_data.tail(25)
    
    pattern = model.predict_pattern(recent_rsi, recent_price)
    print(f"\nPredicted Pattern: {pattern.pattern_type}")
    print(f"Confidence: {pattern.confidence:.3f}")
    print(f"Predicted RSI: {pattern.rsi_prediction:.2f}")
    print(f"Price Direction: {pattern.price_direction}")
    print(f"Signal Strength: {pattern.strength:.3f}")
    
    # Model summary
    summary = model.get_model_summary()
    print(f"\nModel Summary: {summary}")