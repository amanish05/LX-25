"""
Convolutional Neural Network for Chart Pattern Detection
Uses CNN to detect classic chart patterns like triangles, head & shoulders, flags, etc.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import pickle
from pathlib import Path
import json
import cv2
from scipy import ndimage
from skimage.transform import resize

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Pattern CNN model will use fallback implementation.")

from dataclasses import dataclass


@dataclass
class PatternDetection:
    """Chart pattern detection result"""
    pattern_type: str  # 'triangle', 'head_shoulders', 'flag', 'wedge', 'double_top', 'double_bottom', 'none'
    confidence: float  # 0-1
    breakout_direction: int  # -1 (down), 0 (sideways), 1 (up)
    target_price: Optional[float]  # Projected target price
    strength: float  # Signal strength 0-1
    pattern_coordinates: Optional[Dict[str, List[float]]]  # Key pattern points


@dataclass
class CNNModelConfig:
    """Configuration for Pattern CNN model"""
    image_size: Tuple[int, int] = (64, 64)  # Height, Width for pattern images
    lookback_periods: int = 50  # Number of periods to convert to image
    num_channels: int = 3  # RGB channels (price, volume, indicators)
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    augmentation: bool = True  # Data augmentation


class PatternCNNModel:
    """
    CNN model for chart pattern recognition
    
    Features:
    - Converts OHLCV data to visual patterns
    - Detects classic chart patterns using CNN
    - Provides pattern confidence and breakout direction
    - Calculates target prices based on pattern geometry
    """
    
    def __init__(self, config: Optional[CNNModelConfig] = None):
        """Initialize Pattern CNN model"""
        self.config = config or CNNModelConfig()
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Pattern definitions
        self.pattern_types = [
            'triangle_ascending', 'triangle_descending', 'triangle_symmetrical',
            'head_shoulders', 'inverse_head_shoulders',
            'flag_bullish', 'flag_bearish',
            'wedge_rising', 'wedge_falling',
            'double_top', 'double_bottom',
            'rectangle', 'channel_up', 'channel_down',
            'none'
        ]
        
        # Training history
        self.training_history = []
        self.performance_metrics = {}
        
        if not TF_AVAILABLE:
            self.logger.warning("TensorFlow not available. Using fallback pattern analysis.")
    
    def _build_model(self) -> Model:
        """Build CNN model architecture"""
        if not TF_AVAILABLE:
            return None
        
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', 
                   input_shape=(*self.config.image_size, self.config.num_channels)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fully connected layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(256, activation='relu'),
            Dropout(0.5),
            
            # Multi-output
            Dense(len(self.pattern_types), activation='softmax', name='pattern_classification'),
            Dense(3, activation='linear', name='breakout_prediction')  # Direction + confidence + target
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss={
                'pattern_classification': 'sparse_categorical_crossentropy',
                'breakout_prediction': 'mse'
            },
            loss_weights={
                'pattern_classification': 0.7,
                'breakout_prediction': 0.3
            },
            metrics={
                'pattern_classification': 'accuracy',
                'breakout_prediction': 'mae'
            }
        )
        
        return model
    
    def _convert_data_to_image(self, ohlcv_data: pd.DataFrame, 
                              indicators: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Convert OHLCV data to visual pattern image
        
        Args:
            ohlcv_data: OHLC + Volume data
            indicators: Optional technical indicators
            
        Returns:
            3D array representing the pattern image
        """
        if len(ohlcv_data) < self.config.lookback_periods:
            # Pad with first values if insufficient data
            padding_size = self.config.lookback_periods - len(ohlcv_data)
            first_row = ohlcv_data.iloc[0:1]
            padding = pd.concat([first_row] * padding_size, ignore_index=True)
            ohlcv_data = pd.concat([padding, ohlcv_data], ignore_index=True)
        
        # Take last N periods
        data = ohlcv_data.tail(self.config.lookback_periods).copy()
        
        # Normalize prices relative to the range
        price_min = data[['open', 'high', 'low', 'close']].min().min()
        price_max = data[['open', 'high', 'low', 'close']].max().max()
        price_range = price_max - price_min if price_max > price_min else 1
        
        # Create image channels
        image = np.zeros((*self.config.image_size, self.config.num_channels))
        
        # Channel 1: Price pattern (candlesticks)
        for i, (_, row) in enumerate(data.iterrows()):
            x = int((i / len(data)) * self.config.image_size[1])
            
            # Normalize prices to image coordinates
            high_y = int(((row['high'] - price_min) / price_range) * self.config.image_size[0])
            low_y = int(((row['low'] - price_min) / price_range) * self.config.image_size[0])
            open_y = int(((row['open'] - price_min) / price_range) * self.config.image_size[0])
            close_y = int(((row['close'] - price_min) / price_range) * self.config.image_size[0])
            
            # Flip Y coordinate (higher prices at top)
            high_y = self.config.image_size[0] - 1 - high_y
            low_y = self.config.image_size[0] - 1 - low_y
            open_y = self.config.image_size[0] - 1 - open_y
            close_y = self.config.image_size[0] - 1 - close_y
            
            # Draw candlestick
            if x < self.config.image_size[1]:
                # High-low line
                for y in range(min(high_y, low_y), max(high_y, low_y) + 1):
                    if 0 <= y < self.config.image_size[0]:
                        image[y, x, 0] = 0.5
                
                # Body
                body_color = 1.0 if row['close'] > row['open'] else 0.3
                for y in range(min(open_y, close_y), max(open_y, close_y) + 1):
                    if 0 <= y < self.config.image_size[0]:
                        image[y, x, 0] = body_color
        
        # Channel 2: Volume pattern
        if 'volume' in data.columns:
            volume_max = data['volume'].max() if data['volume'].max() > 0 else 1
            for i, (_, row) in enumerate(data.iterrows()):
                x = int((i / len(data)) * self.config.image_size[1])
                volume_height = int((row['volume'] / volume_max) * self.config.image_size[0] * 0.3)
                
                if x < self.config.image_size[1]:
                    for y in range(self.config.image_size[0] - volume_height, self.config.image_size[0]):
                        if 0 <= y < self.config.image_size[0]:
                            image[y, x, 1] = row['volume'] / volume_max
        
        # Channel 3: Additional indicators or moving averages
        if indicators is not None and len(indicators) > 0:
            # Use first indicator as third channel
            indicator_name = indicators.columns[0]
            indicator_data = indicators[indicator_name].tail(self.config.lookback_periods)
            
            if len(indicator_data) > 0:
                ind_min = indicator_data.min()
                ind_max = indicator_data.max()
                ind_range = ind_max - ind_min if ind_max > ind_min else 1
                
                for i, value in enumerate(indicator_data):
                    x = int((i / len(indicator_data)) * self.config.image_size[1])
                    y = int(((value - ind_min) / ind_range) * self.config.image_size[0])
                    y = self.config.image_size[0] - 1 - y  # Flip Y
                    
                    if 0 <= x < self.config.image_size[1] and 0 <= y < self.config.image_size[0]:
                        image[y, x, 2] = (value - ind_min) / ind_range
        else:
            # Use simple moving average as third channel
            sma_20 = data['close'].rolling(window=min(20, len(data))).mean()
            for i, value in enumerate(sma_20):
                if pd.notna(value):
                    x = int((i / len(data)) * self.config.image_size[1])
                    y = int(((value - price_min) / price_range) * self.config.image_size[0])
                    y = self.config.image_size[0] - 1 - y
                    
                    if 0 <= x < self.config.image_size[1] and 0 <= y < self.config.image_size[0]:
                        image[y, x, 2] = 0.7
        
        # Apply smoothing
        for channel in range(self.config.num_channels):
            image[:, :, channel] = ndimage.gaussian_filter(image[:, :, channel], sigma=0.5)
        
        return image
    
    def _identify_pattern_from_data(self, ohlcv_data: pd.DataFrame) -> Tuple[str, float, int]:
        """
        Identify pattern from OHLCV data for training labels
        
        Args:
            ohlcv_data: OHLC + Volume data
            
        Returns:
            Tuple of (pattern_name, confidence, breakout_direction)
        """
        if len(ohlcv_data) < 20:
            return 'none', 0.0, 0
        
        highs = ohlcv_data['high'].values
        lows = ohlcv_data['low'].values
        closes = ohlcv_data['close'].values
        
        # Simple pattern detection for training labels
        # Triangle patterns
        if self._detect_triangle_ascending(highs, lows):
            return 'triangle_ascending', 0.8, 1
        elif self._detect_triangle_descending(highs, lows):
            return 'triangle_descending', 0.8, -1
        elif self._detect_triangle_symmetrical(highs, lows):
            return 'triangle_symmetrical', 0.7, 0
        
        # Head and shoulders
        elif self._detect_head_shoulders(highs):
            return 'head_shoulders', 0.8, -1
        elif self._detect_inverse_head_shoulders(lows):
            return 'inverse_head_shoulders', 0.8, 1
        
        # Double patterns
        elif self._detect_double_top(highs):
            return 'double_top', 0.7, -1
        elif self._detect_double_bottom(lows):
            return 'double_bottom', 0.7, 1
        
        # Flags and channels
        elif self._detect_flag_pattern(highs, lows, closes):
            trend = 1 if closes[-1] > closes[0] else -1
            return f'flag_{"bullish" if trend > 0 else "bearish"}', 0.6, trend
        
        return 'none', 0.3, 0
    
    def _detect_triangle_ascending(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        """Detect ascending triangle pattern"""
        if len(highs) < 10:
            return False
        
        # Check if highs are relatively flat and lows are rising
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        
        high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
        low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
        
        return abs(high_trend) < 0.001 and low_trend > 0.001
    
    def _detect_triangle_descending(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        """Detect descending triangle pattern"""
        if len(highs) < 10:
            return False
        
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        
        high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
        low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
        
        return high_trend < -0.001 and abs(low_trend) < 0.001
    
    def _detect_triangle_symmetrical(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        """Detect symmetrical triangle pattern"""
        if len(highs) < 10:
            return False
        
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        
        high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
        low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
        
        return high_trend < -0.001 and low_trend > 0.001
    
    def _detect_head_shoulders(self, highs: np.ndarray) -> bool:
        """Detect head and shoulders pattern"""
        if len(highs) < 15:
            return False
        
        # Find peaks
        peaks = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1] and highs[i] > highs[i-2] and highs[i] > highs[i+2]:
                peaks.append((i, highs[i]))
        
        if len(peaks) >= 3:
            # Check if middle peak is highest (head)
            peaks.sort(key=lambda x: x[1], reverse=True)
            head = peaks[0]
            shoulders = [p for p in peaks[1:3]]
            
            # Head should be in the middle chronologically
            head_pos = head[0]
            shoulder_positions = [s[0] for s in shoulders]
            
            return min(shoulder_positions) < head_pos < max(shoulder_positions)
        
        return False
    
    def _detect_inverse_head_shoulders(self, lows: np.ndarray) -> bool:
        """Detect inverse head and shoulders pattern"""
        if len(lows) < 15:
            return False
        
        # Find troughs
        troughs = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1] and lows[i] < lows[i-2] and lows[i] < lows[i+2]:
                troughs.append((i, lows[i]))
        
        if len(troughs) >= 3:
            # Check if middle trough is lowest (head)
            troughs.sort(key=lambda x: x[1])
            head = troughs[0]
            shoulders = troughs[1:3]
            
            head_pos = head[0]
            shoulder_positions = [s[0] for s in shoulders]
            
            return min(shoulder_positions) < head_pos < max(shoulder_positions)
        
        return False
    
    def _detect_double_top(self, highs: np.ndarray) -> bool:
        """Detect double top pattern"""
        if len(highs) < 10:
            return False
        
        peaks = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peaks.append((i, highs[i]))
        
        if len(peaks) >= 2:
            # Check if two highest peaks are similar in height
            peaks.sort(key=lambda x: x[1], reverse=True)
            peak1, peak2 = peaks[0], peaks[1]
            height_diff = abs(peak1[1] - peak2[1]) / max(peak1[1], peak2[1])
            
            return height_diff < 0.02  # Within 2%
        
        return False
    
    def _detect_double_bottom(self, lows: np.ndarray) -> bool:
        """Detect double bottom pattern"""
        if len(lows) < 10:
            return False
        
        troughs = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                troughs.append((i, lows[i]))
        
        if len(troughs) >= 2:
            troughs.sort(key=lambda x: x[1])
            trough1, trough2 = troughs[0], troughs[1]
            height_diff = abs(trough1[1] - trough2[1]) / max(trough1[1], trough2[1])
            
            return height_diff < 0.02
        
        return False
    
    def _detect_flag_pattern(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> bool:
        """Detect flag pattern"""
        if len(closes) < 20:
            return False
        
        # Check for strong initial trend followed by consolidation
        first_half = closes[:len(closes)//2]
        second_half = closes[len(closes)//2:]
        
        first_trend = np.polyfit(range(len(first_half)), first_half, 1)[0]
        second_trend = np.polyfit(range(len(second_half)), second_half, 1)[0]
        
        # Strong trend followed by consolidation
        return abs(first_trend) > 0.005 and abs(second_trend) < 0.002
    
    def prepare_training_data(self, market_data: List[pd.DataFrame],
                            indicators_data: Optional[List[pd.DataFrame]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data from market data
        
        Args:
            market_data: List of OHLCV DataFrames
            indicators_data: Optional list of indicator DataFrames
            
        Returns:
            Tuple of (X_images, y_patterns, y_breakouts)
        """
        X_images = []
        y_patterns = []
        y_breakouts = []
        
        for i, data in enumerate(market_data):
            if len(data) < self.config.lookback_periods:
                continue
            
            # Get indicators for this data if available
            indicators = indicators_data[i] if indicators_data and i < len(indicators_data) else None
            
            # Convert to image
            image = self._convert_data_to_image(data, indicators)
            X_images.append(image)
            
            # Get pattern label
            pattern_name, confidence, breakout_dir = self._identify_pattern_from_data(data)
            
            if pattern_name in self.pattern_types:
                pattern_idx = self.pattern_types.index(pattern_name)
            else:
                pattern_idx = self.pattern_types.index('none')
            
            y_patterns.append(pattern_idx)
            y_breakouts.append([breakout_dir, confidence, 0.0])  # direction, confidence, target
        
        return np.array(X_images), np.array(y_patterns), np.array(y_breakouts)
    
    def train(self, market_data: List[pd.DataFrame],
             indicators_data: Optional[List[pd.DataFrame]] = None,
             validation_data: Optional[Tuple[List[pd.DataFrame], List[pd.DataFrame]]] = None) -> Dict[str, Any]:
        """
        Train the CNN model
        
        Args:
            market_data: Training market data
            indicators_data: Optional training indicators
            validation_data: Optional validation data
            
        Returns:
            Training results and metrics
        """
        if not TF_AVAILABLE:
            return self._fallback_training(market_data)
        
        self.logger.info(f"Training Pattern CNN model with {len(market_data)} samples")
        
        # Prepare training data
        X_train, y_patterns, y_breakouts = self.prepare_training_data(market_data, indicators_data)
        
        if len(X_train) == 0:
            raise ValueError("No valid training data generated")
        
        # Prepare validation data if provided
        X_val, y_patterns_val, y_breakouts_val = None, None, None
        if validation_data:
            X_val, y_patterns_val, y_breakouts_val = self.prepare_training_data(
                validation_data[0], validation_data[1] if len(validation_data) > 1 else None
            )
        
        # Build model
        self.model = self._build_model()
        
        # Check if model was built successfully
        if self.model is None:
            self.logger.error("Failed to build CNN model - TensorFlow may not be available")
            return self._fallback_training(market_data)
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=15,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7
            )
        ]
        
        # Prepare targets
        y_train = {
            'pattern_classification': y_patterns,
            'breakout_prediction': y_breakouts
        }
        
        validation_data_tuple = None
        if validation_data and X_val is not None:
            y_val = {
                'pattern_classification': y_patterns_val,
                'breakout_prediction': y_breakouts_val
            }
            validation_data_tuple = (X_val, y_val)
        
        # Train model
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
        pattern_accuracy = accuracy_score(y_patterns, np.argmax(train_predictions[0], axis=1))
        breakout_mae = np.mean(np.abs(train_predictions[1] - y_breakouts))
        
        self.performance_metrics = {
            'pattern_accuracy': float(pattern_accuracy),
            'breakout_mae': float(breakout_mae),
            'final_loss': float(history.history['loss'][-1]),
            'epochs_trained': len(history.history['loss']),
            'total_patterns': len(set(y_patterns))
        }
        
        self.logger.info(f"Training completed. Pattern Accuracy: {pattern_accuracy:.4f}, Breakout MAE: {breakout_mae:.4f}")
        
        return {
            'metrics': self.performance_metrics,
            'history': history.history
        }
    
    def detect_pattern(self, ohlcv_data: pd.DataFrame, 
                      indicators: Optional[pd.DataFrame] = None) -> PatternDetection:
        """
        Detect chart pattern from recent data
        
        Args:
            ohlcv_data: Recent OHLC + Volume data
            indicators: Optional technical indicators
            
        Returns:
            PatternDetection with pattern results
        """
        if not self.is_trained:
            return self._fallback_detection(ohlcv_data)
        
        if len(ohlcv_data) < self.config.lookback_periods:
            self.logger.warning(f"Insufficient data for pattern detection. Need {self.config.lookback_periods}, got {len(ohlcv_data)}")
            return PatternDetection("insufficient_data", 0.0, 0, None, 0.0, None)
        
        # Convert to image
        image = self._convert_data_to_image(ohlcv_data, indicators)
        X = image.reshape(1, *image.shape)
        
        # Make prediction
        if TF_AVAILABLE and self.model:
            predictions = self.model.predict(X, verbose=0)
            pattern_probs = predictions[0][0]
            breakout_pred = predictions[1][0]
        else:
            return self._fallback_detection(ohlcv_data)
        
        # Interpret results
        max_prob_idx = np.argmax(pattern_probs)
        pattern_type = self.pattern_types[max_prob_idx]
        confidence = float(pattern_probs[max_prob_idx])
        
        # Breakout prediction
        breakout_direction = int(np.round(breakout_pred[0]))
        breakout_confidence = float(breakout_pred[1])
        
        # Calculate target price if pattern has directional bias
        target_price = None
        if pattern_type != 'none' and abs(breakout_direction) > 0:
            current_price = ohlcv_data['close'].iloc[-1]
            price_range = ohlcv_data['high'].max() - ohlcv_data['low'].min()
            target_move = price_range * 0.5 * breakout_direction  # Conservative target
            target_price = current_price + target_move
        
        # Overall signal strength
        strength = confidence * breakout_confidence if pattern_type != 'none' else 0.0
        
        # Extract pattern coordinates (simplified)
        pattern_coordinates = self._extract_pattern_coordinates(ohlcv_data, pattern_type)
        
        return PatternDetection(
            pattern_type=pattern_type,
            confidence=confidence,
            breakout_direction=breakout_direction,
            target_price=target_price,
            strength=strength,
            pattern_coordinates=pattern_coordinates
        )
    
    def _extract_pattern_coordinates(self, ohlcv_data: pd.DataFrame, pattern_type: str) -> Dict[str, List[float]]:
        """Extract key coordinates for pattern visualization"""
        coordinates = {}
        
        if pattern_type.startswith('triangle'):
            # Find trend lines for triangles
            highs = ohlcv_data['high'].values
            lows = ohlcv_data['low'].values
            
            # Simple trend line extraction
            coordinates['resistance'] = [highs[0], highs[-1]]
            coordinates['support'] = [lows[0], lows[-1]]
        
        elif pattern_type in ['head_shoulders', 'inverse_head_shoulders']:
            # Find peaks/troughs for head and shoulders
            if pattern_type == 'head_shoulders':
                data = ohlcv_data['high'].values
                reverse = True
            else:
                data = ohlcv_data['low'].values
                reverse = False
            
            peaks = []
            for i in range(1, len(data) - 1):
                if reverse:
                    if data[i] > data[i-1] and data[i] > data[i+1]:
                        peaks.append(data[i])
                else:
                    if data[i] < data[i-1] and data[i] < data[i+1]:
                        peaks.append(data[i])
            
            if len(peaks) >= 3:
                coordinates['key_points'] = peaks[:3]
        
        return coordinates
    
    def _fallback_detection(self, ohlcv_data: pd.DataFrame) -> PatternDetection:
        """Fallback pattern detection when TensorFlow is not available"""
        # Simple rule-based pattern detection
        pattern_name, confidence, breakout_dir = self._identify_pattern_from_data(ohlcv_data)
        
        target_price = None
        if abs(breakout_dir) > 0:
            current_price = ohlcv_data['close'].iloc[-1]
            price_range = ohlcv_data['high'].max() - ohlcv_data['low'].min()
            target_price = current_price + (price_range * 0.3 * breakout_dir)
        
        return PatternDetection(
            pattern_type=pattern_name,
            confidence=confidence,
            breakout_direction=breakout_dir,
            target_price=target_price,
            strength=confidence * 0.7,
            pattern_coordinates=self._extract_pattern_coordinates(ohlcv_data, pattern_name)
        )
    
    def _fallback_training(self, market_data: List[pd.DataFrame]) -> Dict[str, Any]:
        """Fallback training when TensorFlow is not available"""
        self.logger.info("Using fallback training (rule-based pattern analysis)")
        self.is_trained = True
        
        # Analyze patterns in training data
        pattern_counts = {pattern: 0 for pattern in self.pattern_types}
        
        for data in market_data:
            pattern_name, _, _ = self._identify_pattern_from_data(data)
            if pattern_name in pattern_counts:
                pattern_counts[pattern_name] += 1
        
        self.performance_metrics = {
            'pattern_counts': pattern_counts,
            'total_samples': len(market_data),
            'fallback_mode': True
        }
        
        return {'metrics': self.performance_metrics}
    
    def save_model(self, filepath: str):
        """Save trained model and components"""
        model_dir = Path(filepath).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if TF_AVAILABLE and self.model:
            self.model.save(f"{filepath}_model.h5")
        
        # Save configuration and metrics
        model_info = {
            'config': {
                'image_size': self.config.image_size,
                'lookback_periods': self.config.lookback_periods,
                'num_channels': self.config.num_channels,
                'learning_rate': self.config.learning_rate
            },
            'pattern_types': self.pattern_types,
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained,
            'tf_available': TF_AVAILABLE
        }
        
        with open(f"{filepath}_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        self.logger.info(f"Pattern CNN model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and components"""
        try:
            # Load model
            if TF_AVAILABLE and Path(f"{filepath}_model.h5").exists():
                self.model = tf.keras.models.load_model(f"{filepath}_model.h5")
            
            # Load configuration
            with open(f"{filepath}_info.json", 'r') as f:
                model_info = json.load(f)
                self.pattern_types = model_info['pattern_types']
                self.performance_metrics = model_info['performance_metrics']
                self.is_trained = model_info['is_trained']
            
            self.logger.info(f"Pattern CNN model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary and performance metrics"""
        summary = {
            'model_type': 'Pattern_CNN',
            'is_trained': self.is_trained,
            'tensorflow_available': TF_AVAILABLE,
            'config': {
                'image_size': self.config.image_size,
                'lookback_periods': self.config.lookback_periods,
                'num_channels': self.config.num_channels
            },
            'pattern_types': self.pattern_types,
            'performance': self.performance_metrics
        }
        
        if TF_AVAILABLE and self.model:
            summary['model_parameters'] = self.model.count_params()
        
        return summary


# Example usage
if __name__ == "__main__":
    # Generate sample market data for testing
    np.random.seed(42)
    
    def generate_sample_data(periods=100):
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')
        
        # Generate realistic OHLC data with patterns
        base_price = 20000
        prices = [base_price]
        
        for _ in range(periods - 1):
            change = np.random.normal(0, 0.02) * prices[-1]
            prices.append(prices[-1] + change)
        
        # Create OHLC from prices
        opens = prices[:-1]
        closes = prices[1:]
        highs = [max(o, c) * (1 + np.random.uniform(0, 0.01)) for o, c in zip(opens, closes)]
        lows = [min(o, c) * (1 - np.random.uniform(0, 0.01)) for o, c in zip(opens, closes)]
        volumes = np.random.randint(50000, 200000, len(opens))
        
        return pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=dates[:-1])
    
    # Generate training data
    training_data = [generate_sample_data(80) for _ in range(50)]
    validation_data = [generate_sample_data(80) for _ in range(10)]
    
    # Initialize and train model
    model = PatternCNNModel()
    
    print("Training Pattern CNN Model...")
    results = model.train(training_data, validation_data=(validation_data, None))
    print(f"Training Results: {results['metrics']}")
    
    # Test pattern detection
    test_data = generate_sample_data(60)
    pattern = model.detect_pattern(test_data)
    
    print(f"\nDetected Pattern: {pattern.pattern_type}")
    print(f"Confidence: {pattern.confidence:.3f}")
    print(f"Breakout Direction: {pattern.breakout_direction}")
    print(f"Target Price: {pattern.target_price}")
    print(f"Signal Strength: {pattern.strength:.3f}")
    
    # Model summary
    summary = model.get_model_summary()
    print(f"\nModel Summary: {summary}")