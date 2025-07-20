"""
ML-Enhanced Price Action Validator
Reduces false BOS/CHoCH signals using machine learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from dataclasses import dataclass
import json
from pathlib import Path
import pickle

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Price Action ML validator will use fallback rules.")


@dataclass
class StructureBreakValidation:
    """Validation result for a structure break"""
    break_type: str  # 'BOS', 'CHoCH', 'CHoCH+'
    original_confidence: float  # From traditional detection
    ml_confidence: float  # ML validation score
    is_valid: bool  # Final decision
    rejection_reason: Optional[str] = None
    features: Optional[Dict[str, float]] = None


class PriceActionMLValidator:
    """
    ML Validator for Price Action signals
    
    Reduces false signals by validating:
    1. Break of Structure (BOS)
    2. Change of Character (CHoCH)
    3. Order Block quality
    4. Fair Value Gap reliability
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize ML validator"""
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Models for different validations
        self.bos_validator = None
        self.pattern_cnn = None
        self.volume_profile_model = None
        
        # Feature scalers
        self.bos_scaler = StandardScaler()
        self.pattern_scaler = StandardScaler()
        
        # Performance tracking
        self.validation_history = []
        self.false_positive_rate = 0.3  # Initial assumption
        
        # Initialize models if TensorFlow available
        if TF_AVAILABLE:
            self._build_models()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load validator configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'bos_confidence_threshold': 0.7,
            'choch_confidence_threshold': 0.75,
            'pattern_min_score': 0.65,
            'volume_confirmation_weight': 0.3,
            'time_of_day_weight': 0.1,
            'false_positive_penalty': 0.8
        }
    
    def _build_models(self):
        """Build ML models for validation"""
        if not TF_AVAILABLE:
            return
        
        # BOS/CHoCH Validator Model
        self.bos_validator = Sequential([
            Input(shape=(15,)),  # 15 features
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')  # Confidence score
        ])
        
        self.bos_validator.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Pattern Quality CNN
        self.pattern_cnn = Sequential([
            Input(shape=(64, 64, 4)),  # OHLC channels
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Pattern quality score
        ])
        
        self.pattern_cnn.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def validate_structure_break(self, break_event: Dict[str, Any], 
                               market_data: pd.DataFrame) -> StructureBreakValidation:
        """
        Validate a detected structure break (BOS/CHoCH)
        
        Args:
            break_event: Structure break detection from MarketStructure
            market_data: Recent market data for context
            
        Returns:
            Validation result with ML confidence
        """
        try:
            # Extract features for validation
            features = self._extract_break_features(break_event, market_data)
            
            # Get ML confidence score
            if TF_AVAILABLE and self.bos_validator and hasattr(self, 'is_trained') and self.is_trained:
                ml_confidence = self._get_ml_confidence(features)
            else:
                # Fallback rule-based validation
                ml_confidence = self._rule_based_validation(features)
            
            # Determine if valid
            threshold = self.config['bos_confidence_threshold']
            if break_event['type'] == 'CHoCH':
                threshold = self.config['choch_confidence_threshold']
            
            is_valid = ml_confidence >= threshold
            
            # Rejection reason
            rejection_reason = None
            if not is_valid:
                if ml_confidence < 0.3:
                    rejection_reason = "Very low ML confidence - likely false signal"
                elif features['volume_ratio'] < 0.8:
                    rejection_reason = "Insufficient volume confirmation"
                elif features['time_score'] < 0.3:
                    rejection_reason = "Poor timing (market open/close)"
                else:
                    rejection_reason = "Below confidence threshold"
            
            return StructureBreakValidation(
                break_type=break_event['type'],
                original_confidence=break_event.get('confidence', 0.5),
                ml_confidence=ml_confidence,
                is_valid=is_valid,
                rejection_reason=rejection_reason,
                features=features
            )
            
        except Exception as e:
            self.logger.error(f"Error validating structure break: {e}")
            # Conservative approach - reject on error
            return StructureBreakValidation(
                break_type=break_event['type'],
                original_confidence=break_event.get('confidence', 0.5),
                ml_confidence=0.0,
                is_valid=False,
                rejection_reason=f"Validation error: {str(e)}"
            )
    
    def _extract_break_features(self, break_event: Dict, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract features for ML validation"""
        features = {}
        
        # Price action features
        close_prices = market_data['close'].values
        volumes = market_data['volume'].values
        
        # 1. Price momentum leading to break
        momentum_5 = (close_prices[-1] - close_prices[-6]) / close_prices[-6]
        momentum_10 = (close_prices[-1] - close_prices[-11]) / close_prices[-11]
        features['momentum_5'] = momentum_5
        features['momentum_10'] = momentum_10
        
        # 2. Volume confirmation
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        features['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1.0
        features['volume_surge'] = 1.0 if current_volume > avg_volume * 1.5 else 0.0
        
        # 3. Volatility context
        returns = np.diff(close_prices) / close_prices[:-1]
        features['volatility'] = np.std(returns[-20:]) if len(returns) >= 20 else 0.01
        features['volatility_ratio'] = features['volatility'] / np.std(returns) if len(returns) > 0 else 1.0
        
        # 4. Time of day score (avoid first/last 30 min)
        current_time = market_data.index[-1]
        hour = current_time.hour
        minute = current_time.minute
        time_minutes = hour * 60 + minute
        
        # Indian market hours: 9:15 AM to 3:30 PM
        if 555 <= time_minutes <= 585:  # 9:15-9:45
            features['time_score'] = 0.3
        elif 900 <= time_minutes <= 930:  # 3:00-3:30
            features['time_score'] = 0.3
        else:
            features['time_score'] = 1.0
        
        # 5. Structure-specific features
        features['break_magnitude'] = abs(break_event.get('price_change', 0))
        features['levels_broken'] = break_event.get('levels_broken', 1)
        features['swing_strength'] = break_event.get('swing_strength', 0.5)
        
        # 6. Market context
        features['trend_strength'] = self._calculate_trend_strength(close_prices)
        features['support_resistance_distance'] = break_event.get('sr_distance', 0.01)
        
        # 7. Pattern quality scores
        features['pattern_clarity'] = break_event.get('pattern_clarity', 0.5)
        features['confluence_score'] = break_event.get('confluence_score', 0.5)
        
        # 8. Historical reliability
        features['similar_breaks_success_rate'] = self._get_historical_success_rate(break_event['type'])
        
        return features
    
    def _get_ml_confidence(self, features: Dict[str, float]) -> float:
        """Get ML confidence score"""
        try:
            # Prepare feature vector
            feature_vector = np.array([
                features['momentum_5'],
                features['momentum_10'],
                features['volume_ratio'],
                features['volume_surge'],
                features['volatility'],
                features['volatility_ratio'],
                features['time_score'],
                features['break_magnitude'],
                features['levels_broken'],
                features['swing_strength'],
                features['trend_strength'],
                features['support_resistance_distance'],
                features['pattern_clarity'],
                features['confluence_score'],
                features['similar_breaks_success_rate']
            ]).reshape(1, -1)
            
            # Scale features
            if hasattr(self, 'bos_scaler') and hasattr(self.bos_scaler, 'mean_'):
                feature_vector = self.bos_scaler.transform(feature_vector)
            
            # Get prediction
            confidence = self.bos_validator.predict(feature_vector, verbose=0)[0][0]
            
            # Apply false positive penalty if recent performance is poor
            if self.false_positive_rate > 0.4:
                confidence *= self.config['false_positive_penalty']
            
            return float(confidence)
            
        except Exception as e:
            self.logger.error(f"ML confidence calculation failed: {e}")
            return 0.5  # Neutral confidence on error
    
    def _rule_based_validation(self, features: Dict[str, float]) -> float:
        """Fallback rule-based validation"""
        score = 0.5  # Base score
        
        # Volume confirmation (30% weight)
        if features['volume_ratio'] > 1.5:
            score += 0.15
        elif features['volume_ratio'] > 1.2:
            score += 0.08
        elif features['volume_ratio'] < 0.8:
            score -= 0.15
        
        # Momentum alignment (25% weight)
        if abs(features['momentum_5']) > 0.005:  # 0.5% move
            score += 0.125
        if features['momentum_5'] * features['momentum_10'] > 0:  # Same direction
            score += 0.125
        
        # Time of day (10% weight)
        score += (features['time_score'] - 0.5) * 0.2
        
        # Volatility context (15% weight)
        if features['volatility_ratio'] > 1.5:  # Very high vol - reduce confidence
            score -= 0.075
        elif features['volatility_ratio'] > 1.2:  # Higher vol than usual - small bonus
            score += 0.025
        elif features['volatility_ratio'] < 0.8:  # Lower vol
            score -= 0.075
        
        # Pattern quality (20% weight)
        score += (features['pattern_clarity'] - 0.5) * 0.2
        score += (features['confluence_score'] - 0.5) * 0.2
        
        # Additional penalty for poor pattern quality in high volatility
        if features['volatility_ratio'] > 1.2 and features['pattern_clarity'] < 0.5:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength using linear regression"""
        if len(prices) < 10:
            return 0.5
        
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        # Normalize slope
        price_range = np.max(prices) - np.min(prices)
        if price_range > 0:
            normalized_slope = abs(slope) / price_range
            return min(normalized_slope * 10, 1.0)
        return 0.5
    
    def _get_historical_success_rate(self, break_type: str) -> float:
        """Get historical success rate for similar breaks"""
        if not self.validation_history:
            return 0.5  # Neutral if no history
        
        similar_breaks = [v for v in self.validation_history 
                         if v.get('break_type') == break_type]
        
        if len(similar_breaks) < 5:
            return 0.5
        
        successful = sum(1 for v in similar_breaks if v.get('was_successful', False))
        return successful / len(similar_breaks)
    
    def validate_order_block(self, order_block: Dict[str, Any], 
                           market_data: pd.DataFrame) -> Dict[str, float]:
        """Validate order block quality"""
        validation = {
            'volume_score': self._validate_order_block_volume(order_block, market_data),
            'structure_score': self._validate_order_block_structure(order_block, market_data),
            'confluence_score': order_block.get('confluence_score', 0.5),
            'ml_confidence': 0.5  # Default
        }
        
        # Calculate overall score
        validation['overall_score'] = (
            validation['volume_score'] * 0.4 +
            validation['structure_score'] * 0.3 +
            validation['confluence_score'] * 0.3
        )
        
        validation['is_valid'] = bool(validation['overall_score'] >= 0.65)
        
        return validation
    
    def _validate_order_block_volume(self, order_block: Dict, market_data: pd.DataFrame) -> float:
        """Validate order block volume profile"""
        ob_index = order_block.get('index', -1)
        if ob_index < 0 or ob_index >= len(market_data):
            return 0.5
        
        # Get volume at order block
        ob_volume = market_data.iloc[ob_index]['volume']
        avg_volume = market_data['volume'].rolling(20).mean().iloc[ob_index]
        
        if avg_volume > 0:
            volume_ratio = ob_volume / avg_volume
            return min(volume_ratio / 2, 1.0)  # Cap at 2x average
        return 0.5
    
    def _validate_order_block_structure(self, order_block: Dict, market_data: pd.DataFrame) -> float:
        """Validate order block structure"""
        score = 0.5
        
        # Check if order block has been tested
        if order_block.get('times_tested', 0) > 0:
            score += 0.2
        
        # Check if it's at a significant level
        if order_block.get('at_significant_level', False):
            score += 0.2
        
        # Check age of order block
        age = order_block.get('age_bars', 0)
        if 10 <= age <= 50:  # Not too new, not too old
            score += 0.1
        
        return min(score, 1.0)
    
    def update_performance(self, validation_id: str, was_successful: bool):
        """Update validator performance based on actual results"""
        # Find validation in history
        for v in self.validation_history:
            if v.get('id') == validation_id:
                v['was_successful'] = was_successful
                break
        
        # Update false positive rate
        recent_validations = self.validation_history[-100:]  # Last 100
        if len(recent_validations) >= 20:
            false_positives = sum(1 for v in recent_validations 
                                if v.get('ml_confidence', 0) > 0.7 and 
                                not v.get('was_successful', False))
            self.false_positive_rate = false_positives / len(recent_validations)
    
    def train_validator(self, training_data: List[Dict[str, Any]]):
        """Train the ML validator with historical data"""
        if not TF_AVAILABLE or not training_data:
            self.logger.warning("Cannot train validator - TensorFlow not available or no data")
            return
        
        # Prepare training data
        X = []
        y = []
        
        for sample in training_data:
            features = sample['features']
            feature_vector = [
                features['momentum_5'],
                features['momentum_10'],
                features['volume_ratio'],
                features['volume_surge'],
                features['volatility'],
                features['volatility_ratio'],
                features['time_score'],
                features['break_magnitude'],
                features['levels_broken'],
                features['swing_strength'],
                features['trend_strength'],
                features['support_resistance_distance'],
                features['pattern_clarity'],
                features['confluence_score'],
                features['similar_breaks_success_rate']
            ]
            X.append(feature_vector)
            y.append(1 if sample['was_successful'] else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X = self.bos_scaler.fit_transform(X)
        
        # Train model
        self.bos_validator.fit(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        self.is_trained = True
        self.logger.info("Price action ML validator trained successfully")
    
    def save_model(self, filepath: str):
        """Save trained model and scalers"""
        if not hasattr(self, 'is_trained') or not self.is_trained:
            self.logger.warning("Model not trained, nothing to save")
            return
        
        # Prepare scaler params if available
        scaler_params = {}
        if hasattr(self.bos_scaler, 'mean_') and hasattr(self.bos_scaler, 'scale_'):
            scaler_params = {
                'mean': self.bos_scaler.mean_.tolist(),
                'scale': self.bos_scaler.scale_.tolist()
            }
        
        model_data = {
            'config': self.config,
            'false_positive_rate': self.false_positive_rate,
            'validation_history': self.validation_history[-1000:],  # Keep last 1000
            'scaler_params': scaler_params
        }
        
        # Save model data
        with open(f"{filepath}_data.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save TensorFlow model
        if TF_AVAILABLE and self.bos_validator:
            self.bos_validator.save(f"{filepath}_bos_model.h5")
            if self.pattern_cnn:
                self.pattern_cnn.save(f"{filepath}_pattern_model.h5")
        
        self.logger.info(f"Price action validator saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and scalers"""
        try:
            # Load model data
            with open(f"{filepath}_data.pkl", 'rb') as f:
                model_data = pickle.load(f)
            
            self.config = model_data['config']
            self.false_positive_rate = model_data['false_positive_rate']
            self.validation_history = model_data['validation_history']
            
            # Restore scaler if available
            if model_data['scaler_params']:
                self.bos_scaler.mean_ = np.array(model_data['scaler_params']['mean'])
                self.bos_scaler.scale_ = np.array(model_data['scaler_params']['scale'])
            
            # Load TensorFlow models
            if TF_AVAILABLE:
                if Path(f"{filepath}_bos_model.h5").exists():
                    self.bos_validator = tf.keras.models.load_model(f"{filepath}_bos_model.h5")
                if Path(f"{filepath}_pattern_model.h5").exists():
                    self.pattern_cnn = tf.keras.models.load_model(f"{filepath}_pattern_model.h5")
            
            self.is_trained = True
            self.logger.info(f"Price action validator loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise