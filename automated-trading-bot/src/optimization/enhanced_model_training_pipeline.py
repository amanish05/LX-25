"""
Enhanced Model Training Pipeline with Advanced Features
Integrates market regime detection, order flow analysis, and robust validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

# Import our new modules
from src.bot_selection.market_regime_detector import MarketRegimeDetector, MarketRegime
from src.optimization.order_flow_analyzer import OrderFlowAnalyzer
from src.optimization.feature_importance_tracker import FeatureImportanceTracker
from src.optimization.time_series_validator import TimeSeriesValidator
from src.data.data_validator import DataValidator
from src.data.historical_data_collector import EnhancedHistoricalDataCollector

# Import existing modules
from src.optimization.model_training_pipeline import ModelTrainingPipeline
from src.indicators.price_action_composite import PriceActionComposite
from src.indicators.advanced_confirmation import AdvancedConfirmationSystem
from src.indicators.signal_validator import SignalValidator

# Import Individual Indicator Intelligence components
try:
    from src.ml.indicator_ensemble import IndicatorEnsemble, EnsembleConfig
    from src.ml.models.rsi_lstm_model import RSILSTMModel, RSIModelConfig
    from src.ml.models.pattern_cnn_model import PatternCNNModel, CNNModelConfig
    from src.ml.models.adaptive_thresholds_rl import AdaptiveThresholdsRL, RLConfig
    from src.optimization.genetic_optimizer import IndividualIndicatorOptimizer, EnsembleOptimizer
    INDIVIDUAL_INDICATOR_INTELLIGENCE_AVAILABLE = True
except ImportError:
    INDIVIDUAL_INDICATOR_INTELLIGENCE_AVAILABLE = False


class EnhancedModelTrainingPipeline(ModelTrainingPipeline):
    """
    Enhanced training pipeline with advanced features:
    - Market regime-aware training
    - Order flow features
    - SHAP feature importance
    - Robust time series validation
    - Advanced models (LightGBM, XGBoost)
    """
    
    def __init__(self, config_path: str = 'config/price_action_fine_tuned.json'):
        """Initialize enhanced pipeline"""
        super().__init__(config_path)
        
        # Initialize new components
        self.regime_detector = MarketRegimeDetector()
        self.order_flow_analyzer = OrderFlowAnalyzer()
        self.feature_tracker = FeatureImportanceTracker()
        self.ts_validator = TimeSeriesValidator()
        self.data_validator = DataValidator()
        
        # Enhanced models
        self.models.update({
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        })
        
        # Results storage
        self.enhanced_results = {}
        self.regime_analysis = {}
        self.feature_analysis = {}
        
        # Individual Indicator Intelligence components
        self.indicator_ensemble = None
        self.ml_models = {}
        self.individual_indicator_optimizer = None
        
    def generate_enhanced_features(self, data: pd.DataFrame, 
                                 tick_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate features including new advanced features
        
        Args:
            data: OHLCV data
            tick_data: Optional tick data for order flow analysis
            
        Returns:
            Enhanced feature DataFrame
        """
        # Start with base features
        features = self.generate_features(data)
        
        # 1. Market Regime Features
        if len(data) > 100:  # Need sufficient data for regime detection
            # Fit regime detector if not already fitted
            if not self.regime_detector.is_fitted:
                self.regime_detector.fit(data)
            
            # Detect current regime
            regime_analysis = self.regime_detector.detect_regime(data)
            
            # Add regime features
            features['market_regime'] = regime_analysis.regime_history.map(
                lambda x: list(self.regime_detector.regime_map.values()).index(
                    self.regime_detector.regime_map.get(x, MarketRegime.UNKNOWN)
                )
            )
            features['regime_probability'] = regime_analysis.regime_probability
            features['regime_duration'] = regime_analysis.regime_duration
            
            # Add regime characteristics
            for char_name, char_value in regime_analysis.regime_characteristics.items():
                if isinstance(char_value, (int, float)):
                    features[f'regime_{char_name}'] = char_value
        
        # 2. Order Flow Features (if tick data available)
        if tick_data is not None and len(tick_data) > 0:
            # Calculate microstructure features
            flow_features = self.order_flow_analyzer.calculate_microstructure_features(
                tick_data, window_size=100
            )
            
            # Merge with main features (align by timestamp)
            if not flow_features.empty:
                # Resample tick features to match OHLCV frequency
                flow_features_resampled = flow_features.resample(
                    pd.infer_freq(data.index) or '5T'
                ).last().fillna(method='ffill')
                
                # Merge
                features = features.join(flow_features_resampled, how='left')
        
        # 3. Advanced Price Action Features
        # Market structure trend
        features['structure_trend'] = self._calculate_structure_trend(data)
        
        # Volatility regime
        features['volatility_regime'] = self._classify_volatility_regime(features['volatility_20'])
        
        # Volume profile
        features['volume_profile'] = self._calculate_volume_profile(data)
        
        # 4. Interaction Features
        # Momentum * Volume
        features['momentum_volume'] = features.get('momentum_5', 0) * features.get('volume_ratio', 1)
        
        # Volatility * Regime
        features['volatility_regime_interaction'] = features.get('volatility_ratio', 1) * features.get('market_regime', 0)
        
        # Trend * Market Structure
        features['trend_structure'] = features.get('trend_strength', 0) * features.get('structure_trend', 0)
        
        # 5. Lag Features (important for time series)
        lag_features = ['returns', 'volume_ratio', 'volatility_20']
        for feature in lag_features:
            if feature in features.columns:
                for lag in [1, 5, 20]:
                    features[f'{feature}_lag{lag}'] = features[feature].shift(lag)
        
        # Remove NaN values
        features = features.dropna()
        
        return features
    
    def _calculate_structure_trend(self, data: pd.DataFrame) -> pd.Series:
        """Calculate market structure trend strength"""
        # Simplified version - in practice would use MarketStructure indicator
        highs = data['high'].rolling(20).max()
        lows = data['low'].rolling(20).min()
        
        # Trend strength based on higher highs/lower lows
        hh = (data['high'] > highs.shift(1)).rolling(5).sum()
        ll = (data['low'] < lows.shift(1)).rolling(5).sum()
        
        trend = (hh - ll) / 5  # Normalize to -1 to 1
        return trend.fillna(0)
    
    def _classify_volatility_regime(self, volatility: pd.Series) -> pd.Series:
        """Classify volatility into regimes"""
        # Calculate volatility percentiles
        vol_percentiles = volatility.rolling(252).rank(pct=True)
        
        # Classify into regimes
        regime = pd.Series(index=volatility.index, dtype=int)
        regime[vol_percentiles < 0.25] = 0  # Low volatility
        regime[(vol_percentiles >= 0.25) & (vol_percentiles < 0.75)] = 1  # Normal
        regime[vol_percentiles >= 0.75] = 2  # High volatility
        
        return regime.fillna(1)
    
    def _calculate_volume_profile(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volume profile indicator"""
        # Volume at price levels
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate relative volume at different price levels
        price_bins = pd.qcut(typical_price, q=10, duplicates='drop')
        volume_profile = data.groupby(price_bins)['volume'].transform('mean')
        
        # Normalize by current volume
        return (data['volume'] / volume_profile).fillna(1)
    
    def train_enhanced_models(self, 
                            X_train: pd.DataFrame, 
                            y_train: pd.Series,
                            X_val: pd.DataFrame, 
                            y_val: pd.Series,
                            use_shap: bool = True,
                            use_regime_specific: bool = True) -> Dict[str, Any]:
        """Train models with enhanced features and analysis
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            use_shap: Calculate SHAP values
            use_regime_specific: Train regime-specific models
            
        Returns:
            Dictionary with results
        """
        results = {}
        
        # 1. Train base models
        base_results = self.train_models(X_train, y_train, X_val, y_val)
        results['base_models'] = base_results
        
        # 2. Time Series Cross-Validation
        print("\nPerforming time series cross-validation...")
        cv_results = {}
        
        for model_name, model in self.models.items():
            print(f"\nValidating {model_name}...")
            
            # Combine train and validation for CV
            X_cv = pd.concat([X_train, X_val])
            y_cv = pd.concat([y_train, y_val])
            
            # Ensure datetime index
            if not isinstance(X_cv.index, pd.DatetimeIndex):
                # Create synthetic datetime index
                X_cv.index = pd.date_range(
                    start='2020-01-01', 
                    periods=len(X_cv), 
                    freq='D'
                )
            
            # Perform purged k-fold validation
            cv_result = self.ts_validator.validate_model(
                model=model,
                X=X_cv,
                y=y_cv,
                cv_method='purged_kfold',
                scoring='accuracy',
                n_splits=5,
                purge_days=5,
                embargo_days=2
            )
            
            cv_results[model_name] = cv_result
            
            print(f"{model_name} - CV Avg Accuracy: {cv_result.avg_metrics['accuracy']:.3f} "
                  f"(±{cv_result.std_metrics['accuracy']:.3f})")
        
        results['cv_results'] = cv_results
        
        # 3. SHAP Feature Importance Analysis
        if use_shap:
            print("\nCalculating SHAP values...")
            shap_results = {}
            
            for model_name in ['random_forest', 'lightgbm']:  # Tree-based models
                if model_name in self.models:
                    model = self.models[model_name]
                    
                    # Fit model if not already fitted
                    if not hasattr(model, 'classes_'):
                        model.fit(self.scaler.transform(X_train), y_train)
                    
                    # Calculate SHAP values
                    shap_analysis = self.feature_tracker.calculate_shap_values(
                        model=model,
                        X_data=X_val,
                        model_name=model_name,
                        data_subset='validation',
                        sample_size=min(1000, len(X_val))
                    )
                    
                    shap_results[model_name] = shap_analysis
            
            results['shap_analysis'] = shap_results
            
            # Get top features
            top_features = self.feature_tracker.get_top_features(n_features=20)
            results['top_features'] = top_features
            print(f"\nTop 5 features: {[f[0] for f in top_features[:5]]}")
        
        # 4. Regime-Specific Model Training
        if use_regime_specific and 'market_regime' in X_train.columns:
            print("\nTraining regime-specific models...")
            regime_models = {}
            
            for regime_id in X_train['market_regime'].unique():
                # Filter data for this regime
                regime_mask_train = X_train['market_regime'] == regime_id
                regime_mask_val = X_val['market_regime'] == regime_id
                
                if regime_mask_train.sum() > 50 and regime_mask_val.sum() > 10:
                    X_regime_train = X_train[regime_mask_train]
                    y_regime_train = y_train[regime_mask_train]
                    X_regime_val = X_val[regime_mask_val]
                    y_regime_val = y_val[regime_mask_val]
                    
                    # Train LightGBM for this regime
                    regime_model = lgb.LGBMClassifier(
                        n_estimators=50,
                        learning_rate=0.1,
                        max_depth=4,
                        random_state=42
                    )
                    
                    regime_model.fit(
                        self.scaler.transform(X_regime_train),
                        y_regime_train
                    )
                    
                    # Evaluate
                    regime_score = regime_model.score(
                        self.scaler.transform(X_regime_val),
                        y_regime_val
                    )
                    
                    regime_name = list(self.regime_detector.regime_map.values())[int(regime_id)]
                    regime_models[regime_name.value] = {
                        'model': regime_model,
                        'accuracy': regime_score,
                        'train_samples': regime_mask_train.sum(),
                        'val_samples': regime_mask_val.sum()
                    }
                    
                    print(f"  {regime_name.value}: Accuracy={regime_score:.3f}, "
                          f"Samples={regime_mask_train.sum()}")
            
            results['regime_models'] = regime_models
        
        # 5. Feature Analysis Report
        feature_analysis = self.feature_tracker.analyze_feature_trends(days=30)
        results['feature_trends'] = feature_analysis
        
        # 6. Generate ensemble predictions
        ensemble_predictions = self._generate_ensemble_predictions(
            X_val, base_results, regime_models if use_regime_specific else None
        )
        results['ensemble_predictions'] = ensemble_predictions
        
        self.enhanced_results = results
        return results
    
    def _generate_ensemble_predictions(self, 
                                     X_val: pd.DataFrame,
                                     base_results: Dict,
                                     regime_models: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate ensemble predictions combining all models"""
        predictions = {}
        
        # Collect base model predictions
        base_preds = []
        for model_name, model_results in base_results.items():
            preds = model_results['predictions']['val']
            base_preds.append(preds)
            predictions[f'{model_name}_pred'] = preds
        
        # Simple majority voting
        ensemble_pred = np.round(np.mean(base_preds, axis=0)).astype(int)
        predictions['ensemble_vote'] = ensemble_pred
        
        # Weighted ensemble (by validation accuracy)
        weights = []
        for model_name, model_results in base_results.items():
            weights.append(model_results['val_metrics']['accuracy'])
        
        weights = np.array(weights) / np.sum(weights)
        weighted_pred = np.round(np.average(base_preds, axis=0, weights=weights)).astype(int)
        predictions['ensemble_weighted'] = weighted_pred
        
        # Regime-aware predictions
        if regime_models and 'market_regime' in X_val.columns:
            regime_pred = np.zeros(len(X_val))
            
            for regime_name, regime_data in regime_models.items():
                regime_id = list(self.regime_detector.regime_map.keys())[
                    list(self.regime_detector.regime_map.values()).index(
                        MarketRegime[regime_name.upper().replace(' ', '_')]
                    )
                ]
                
                mask = X_val['market_regime'] == regime_id
                if mask.any():
                    model = regime_data['model']
                    regime_pred[mask] = model.predict(self.scaler.transform(X_val[mask]))
            
            predictions['regime_aware'] = regime_pred.astype(int)
        
        return predictions
    
    def generate_enhanced_signals(self, 
                                data: pd.DataFrame,
                                model_name: str = 'lightgbm',
                                use_ensemble: bool = True) -> pd.DataFrame:
        """Generate trading signals using enhanced models
        
        Args:
            data: Market data
            model_name: Model to use
            use_ensemble: Use ensemble predictions
            
        Returns:
            DataFrame with signals
        """
        # Generate enhanced features
        features = self.generate_enhanced_features(data)
        
        # Base signals
        signals = self.generate_trading_signals(data, model_name)
        
        if use_ensemble and hasattr(self, 'enhanced_results'):
            # Add ensemble predictions
            features_scaled = self.scaler.transform(features)
            
            # Get predictions from all models
            ensemble_preds = []
            ensemble_probs = []
            
            for name, model in self.models.items():
                if hasattr(model, 'predict'):
                    preds = model.predict(features_scaled)
                    probs = model.predict_proba(features_scaled)[:, 1]
                    ensemble_preds.append(preds)
                    ensemble_probs.append(probs)
            
            # Ensemble signal
            signals['ensemble_signal'] = np.round(np.mean(ensemble_preds, axis=0))
            signals['ensemble_probability'] = np.mean(ensemble_probs, axis=0)
            
            # Regime-aware adjustments
            if 'market_regime' in features.columns:
                current_regime = features['market_regime'].iloc[-1]
                regime_name = list(self.regime_detector.regime_map.values())[int(current_regime)]
                
                # Adjust confidence based on regime
                regime_multipliers = {
                    MarketRegime.TRENDING_UP: 1.2,
                    MarketRegime.TRENDING_DOWN: 0.8,
                    MarketRegime.RANGING: 0.9,
                    MarketRegime.VOLATILE: 0.7
                }
                
                multiplier = regime_multipliers.get(regime_name, 1.0)
                signals['regime_adjusted_confidence'] = signals['ensemble_probability'] * multiplier
        
        return signals
    
    def save_enhanced_models(self, model_dir: str = "models/enhanced"):
        """Save all enhanced models and components"""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save base models
        for name, model in self.models.items():
            with open(model_path / f"{name}_model.pkl", 'wb') as f:
                pickle.dump(model, f)
        
        # Save scaler
        with open(model_path / "scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save regime detector
        self.regime_detector.save_model(str(model_path / "regime_detector.pkl"))
        
        # Save feature importance history
        self.feature_tracker._save_history()
        
        # Save results
        with open(model_path / "enhanced_results.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_serializable = self._make_serializable(self.enhanced_results)
            json.dump(results_serializable, f, indent=2)
        
        print(f"Enhanced models saved to {model_dir}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        else:
            return obj
    
    def generate_performance_report(self, output_dir: str = "reports"):
        """Generate comprehensive performance report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Feature importance report
        self.feature_tracker.generate_feature_report(
            str(output_path / "feature_importance_report.md")
        )
        
        # 2. Cross-validation reports
        if 'cv_results' in self.enhanced_results:
            for model_name, cv_result in self.enhanced_results['cv_results'].items():
                self.ts_validator.generate_validation_report(
                    cv_result,
                    model_name,
                    str(output_path / f"cv_report_{model_name}.md")
                )
        
        # 3. Enhanced performance summary
        self._generate_enhanced_summary(output_path / "enhanced_model_summary.md")
        
        print(f"Performance reports generated in {output_dir}")
    
    def _generate_enhanced_summary(self, output_path: Path):
        """Generate summary of enhanced model performance"""
        lines = [
            "# Enhanced Model Training Summary",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Model Performance Comparison\n"
        ]
        
        # Base model performance
        if 'base_models' in self.enhanced_results:
            lines.append("### Base Models\n")
            lines.append("| Model | Val Accuracy | Val Precision | Val Recall | Val F1 |")
            lines.append("|-------|--------------|---------------|------------|---------|")
            
            for model, results in self.enhanced_results['base_models'].items():
                metrics = results['val_metrics']
                lines.append(
                    f"| {model} | {metrics['accuracy']:.3f} | "
                    f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | "
                    f"{metrics['f1']:.3f} |"
                )
        
        # Cross-validation results
        if 'cv_results' in self.enhanced_results:
            lines.extend([
                "\n### Cross-Validation Results\n",
                "| Model | CV Accuracy (mean ± std) | Best Fold | Worst Fold |",
                "|-------|-------------------------|-----------|------------|"
            ])
            
            for model, cv_result in self.enhanced_results['cv_results'].items():
                lines.append(
                    f"| {model} | {cv_result.avg_metrics['accuracy']:.3f} ± "
                    f"{cv_result.std_metrics['accuracy']:.3f} | "
                    f"{cv_result.best_fold} | {cv_result.worst_fold} |"
                )
        
        # Top features
        if 'top_features' in self.enhanced_results:
            lines.extend([
                "\n## Top 10 Most Important Features\n",
                "| Rank | Feature | Importance |",
                "|------|---------|------------|"
            ])
            
            for i, (feature, importance) in enumerate(self.enhanced_results['top_features'][:10], 1):
                lines.append(f"| {i} | {feature} | {importance:.4f} |")
        
        # Regime-specific models
        if 'regime_models' in self.enhanced_results:
            lines.extend([
                "\n## Regime-Specific Model Performance\n",
                "| Regime | Accuracy | Training Samples |",
                "|--------|----------|------------------|"
            ])
            
            for regime, data in self.enhanced_results['regime_models'].items():
                lines.append(
                    f"| {regime} | {data['accuracy']:.3f} | {data['train_samples']} |"
                )
        
        # Key improvements
        lines.extend([
            "\n## Key Improvements\n",
            "1. **Enhanced Features**: Added market regime, order flow, and interaction features",
            "2. **Robust Validation**: Implemented purged k-fold cross-validation",
            "3. **Feature Importance**: SHAP-based feature analysis for interpretability",
            "4. **Regime Awareness**: Separate models for different market conditions",
            "5. **Ensemble Methods**: Multiple model combination strategies"
        ])
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
    
    def setup_individual_indicator_intelligence(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Setup Individual Indicator Intelligence system
        
        Args:
            config: Configuration for ML models and ensemble
            
        Returns:
            True if setup successful, False otherwise
        """
        if not INDIVIDUAL_INDICATOR_INTELLIGENCE_AVAILABLE:
            self.logger.warning("Individual Indicator Intelligence not available. Missing ML dependencies.")
            return False
        
        try:
            self.logger.info("Setting up Individual Indicator Intelligence system...")
            
            # Initialize ML models with configuration
            self._initialize_ml_models(config)
            
            # Setup indicator ensemble
            self._setup_indicator_ensemble(config)
            
            # Initialize optimizer
            self.individual_indicator_optimizer = IndividualIndicatorOptimizer(
                population_size=20,
                generations=15,
                n_jobs=2
            )
            
            self.logger.info("Individual Indicator Intelligence system setup complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup Individual Indicator Intelligence: {e}")
            return False
    
    def _initialize_ml_models(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ML models for Individual Indicator Intelligence"""
        
        # Default configurations
        default_config = {
            'rsi_lstm': {
                'lstm_units': 50,
                'dropout_rate': 0.2,
                'sequence_length': 20,
                'learning_rate': 0.001
            },
            'pattern_cnn': {
                'image_size': (64, 64),
                'lookback_periods': 50,
                'learning_rate': 0.001
            },
            'adaptive_thresholds': {
                'learning_rate': 0.0003,
                'total_timesteps': 10000
            }
        }
        
        # Use provided config or defaults
        ml_config = config.get('ml_models', default_config) if config else default_config
        
        # Initialize RSI LSTM Model
        try:
            rsi_config = RSIModelConfig(**ml_config['rsi_lstm'])
            self.ml_models['rsi_lstm'] = RSILSTMModel(rsi_config)
            self.logger.info("RSI LSTM model initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize RSI LSTM: {e}")
        
        # Initialize Pattern CNN Model
        try:
            cnn_config = CNNModelConfig(**ml_config['pattern_cnn'])
            self.ml_models['pattern_cnn'] = PatternCNNModel(cnn_config)
            self.logger.info("Pattern CNN model initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Pattern CNN: {e}")
        
        # Initialize Adaptive Thresholds RL Model
        try:
            rl_config = RLConfig(**ml_config['adaptive_thresholds'])
            self.ml_models['adaptive_thresholds'] = AdaptiveThresholdsRL(rl_config)
            self.logger.info("Adaptive Thresholds RL model initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Adaptive Thresholds RL: {e}")
    
    def _setup_indicator_ensemble(self, config: Optional[Dict[str, Any]] = None):
        """Setup indicator ensemble system"""
        
        # Default ensemble configuration
        default_ensemble_config = EnsembleConfig(
            weights={
                'ml_models': 0.4,
                'technical_indicators': 0.3,
                'price_action': 0.2,
                'confirmation_systems': 0.1
            },
            indicator_weights={
                'rsi_lstm': 0.15,
                'pattern_cnn': 0.15,
                'adaptive_thresholds': 0.10,
                'advanced_rsi': 0.10,
                'oscillator_matrix': 0.10,
                'price_action_composite': 0.20,
                'advanced_confirmation': 0.10,
                'signal_validator': 0.10
            },
            min_consensus_ratio=0.6,
            min_confidence=0.5,
            adaptive_weights=True
        )
        
        # Use provided config or defaults
        if config and 'ensemble' in config:
            ensemble_config = EnsembleConfig(**config['ensemble'])
        else:
            ensemble_config = default_ensemble_config
        
        # Initialize ensemble
        self.indicator_ensemble = IndicatorEnsemble(ensemble_config)
        
        # Add ML models to ensemble
        for name, model in self.ml_models.items():
            weight = ensemble_config.indicator_weights.get(name, 0.1)
            self.indicator_ensemble.add_ml_model(name, model, weight)
        
        # Add traditional indicators to ensemble
        try:
            self.indicator_ensemble.add_traditional_indicator(
                'advanced_rsi', 
                self.rsi_advanced,
                ensemble_config.indicator_weights.get('advanced_rsi', 0.1)
            )
            
            self.indicator_ensemble.add_traditional_indicator(
                'oscillator_matrix',
                self.oscillator_matrix,
                ensemble_config.indicator_weights.get('oscillator_matrix', 0.1)
            )
            
            self.indicator_ensemble.add_traditional_indicator(
                'price_action_composite',
                self.price_action,
                ensemble_config.indicator_weights.get('price_action_composite', 0.2)
            )
            
        except Exception as e:
            self.logger.warning(f"Some traditional indicators not available: {e}")
    
    def train_individual_indicator_intelligence(self, 
                                               train_data: pd.DataFrame,
                                               validation_data: Optional[pd.DataFrame] = None,
                                               optimize_ensemble: bool = True) -> Dict[str, Any]:
        """
        Train Individual Indicator Intelligence system
        
        Args:
            train_data: Training data
            validation_data: Optional validation data
            optimize_ensemble: Whether to optimize ensemble parameters
            
        Returns:
            Training results and performance metrics
        """
        if not INDIVIDUAL_INDICATOR_INTELLIGENCE_AVAILABLE or not self.indicator_ensemble:
            raise ValueError("Individual Indicator Intelligence not available or not setup")
        
        self.logger.info("Training Individual Indicator Intelligence system...")
        start_time = datetime.now()
        
        results = {}
        
        # 1. Train individual ML models
        ml_training_results = self._train_ml_models(train_data, validation_data)
        results['ml_model_training'] = ml_training_results
        
        # 2. Optimize ensemble parameters if requested
        if optimize_ensemble:
            ensemble_optimization_results = self._optimize_ensemble_parameters(train_data)
            results['ensemble_optimization'] = ensemble_optimization_results
        
        # 3. Evaluate ensemble performance
        ensemble_performance = self._evaluate_ensemble_performance(validation_data or train_data)
        results['ensemble_performance'] = ensemble_performance
        
        # 4. Generate comprehensive signals and evaluate
        signal_evaluation = self._evaluate_signal_generation(validation_data or train_data)
        results['signal_evaluation'] = signal_evaluation
        
        training_time = (datetime.now() - start_time).total_seconds()
        results['training_time'] = training_time
        results['timestamp'] = datetime.now().isoformat()
        
        self.logger.info(f"Individual Indicator Intelligence training completed in {training_time:.1f}s")
        
        # Store results
        self.enhanced_results['individual_indicator_intelligence'] = results
        
        return results
    
    def _train_ml_models(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train individual ML models"""
        ml_results = {}
        
        # Prepare common data
        rsi_data = self._calculate_rsi(train_data['close'])
        price_data = train_data['close']
        
        # Train RSI LSTM
        if 'rsi_lstm' in self.ml_models:
            try:
                self.logger.info("Training RSI LSTM model...")
                val_rsi = self._calculate_rsi(validation_data['close']) if validation_data is not None else None
                val_price = validation_data['close'] if validation_data is not None else None
                
                val_tuple = (val_rsi, val_price) if val_rsi is not None else None
                
                rsi_results = self.ml_models['rsi_lstm'].train(
                    rsi_data, price_data, val_tuple
                )
                ml_results['rsi_lstm'] = rsi_results
                
            except Exception as e:
                self.logger.error(f"RSI LSTM training failed: {e}")
                ml_results['rsi_lstm'] = {'error': str(e)}
        
        # Train Pattern CNN
        if 'pattern_cnn' in self.ml_models:
            try:
                self.logger.info("Training Pattern CNN model...")
                
                # Prepare data chunks for CNN training
                training_chunks = self._prepare_cnn_training_data(train_data)
                validation_chunks = self._prepare_cnn_training_data(validation_data) if validation_data is not None else None
                
                cnn_results = self.ml_models['pattern_cnn'].train(
                    training_chunks,
                    validation_data=(validation_chunks, None) if validation_chunks else None
                )
                ml_results['pattern_cnn'] = cnn_results
                
            except Exception as e:
                self.logger.error(f"Pattern CNN training failed: {e}")
                ml_results['pattern_cnn'] = {'error': str(e)}
        
        # Train Adaptive Thresholds RL
        if 'adaptive_thresholds' in self.ml_models:
            try:
                self.logger.info("Training Adaptive Thresholds RL model...")
                
                rl_results = self.ml_models['adaptive_thresholds'].train(
                    train_data,
                    validation_data=validation_data
                )
                ml_results['adaptive_thresholds'] = rl_results
                
            except Exception as e:
                self.logger.error(f"Adaptive Thresholds RL training failed: {e}")
                ml_results['adaptive_thresholds'] = {'error': str(e)}
        
        return ml_results
    
    def _prepare_cnn_training_data(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """Prepare data chunks for CNN training"""
        if data is None or len(data) < 100:
            return []
        
        chunks = []
        window_size = 80
        step_size = 20
        
        for i in range(window_size, len(data), step_size):
            chunk = data.iloc[i-window_size:i].copy()
            if len(chunk) == window_size:
                chunks.append(chunk)
        
        return chunks
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI for a price series"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _optimize_ensemble_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize ensemble parameters using genetic algorithm"""
        if not self.individual_indicator_optimizer:
            return {'error': 'Optimizer not initialized'}
        
        try:
            self.logger.info("Optimizing ensemble parameters...")
            
            # Simple evaluation function for ensemble optimization
            def evaluate_ensemble_config(params, data):
                try:
                    # Update ensemble with new parameters
                    if 'min_consensus_ratio' in params:
                        self.indicator_ensemble.config.min_consensus_ratio = params['min_consensus_ratio']
                    if 'min_confidence' in params:
                        self.indicator_ensemble.config.min_confidence = params['min_confidence']
                    
                    # Evaluate ensemble on a subset of data
                    subset_data = data.tail(200) if len(data) > 200 else data
                    
                    signals_generated = 0
                    successful_signals = 0
                    
                    window_size = 50
                    for i in range(window_size, len(subset_data) - 5, 10):
                        window_data = subset_data.iloc[i-window_size:i]
                        
                        try:
                            signal = self.indicator_ensemble.generate_ensemble_signal(window_data)
                            
                            if signal and signal.signal_type != 'hold':
                                signals_generated += 1
                                
                                # Simple forward return evaluation
                                current_price = subset_data.iloc[i]['close']
                                future_price = subset_data.iloc[i + 5]['close']
                                
                                if signal.signal_type == 'buy':
                                    success = future_price > current_price
                                else:
                                    success = future_price < current_price
                                
                                if success:
                                    successful_signals += 1
                        except:
                            continue
                    
                    if signals_generated == 0:
                        return 0.0
                    
                    win_rate = successful_signals / signals_generated
                    signal_frequency = signals_generated / (len(subset_data) / 10)
                    
                    # Balanced scoring
                    score = win_rate * 0.7 + min(signal_frequency, 0.3) * 0.3
                    return score
                    
                except Exception as e:
                    return 0.0
            
            # Run optimization
            optimization_result = self.individual_indicator_optimizer.optimize(
                evaluate_ensemble_config,
                data,
                verbose=False
            )
            
            return {
                'best_parameters': optimization_result.best_parameters,
                'best_fitness': optimization_result.best_fitness,
                'optimization_time': optimization_result.optimization_time
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble optimization failed: {e}")
            return {'error': str(e)}
    
    def _evaluate_ensemble_performance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate ensemble performance on data"""
        if len(data) < 100:
            return {'error': 'Insufficient data for evaluation'}
        
        try:
            signals_generated = []
            returns = []
            
            window_size = 50
            step_size = 5
            
            for i in range(window_size, len(data) - step_size, step_size):
                window_data = data.iloc[i-window_size:i]
                
                try:
                    signal = self.indicator_ensemble.generate_ensemble_signal(window_data)
                    
                    if signal and signal.signal_type != 'hold':
                        signals_generated.append(signal)
                        
                        # Calculate return
                        current_price = data.iloc[i]['close']
                        future_price = data.iloc[i + step_size]['close']
                        
                        if signal.signal_type == 'buy':
                            ret = (future_price - current_price) / current_price
                        else:
                            ret = (current_price - future_price) / current_price
                        
                        weighted_return = ret * signal.strength * signal.confidence
                        returns.append(weighted_return)
                
                except Exception:
                    continue
            
            if len(returns) == 0:
                return {'error': 'No signals generated'}
            
            # Calculate performance metrics
            total_return = sum(returns)
            avg_return = np.mean(returns)
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
            sharpe = avg_return / (np.std(returns) + 1e-8)
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # Signal quality metrics
            avg_confidence = np.mean([s.confidence for s in signals_generated])
            avg_consensus = np.mean([s.consensus_ratio for s in signals_generated])
            avg_strength = np.mean([s.strength for s in signals_generated])
            
            return {
                'total_signals': len(signals_generated),
                'total_return': total_return,
                'avg_return': avg_return,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'avg_confidence': avg_confidence,
                'avg_consensus_ratio': avg_consensus,
                'avg_signal_strength': avg_strength,
                'signal_frequency': len(signals_generated) / (len(data) / step_size)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        if not returns:
            return 0.0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return float(np.min(drawdown))
    
    def _evaluate_signal_generation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate signal generation capabilities"""
        try:
            # Test ensemble signal generation
            test_window = data.tail(100) if len(data) > 100 else data
            
            ensemble_signals = []
            individual_signals = {}
            
            # Generate signals from ensemble
            try:
                signal = self.indicator_ensemble.generate_ensemble_signal(test_window)
                if signal:
                    ensemble_signals.append(signal)
            except Exception as e:
                self.logger.warning(f"Ensemble signal generation failed: {e}")
            
            # Test individual ML model signal generation
            for name, model in self.ml_models.items():
                try:
                    if name == 'rsi_lstm':
                        rsi_data = self._calculate_rsi(test_window['close'])
                        pattern = model.predict_pattern(rsi_data.tail(25), test_window['close'].tail(25))
                        individual_signals[name] = {
                            'pattern_type': pattern.pattern_type,
                            'confidence': pattern.confidence,
                            'strength': pattern.strength
                        }
                    elif name == 'pattern_cnn':
                        pattern = model.detect_pattern(test_window)
                        individual_signals[name] = {
                            'pattern_type': pattern.pattern_type,
                            'confidence': pattern.confidence,
                            'strength': pattern.strength
                        }
                    elif name == 'adaptive_thresholds':
                        thresholds = model.get_current_thresholds()
                        individual_signals[name] = {
                            'thresholds': thresholds,
                            'model_ready': model.is_trained
                        }
                except Exception as e:
                    individual_signals[name] = {'error': str(e)}
            
            return {
                'ensemble_signals_generated': len(ensemble_signals),
                'ensemble_signals': [
                    {
                        'signal_type': s.signal_type,
                        'strength': s.strength,
                        'confidence': s.confidence,
                        'consensus_ratio': s.consensus_ratio,
                        'contributing_indicators': s.contributing_indicators
                    } for s in ensemble_signals
                ],
                'individual_model_signals': individual_signals
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_individual_indicator_intelligence_report(self, output_path: str = "reports/individual_indicator_intelligence_report.md"):
        """Generate comprehensive Individual Indicator Intelligence report"""
        if 'individual_indicator_intelligence' not in self.enhanced_results:
            self.logger.warning("No Individual Indicator Intelligence results available")
            return
        
        results = self.enhanced_results['individual_indicator_intelligence']
        
        lines = [
            "# Individual Indicator Intelligence Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Training Time: {results.get('training_time', 0):.1f}s",
            "\n## System Overview\n",
            "The Individual Indicator Intelligence system combines multiple ML models with traditional indicators",
            "to create a sophisticated ensemble trading signal generation system.\n",
            "### Components:\n",
            "- **RSI LSTM Model**: Neural network for RSI pattern recognition",
            "- **Pattern CNN Model**: Convolutional network for chart pattern detection", 
            "- **Adaptive Thresholds RL**: Reinforcement learning for dynamic threshold adjustment",
            "- **Indicator Ensemble**: Intelligent combination of all signals with dynamic weights\n"
        ]
        
        # ML Model Training Results
        if 'ml_model_training' in results:
            lines.extend([
                "## ML Model Training Results\n",
                "| Model | Status | Key Metrics |",
                "|-------|--------|-------------|"
            ])
            
            for model_name, model_results in results['ml_model_training'].items():
                if 'error' in model_results:
                    status = "❌ Failed"
                    metrics = f"Error: {model_results['error']}"
                else:
                    status = "✅ Success"
                    if 'metrics' in model_results:
                        metrics_dict = model_results['metrics']
                        if model_name == 'rsi_lstm':
                            metrics = f"RSI MAE: {metrics_dict.get('rsi_mae', 0):.4f}, Pattern Acc: {metrics_dict.get('pattern_accuracy', 0):.3f}"
                        elif model_name == 'pattern_cnn':
                            metrics = f"Pattern Acc: {metrics_dict.get('pattern_accuracy', 0):.3f}, Breakout MAE: {metrics_dict.get('breakout_mae', 0):.4f}"
                        elif model_name == 'adaptive_thresholds':
                            metrics = f"Training timesteps: {metrics_dict.get('total_timesteps', 0)}"
                        else:
                            metrics = str(metrics_dict)
                    else:
                        metrics = "Training completed"
                
                lines.append(f"| {model_name} | {status} | {metrics} |")
        
        # Ensemble Performance
        if 'ensemble_performance' in results:
            perf = results['ensemble_performance']
            
            if 'error' not in perf:
                lines.extend([
                    "\n## Ensemble Performance\n",
                    f"- **Total Signals Generated**: {perf.get('total_signals', 0)}",
                    f"- **Win Rate**: {perf.get('win_rate', 0):.1%}",
                    f"- **Average Return per Signal**: {perf.get('avg_return', 0):.2%}",
                    f"- **Sharpe Ratio**: {perf.get('sharpe_ratio', 0):.3f}",
                    f"- **Average Signal Confidence**: {perf.get('avg_confidence', 0):.3f}",
                    f"- **Average Consensus Ratio**: {perf.get('avg_consensus_ratio', 0):.3f}",
                    f"- **Signal Frequency**: {perf.get('signal_frequency', 0):.1%}",
                    f"- **Max Drawdown**: {perf.get('max_drawdown', 0):.2%}\n"
                ])
            else:
                lines.append(f"\n## Ensemble Performance\n\n❌ Error: {perf['error']}\n")
        
        # Signal Evaluation
        if 'signal_evaluation' in results:
            sig_eval = results['signal_evaluation']
            
            if 'error' not in sig_eval:
                lines.extend([
                    "## Signal Generation Analysis\n",
                    f"**Ensemble Signals**: {sig_eval.get('ensemble_signals_generated', 0)} signals generated\n"
                ])
                
                # Individual model analysis
                if 'individual_model_signals' in sig_eval:
                    lines.append("### Individual Model Status\n")
                    lines.extend([
                        "| Model | Status | Key Information |",
                        "|-------|--------|-----------------|"
                    ])
                    
                    for model_name, model_signal in sig_eval['individual_model_signals'].items():
                        if 'error' in model_signal:
                            status = "❌ Error"
                            info = model_signal['error']
                        else:
                            status = "✅ Active"
                            if model_name in ['rsi_lstm', 'pattern_cnn']:
                                info = f"Pattern: {model_signal.get('pattern_type', 'N/A')}, Confidence: {model_signal.get('confidence', 0):.3f}"
                            else:
                                info = f"Ready: {model_signal.get('model_ready', False)}"
                        
                        lines.append(f"| {model_name} | {status} | {info} |")
        
        # Optimization Results
        if 'ensemble_optimization' in results:
            opt_results = results['ensemble_optimization']
            
            if 'error' not in opt_results:
                lines.extend([
                    "\n## Ensemble Optimization Results\n",
                    f"- **Optimization Fitness**: {opt_results.get('best_fitness', 0):.4f}",
                    f"- **Optimization Time**: {opt_results.get('optimization_time', 0):.1f}s\n",
                    "### Optimized Parameters\n"
                ])
                
                if 'best_parameters' in opt_results:
                    for param, value in opt_results['best_parameters'].items():
                        if isinstance(value, float):
                            lines.append(f"- **{param}**: {value:.4f}")
                        else:
                            lines.append(f"- **{param}**: {value}")
        
        # Recommendations
        lines.extend([
            "\n## Recommendations\n",
            "1. **Monitor Performance**: Track ensemble win rate and signal quality metrics",
            "2. **Regular Retraining**: Retrain ML models monthly with new data",
            "3. **Parameter Optimization**: Run ensemble optimization weekly for changing markets",
            "4. **Signal Validation**: Use signal validator to filter low-quality signals",
            "5. **Risk Management**: Implement position sizing based on signal confidence\n",
            "## Next Steps\n",
            "1. Deploy ensemble in paper trading mode for validation",
            "2. Collect performance data for further optimization",
            "3. Implement adaptive learning for real-time improvement",
            "4. Add more sophisticated risk management rules"
        ])
        
        # Write report
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        self.logger.info(f"Individual Indicator Intelligence report generated: {output_path}")
    
    def save_individual_indicator_intelligence_models(self, model_dir: str = "models/individual_indicator_intelligence"):
        """Save all Individual Indicator Intelligence models"""
        if not INDIVIDUAL_INDICATOR_INTELLIGENCE_AVAILABLE or not self.ml_models:
            self.logger.warning("No Individual Indicator Intelligence models to save")
            return
        
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual ML models
        for name, model in self.ml_models.items():
            try:
                model.save_model(str(model_path / name))
                self.logger.info(f"Saved {name} model")
            except Exception as e:
                self.logger.error(f"Failed to save {name} model: {e}")
        
        # Save ensemble configuration
        if self.indicator_ensemble:
            try:
                self.indicator_ensemble.save_ensemble(str(model_path / "ensemble_config.json"))
                self.logger.info("Saved ensemble configuration")
            except Exception as e:
                self.logger.error(f"Failed to save ensemble: {e}")
        
        # Save optimizer state
        if self.individual_indicator_optimizer:
            try:
                self.individual_indicator_optimizer.save_results(str(model_path / "optimizer_results"))
                self.logger.info("Saved optimizer results")
            except Exception as e:
                self.logger.error(f"Failed to save optimizer: {e}")
        
        self.logger.info(f"Individual Indicator Intelligence models saved to {model_dir}")