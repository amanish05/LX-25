"""
SHAP Feature Importance Tracker for ML Model Interpretability
Tracks and analyzes feature importance over time for better model understanding
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import shap
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator


@dataclass
class FeatureImportance:
    """Container for feature importance data"""
    feature_name: str
    importance_score: float
    shap_value: float
    timestamp: datetime
    model_name: str
    data_subset: str  # 'train', 'val', 'test', 'production'
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    

@dataclass
class FeatureAnalysis:
    """Comprehensive feature analysis results"""
    feature_name: str
    avg_importance: float
    importance_trend: float  # Positive = increasing, negative = decreasing
    stability_score: float  # 0-1, higher = more stable
    interaction_effects: Dict[str, float]
    time_varying_importance: pd.Series
    recommendation: str


class FeatureImportanceTracker:
    """
    Tracks and analyzes feature importance using SHAP values
    
    Features:
    - SHAP value calculation for any sklearn-compatible model
    - Feature importance tracking over time
    - Feature interaction analysis
    - Automatic feature selection recommendations
    - Drift detection in feature importance
    """
    
    def __init__(self,
                 storage_path: str = "models/feature_importance",
                 min_importance_threshold: float = 0.01,
                 stability_window: int = 30):
        """Initialize feature importance tracker
        
        Args:
            storage_path: Path to store feature importance history
            min_importance_threshold: Minimum importance to consider feature useful
            stability_window: Days to consider for stability calculation
        """
        self.logger = logging.getLogger(__name__)
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.min_importance_threshold = min_importance_threshold
        self.stability_window = stability_window
        
        # History tracking
        self.importance_history: Dict[str, List[FeatureImportance]] = defaultdict(list)
        self.shap_explainers: Dict[str, Any] = {}
        
        # Load existing history
        self._load_history()
        
    def calculate_shap_values(self,
                            model: BaseEstimator,
                            X_data: pd.DataFrame,
                            model_name: str,
                            data_subset: str = 'train',
                            sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Calculate SHAP values for a model and dataset
        
        Args:
            model: Trained sklearn-compatible model
            X_data: Feature data
            model_name: Name identifier for the model
            data_subset: Type of data ('train', 'val', 'test', 'production')
            sample_size: Number of samples to use (None = all)
            
        Returns:
            Dictionary with SHAP analysis results
        """
        self.logger.info(f"Calculating SHAP values for {model_name} on {data_subset} data")
        
        # Sample data if needed
        if sample_size and len(X_data) > sample_size:
            X_sample = X_data.sample(n=sample_size, random_state=42)
        else:
            X_sample = X_data
        
        # Create or get SHAP explainer
        if model_name not in self.shap_explainers:
            self.shap_explainers[model_name] = self._create_explainer(model, X_sample)
        
        explainer = self.shap_explainers[model_name]
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            # For multi-class, use the positive class (index 1) for binary classification
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(
            shap_values, X_sample, model_name, data_subset
        )
        
        # Store results
        timestamp = datetime.now()
        for feat_name, importance_data in feature_importance.items():
            self.importance_history[feat_name].append(
                FeatureImportance(
                    feature_name=feat_name,
                    importance_score=importance_data['importance'],
                    shap_value=importance_data['mean_shap'],
                    timestamp=timestamp,
                    model_name=model_name,
                    data_subset=data_subset,
                    confidence_interval=importance_data['confidence_interval']
                )
            )
        
        # Calculate interaction effects
        interaction_effects = self._calculate_interaction_effects(
            explainer, X_sample, sample_size=min(100, len(X_sample))
        )
        
        results = {
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'interaction_effects': interaction_effects,
            'base_value': explainer.expected_value,
            'timestamp': timestamp
        }
        
        # Save history
        self._save_history()
        
        return results
    
    def _create_explainer(self, model: BaseEstimator, X_sample: pd.DataFrame) -> Any:
        """Create appropriate SHAP explainer for the model"""
        model_type = type(model).__name__
        
        if 'Tree' in model_type or 'Forest' in model_type or 'Boost' in model_type:
            # Tree-based models (Random Forest, Gradient Boosting, XGBoost, etc.)
            return shap.TreeExplainer(model)
        elif 'Linear' in model_type:
            # Linear models
            return shap.LinearExplainer(model, X_sample)
        else:
            # Generic kernel explainer (slower but works for any model)
            return shap.KernelExplainer(model.predict_proba, X_sample.sample(min(100, len(X_sample))))
    
    def _calculate_feature_importance(self,
                                    shap_values: np.ndarray,
                                    X_data: pd.DataFrame,
                                    model_name: str,
                                    data_subset: str) -> Dict[str, Dict[str, float]]:
        """Calculate feature importance from SHAP values"""
        feature_importance = {}
        
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Calculate confidence intervals using bootstrap
        n_bootstrap = 100
        bootstrap_importance = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(shap_values), size=len(shap_values), replace=True)
            bootstrap_mean = np.abs(shap_values[indices]).mean(axis=0)
            bootstrap_importance.append(bootstrap_mean)
        
        bootstrap_importance = np.array(bootstrap_importance)
        
        for i, feature_name in enumerate(X_data.columns):
            importance = mean_abs_shap[i]
            mean_shap = shap_values[:, i].mean()
            
            # Calculate 95% confidence interval
            ci_lower = np.percentile(bootstrap_importance[:, i], 2.5)
            ci_upper = np.percentile(bootstrap_importance[:, i], 97.5)
            
            feature_importance[feature_name] = {
                'importance': float(importance),
                'mean_shap': float(mean_shap),
                'confidence_interval': (float(ci_lower), float(ci_upper)),
                'std_shap': float(shap_values[:, i].std()),
                'min_shap': float(shap_values[:, i].min()),
                'max_shap': float(shap_values[:, i].max())
            }
        
        return feature_importance
    
    def _calculate_interaction_effects(self,
                                     explainer: Any,
                                     X_sample: pd.DataFrame,
                                     sample_size: int = 100) -> Dict[str, Dict[str, float]]:
        """Calculate feature interaction effects"""
        interaction_effects = defaultdict(dict)
        
        try:
            # Calculate SHAP interaction values (computationally expensive)
            # Only for tree-based models
            if hasattr(explainer, 'shap_interaction_values'):
                interaction_sample = X_sample.sample(min(sample_size, len(X_sample)))
                interaction_values = explainer.shap_interaction_values(interaction_sample)
                
                # Calculate mean absolute interaction effects
                for i, feat1 in enumerate(X_sample.columns):
                    for j, feat2 in enumerate(X_sample.columns):
                        if i != j:
                            interaction_strength = np.abs(interaction_values[:, i, j]).mean()
                            interaction_effects[feat1][feat2] = float(interaction_strength)
        except Exception as e:
            self.logger.warning(f"Could not calculate interaction effects: {e}")
        
        return dict(interaction_effects)
    
    def analyze_feature_trends(self, 
                             feature_names: Optional[List[str]] = None,
                             days: int = 30) -> Dict[str, FeatureAnalysis]:
        """Analyze feature importance trends over time
        
        Args:
            feature_names: Features to analyze (None = all)
            days: Number of days to analyze
            
        Returns:
            Dictionary of FeatureAnalysis objects
        """
        if feature_names is None:
            feature_names = list(self.importance_history.keys())
        
        cutoff_date = datetime.now() - timedelta(days=days)
        analyses = {}
        
        for feature in feature_names:
            if feature not in self.importance_history:
                continue
            
            # Get recent history
            recent_history = [
                h for h in self.importance_history[feature]
                if h.timestamp >= cutoff_date
            ]
            
            if not recent_history:
                continue
            
            # Convert to time series
            importance_series = pd.Series(
                [h.importance_score for h in recent_history],
                index=[h.timestamp for h in recent_history]
            ).sort_index()
            
            # Calculate metrics
            avg_importance = importance_series.mean()
            
            # Trend (linear regression slope)
            if len(importance_series) > 1:
                x = np.arange(len(importance_series))
                slope = np.polyfit(x, importance_series.values, 1)[0]
                importance_trend = slope / avg_importance if avg_importance > 0 else 0
            else:
                importance_trend = 0
            
            # Stability (inverse of coefficient of variation)
            cv = importance_series.std() / importance_series.mean() if importance_series.mean() > 0 else 1
            stability_score = 1 / (1 + cv)
            
            # Get interaction effects from latest calculation
            interaction_effects = {}
            latest_interactions = self._get_latest_interactions(feature)
            if latest_interactions:
                interaction_effects = latest_interactions
            
            # Generate recommendation
            recommendation = self._generate_feature_recommendation(
                avg_importance, importance_trend, stability_score
            )
            
            analyses[feature] = FeatureAnalysis(
                feature_name=feature,
                avg_importance=avg_importance,
                importance_trend=importance_trend,
                stability_score=stability_score,
                interaction_effects=interaction_effects,
                time_varying_importance=importance_series,
                recommendation=recommendation
            )
        
        return analyses
    
    def _generate_feature_recommendation(self,
                                       avg_importance: float,
                                       trend: float,
                                       stability: float) -> str:
        """Generate recommendation for a feature"""
        if avg_importance < self.min_importance_threshold:
            if trend > 0.1:
                return "Monitor - Low importance but increasing"
            else:
                return "Consider removing - Consistently low importance"
        elif avg_importance > 0.1:
            if stability > 0.8:
                return "Keep - High importance and stable"
            else:
                return "Keep but monitor - High importance but unstable"
        else:
            if trend > 0.1 and stability > 0.7:
                return "Keep - Moderate importance, positive trend"
            elif trend < -0.1:
                return "Review - Declining importance"
            else:
                return "Keep - Moderate importance"
    
    def get_top_features(self, 
                        n_features: int = 20,
                        model_name: Optional[str] = None,
                        data_subset: Optional[str] = None) -> List[Tuple[str, float]]:
        """Get top N most important features
        
        Args:
            n_features: Number of top features to return
            model_name: Filter by model name
            data_subset: Filter by data subset
            
        Returns:
            List of (feature_name, importance) tuples
        """
        # Aggregate importance across history
        feature_scores = defaultdict(list)
        
        for feature, history in self.importance_history.items():
            for h in history:
                if model_name and h.model_name != model_name:
                    continue
                if data_subset and h.data_subset != data_subset:
                    continue
                feature_scores[feature].append(h.importance_score)
        
        # Calculate mean importance
        mean_importance = {
            feature: np.mean(scores) for feature, scores in feature_scores.items()
        }
        
        # Sort and return top N
        sorted_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n_features]
    
    def detect_feature_drift(self,
                           reference_period_days: int = 30,
                           test_period_days: int = 7,
                           drift_threshold: float = 0.3) -> Dict[str, bool]:
        """Detect significant changes in feature importance
        
        Args:
            reference_period_days: Days for reference period
            test_period_days: Days for test period
            drift_threshold: Threshold for drift detection (relative change)
            
        Returns:
            Dictionary of feature -> drift detected (True/False)
        """
        drift_detected = {}
        
        reference_start = datetime.now() - timedelta(days=reference_period_days + test_period_days)
        reference_end = datetime.now() - timedelta(days=test_period_days)
        test_start = reference_end
        test_end = datetime.now()
        
        for feature in self.importance_history:
            # Get reference period importance
            ref_importance = [
                h.importance_score for h in self.importance_history[feature]
                if reference_start <= h.timestamp < reference_end
            ]
            
            # Get test period importance
            test_importance = [
                h.importance_score for h in self.importance_history[feature]
                if test_start <= h.timestamp <= test_end
            ]
            
            if ref_importance and test_importance:
                ref_mean = np.mean(ref_importance)
                test_mean = np.mean(test_importance)
                
                if ref_mean > 0:
                    relative_change = abs(test_mean - ref_mean) / ref_mean
                    drift_detected[feature] = relative_change > drift_threshold
                else:
                    drift_detected[feature] = test_mean > self.min_importance_threshold
        
        return drift_detected
    
    def plot_feature_importance(self,
                              top_n: int = 20,
                              model_name: Optional[str] = None,
                              save_path: Optional[str] = None):
        """Plot feature importance visualization
        
        Args:
            top_n: Number of top features to show
            model_name: Filter by model name
            save_path: Path to save the plot
        """
        # Get top features
        top_features = self.get_top_features(top_n, model_name)
        
        if not top_features:
            self.logger.warning("No features to plot")
            return
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Bar plot of feature importance
        features, importances = zip(*top_features)
        y_pos = np.arange(len(features))
        
        ax1.barh(y_pos, importances)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(features)
        ax1.set_xlabel('Average Importance (SHAP)')
        ax1.set_title(f'Top {top_n} Feature Importance')
        ax1.grid(True, alpha=0.3)
        
        # Time series plot of top 5 features
        top_5_features = [f[0] for f in top_features[:5]]
        
        for feature in top_5_features:
            analysis = self.analyze_feature_trends([feature], days=30)
            if feature in analysis:
                ts = analysis[feature].time_varying_importance
                ax2.plot(ts.index, ts.values, label=feature, marker='o', markersize=4)
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Importance Score')
        ax2.set_title('Feature Importance Over Time (Top 5)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def generate_feature_report(self, output_path: str = "reports/feature_importance_report.md"):
        """Generate comprehensive feature importance report
        
        Args:
            output_path: Path to save the report
        """
        report_lines = [
            "# Feature Importance Analysis Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Executive Summary\n"
        ]
        
        # Get overall statistics
        total_features = len(self.importance_history)
        top_features = self.get_top_features(10)
        
        report_lines.extend([
            f"- Total features tracked: {total_features}",
            f"- Total importance records: {sum(len(h) for h in self.importance_history.values())}",
            f"- Models analyzed: {len(set(h.model_name for hist in self.importance_history.values() for h in hist))}",
            "\n## Top 10 Most Important Features\n"
        ])
        
        # Top features table
        report_lines.append("| Rank | Feature | Avg Importance | Trend | Stability |")
        report_lines.append("|------|---------|----------------|-------|-----------|")
        
        analyses = self.analyze_feature_trends([f[0] for f in top_features])
        
        for i, (feature, importance) in enumerate(top_features, 1):
            if feature in analyses:
                analysis = analyses[feature]
                trend_symbol = "↑" if analysis.importance_trend > 0.05 else "↓" if analysis.importance_trend < -0.05 else "→"
                report_lines.append(
                    f"| {i} | {feature} | {importance:.4f} | {trend_symbol} {analysis.importance_trend:+.2%} | {analysis.stability_score:.2%} |"
                )
        
        # Feature recommendations
        report_lines.extend([
            "\n## Feature Recommendations\n",
            "### Features to Remove (Low Importance)"
        ])
        
        low_importance_features = []
        declining_features = []
        unstable_features = []
        
        for feature, analysis in analyses.items():
            if "Consider removing" in analysis.recommendation:
                low_importance_features.append(feature)
            elif "Declining" in analysis.recommendation:
                declining_features.append(feature)
            elif "unstable" in analysis.recommendation:
                unstable_features.append(feature)
        
        if low_importance_features:
            for feature in low_importance_features[:5]:
                report_lines.append(f"- {feature}: {analyses[feature].avg_importance:.4f}")
        else:
            report_lines.append("- None identified")
        
        report_lines.extend([
            "\n### Features with Declining Importance"
        ])
        
        if declining_features:
            for feature in declining_features[:5]:
                report_lines.append(f"- {feature}: {analyses[feature].importance_trend:+.2%} trend")
        else:
            report_lines.append("- None identified")
        
        # Drift detection
        drift_results = self.detect_feature_drift()
        features_with_drift = [f for f, has_drift in drift_results.items() if has_drift]
        
        report_lines.extend([
            "\n## Feature Drift Detection\n",
            f"Features with significant drift: {len(features_with_drift)}"
        ])
        
        if features_with_drift:
            report_lines.append("\nDrifted features:")
            for feature in features_with_drift[:10]:
                report_lines.append(f"- {feature}")
        
        # Feature interactions
        report_lines.extend([
            "\n## Strong Feature Interactions\n",
            "Top feature pairs with strong interactions:"
        ])
        
        all_interactions = []
        for feature in top_features[:5]:
            latest_interactions = self._get_latest_interactions(feature[0])
            if latest_interactions:
                for interact_feat, strength in latest_interactions.items():
                    all_interactions.append((feature[0], interact_feat, strength))
        
        all_interactions.sort(key=lambda x: x[2], reverse=True)
        
        for feat1, feat2, strength in all_interactions[:10]:
            report_lines.append(f"- {feat1} ↔ {feat2}: {strength:.4f}")
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Feature importance report saved to {output_path}")
    
    def _get_latest_interactions(self, feature: str) -> Dict[str, float]:
        """Get latest interaction effects for a feature"""
        # This is a placeholder - would need to store interaction history
        return {}
    
    def _save_history(self):
        """Save importance history to disk"""
        history_data = {}
        
        for feature, history_list in self.importance_history.items():
            history_data[feature] = [
                {
                    'importance_score': h.importance_score,
                    'shap_value': h.shap_value,
                    'timestamp': h.timestamp.isoformat(),
                    'model_name': h.model_name,
                    'data_subset': h.data_subset,
                    'confidence_interval': h.confidence_interval
                }
                for h in history_list
            ]
        
        with open(self.storage_path / 'importance_history.json', 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def _load_history(self):
        """Load importance history from disk"""
        history_file = self.storage_path / 'importance_history.json'
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                for feature, history_list in history_data.items():
                    for h in history_list:
                        self.importance_history[feature].append(
                            FeatureImportance(
                                feature_name=feature,
                                importance_score=h['importance_score'],
                                shap_value=h['shap_value'],
                                timestamp=datetime.fromisoformat(h['timestamp']),
                                model_name=h['model_name'],
                                data_subset=h['data_subset'],
                                confidence_interval=tuple(h['confidence_interval'])
                            )
                        )
                
                self.logger.info(f"Loaded feature importance history for {len(self.importance_history)} features")
            except Exception as e:
                self.logger.error(f"Error loading history: {e}")
    
    def suggest_feature_engineering(self, 
                                  top_n_interactions: int = 5) -> List[Dict[str, Any]]:
        """Suggest new features based on interactions and importance patterns
        
        Args:
            top_n_interactions: Number of top interactions to consider
            
        Returns:
            List of feature engineering suggestions
        """
        suggestions = []
        
        # Get top features and their analyses
        top_features = self.get_top_features(20)
        analyses = self.analyze_feature_trends([f[0] for f in top_features])
        
        # 1. Suggest interaction features
        all_interactions = []
        for feature in top_features[:10]:
            latest_interactions = self._get_latest_interactions(feature[0])
            if latest_interactions:
                for interact_feat, strength in latest_interactions.items():
                    if strength > 0.05:  # Significant interaction
                        all_interactions.append((feature[0], interact_feat, strength))
        
        all_interactions.sort(key=lambda x: x[2], reverse=True)
        
        for feat1, feat2, strength in all_interactions[:top_n_interactions]:
            suggestions.append({
                'type': 'interaction',
                'features': [feat1, feat2],
                'operation': 'multiply',
                'reason': f'Strong interaction detected (strength: {strength:.3f})',
                'expected_importance': strength * 1.5  # Heuristic
            })
        
        # 2. Suggest polynomial features for high-importance features
        for feature, importance in top_features[:5]:
            if importance > 0.1:
                suggestions.append({
                    'type': 'polynomial',
                    'features': [feature],
                    'operation': 'square',
                    'reason': f'High importance feature ({importance:.3f})',
                    'expected_importance': importance * 0.5
                })
        
        # 3. Suggest ratio features for related features
        # This would require domain knowledge - simplified example
        numeric_features = [f for f, _ in top_features if 'ratio' not in f.lower()]
        
        if len(numeric_features) >= 2:
            suggestions.append({
                'type': 'ratio',
                'features': numeric_features[:2],
                'operation': 'divide',
                'reason': 'Create ratio of important features',
                'expected_importance': 0.05
            })
        
        return suggestions