"""
Time Series Cross-Validation with Purging and Embargo
Implements robust cross-validation techniques for financial time series
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Generator
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ValidationFold:
    """Container for a single validation fold"""
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    fold_id: int
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: Optional[datetime] = None
    test_end: Optional[datetime] = None


@dataclass
class ValidationResults:
    """Container for cross-validation results"""
    fold_results: List[Dict[str, float]]
    avg_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    predictions: Optional[pd.DataFrame] = None
    best_fold: int = 0
    worst_fold: int = 0


class PurgedKFold(BaseCrossValidator):
    """
    K-Fold cross-validator with purging and embargo for time series
    
    Prevents data leakage by:
    1. Purging: Removing training samples too close to validation
    2. Embargo: Adding gap between train and validation sets
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 purge_days: int = 5,
                 embargo_days: int = 2):
        """Initialize Purged K-Fold validator
        
        Args:
            n_splits: Number of folds
            purge_days: Days to purge before validation set
            embargo_days: Days gap between train and validation
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        
    def split(self, X: pd.DataFrame, y=None, groups=None) -> Generator:
        """Generate indices to split data into training and validation sets
        
        Args:
            X: Features DataFrame with datetime index
            y: Target values (not used but required by sklearn)
            groups: Group labels (not used)
            
        Yields:
            train_idx, val_idx for each fold
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have DatetimeIndex")
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate fold size
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Define validation set
            val_start_idx = i * fold_size
            val_end_idx = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            
            val_indices = indices[val_start_idx:val_end_idx]
            
            # Get validation dates
            val_start_date = X.index[val_start_idx]
            val_end_date = X.index[val_end_idx - 1]
            
            # Define training set with purging and embargo
            embargo_date = val_start_date - timedelta(days=self.embargo_days)
            purge_start_date = val_start_date - timedelta(days=self.purge_days + self.embargo_days)
            purge_end_date = val_end_date + timedelta(days=self.purge_days)
            
            # Training indices: everything except validation and purged regions
            train_mask = (
                ((X.index < purge_start_date) | (X.index > purge_end_date)) &
                (X.index < embargo_date)
            )
            train_indices = indices[train_mask]
            
            yield train_indices, val_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits


class CombinatorialPurgedKFold(BaseCrossValidator):
    """
    Combinatorial Purged Cross-Validation
    
    Generates multiple train/test combinations with purging
    More robust than standard k-fold for financial data
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 n_test_splits: int = 2,
                 purge_days: int = 5,
                 embargo_days: int = 2):
        """Initialize Combinatorial Purged K-Fold
        
        Args:
            n_splits: Total number of splits
            n_test_splits: Number of splits to use for testing
            purge_days: Days to purge around test set
            embargo_days: Embargo period between sets
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        
    def split(self, X: pd.DataFrame, y=None, groups=None) -> Generator:
        """Generate train/test split combinations"""
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have DatetimeIndex")
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Split data into n_splits groups
        split_size = n_samples // self.n_splits
        splits = []
        
        for i in range(self.n_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < self.n_splits - 1 else n_samples
            splits.append((start_idx, end_idx))
        
        # Generate combinations of test splits
        from itertools import combinations
        
        for test_splits in combinations(range(self.n_splits), self.n_test_splits):
            # Combine test splits
            test_indices = []
            for split_idx in test_splits:
                start_idx, end_idx = splits[split_idx]
                test_indices.extend(range(start_idx, end_idx))
            
            test_indices = np.array(test_indices)
            
            # Get test date range for purging
            test_start_date = X.index[test_indices.min()]
            test_end_date = X.index[test_indices.max()]
            
            # Apply purging and embargo
            purge_start = test_start_date - timedelta(days=self.purge_days + self.embargo_days)
            purge_end = test_end_date + timedelta(days=self.purge_days + self.embargo_days)
            
            # Training indices: not in test and not in purge zone
            train_mask = ~np.isin(indices, test_indices)
            train_mask &= (X.index < purge_start) | (X.index > purge_end)
            train_indices = indices[train_mask]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations"""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)


class WalkForwardValidator:
    """
    Walk-Forward Analysis for time series
    
    Train on expanding or rolling window, test on next period
    Most realistic for trading strategy validation
    """
    
    def __init__(self,
                 train_period_days: int = 252,  # 1 year
                 test_period_days: int = 63,     # 3 months
                 step_days: int = 21,            # 1 month
                 expanding: bool = False,
                 min_train_days: int = 126):     # 6 months minimum
        """Initialize Walk-Forward validator
        
        Args:
            train_period_days: Training window size
            test_period_days: Test window size
            step_days: Step size between windows
            expanding: Use expanding window (True) or rolling (False)
            min_train_days: Minimum training days required
        """
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.step_days = step_days
        self.expanding = expanding
        self.min_train_days = min_train_days
        
    def split(self, X: pd.DataFrame, y=None) -> Generator[ValidationFold, None, None]:
        """Generate walk-forward validation folds
        
        Args:
            X: Features DataFrame with datetime index
            y: Target values (optional)
            
        Yields:
            ValidationFold objects
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have DatetimeIndex")
        
        start_date = X.index[0]
        end_date = X.index[-1]
        
        # Initial training end date
        if self.expanding:
            train_start = start_date
        else:
            train_start = start_date
        
        train_end = start_date + timedelta(days=self.train_period_days)
        fold_id = 0
        
        while train_end + timedelta(days=self.test_period_days) <= end_date:
            # Define test period
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.test_period_days - 1)
            
            # Get indices
            train_mask = (X.index >= train_start) & (X.index <= train_end)
            test_mask = (X.index >= test_start) & (X.index <= test_end)
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            # Check minimum training size
            if len(train_indices) >= self.min_train_days and len(test_indices) > 0:
                # Create validation set from end of training
                val_size = min(len(train_indices) // 5, 63)  # 20% of train or 3 months
                val_indices = train_indices[-val_size:]
                train_indices = train_indices[:-val_size]
                
                yield ValidationFold(
                    train_indices=train_indices,
                    val_indices=val_indices,
                    test_indices=test_indices,
                    fold_id=fold_id,
                    train_start=X.index[train_indices[0]],
                    train_end=X.index[train_indices[-1]],
                    val_start=X.index[val_indices[0]],
                    val_end=X.index[val_indices[-1]],
                    test_start=X.index[test_indices[0]],
                    test_end=X.index[test_indices[-1]]
                )
                fold_id += 1
            
            # Move windows
            if not self.expanding:
                train_start += timedelta(days=self.step_days)
            train_end += timedelta(days=self.step_days)


class TimeSeriesValidator:
    """
    Comprehensive time series validation framework
    
    Implements multiple validation strategies:
    - Purged K-Fold
    - Combinatorial Purged K-Fold
    - Walk-Forward Analysis
    """
    
    def __init__(self):
        """Initialize time series validator"""
        self.logger = logging.getLogger(__name__)
        
    def validate_model(self,
                      model: BaseEstimator,
                      X: pd.DataFrame,
                      y: pd.Series,
                      cv_method: str = 'purged_kfold',
                      scoring: str = 'accuracy',
                      **cv_params) -> ValidationResults:
        """Validate model using specified cross-validation method
        
        Args:
            model: Sklearn-compatible model
            X: Features DataFrame with datetime index
            y: Target series
            cv_method: CV method ('purged_kfold', 'combinatorial', 'walk_forward')
            scoring: Scoring metric
            **cv_params: Parameters for CV method
            
        Returns:
            ValidationResults object
        """
        self.logger.info(f"Starting {cv_method} validation with {scoring} scoring")
        
        # Get cross-validator
        cv = self._get_cv_method(cv_method, **cv_params)
        
        # Store results
        fold_results = []
        all_predictions = []
        
        # Perform cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            # Handle different CV types
            if isinstance(cv, WalkForwardValidator):
                # Walk-forward returns ValidationFold objects
                fold = train_idx  # Actually a ValidationFold object
                train_idx = fold.train_indices
                val_idx = fold.val_indices
                test_idx = fold.test_indices
            else:
                test_idx = None
            
            # Train model
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            # Clone model to avoid contamination
            from sklearn.base import clone
            fold_model = clone(model)
            
            # Fit model
            fold_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_val = fold_model.predict(X_val)
            
            # Calculate metrics
            fold_metrics = self._calculate_metrics(y_val, y_pred_val, scoring)
            fold_metrics['fold'] = fold_idx
            
            # Add test set metrics if available
            if test_idx is not None:
                X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
                y_pred_test = fold_model.predict(X_test)
                test_metrics = self._calculate_metrics(y_test, y_pred_test, scoring)
                fold_metrics.update({f'test_{k}': v for k, v in test_metrics.items()})
            
            fold_results.append(fold_metrics)
            
            # Store predictions
            for idx, pred in zip(val_idx, y_pred_val):
                all_predictions.append({
                    'index': X.index[idx],
                    'fold': fold_idx,
                    'y_true': y.iloc[idx],
                    'y_pred': pred,
                    'set': 'validation'
                })
            
            if test_idx is not None:
                for idx, pred in zip(test_idx, y_pred_test):
                    all_predictions.append({
                        'index': X.index[idx],
                        'fold': fold_idx,
                        'y_true': y.iloc[idx],
                        'y_pred': pred,
                        'set': 'test'
                    })
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame(all_predictions)
        
        # Calculate aggregate metrics
        avg_metrics, std_metrics = self._aggregate_metrics(fold_results)
        
        # Find best and worst folds
        primary_metric = scoring if scoring in fold_results[0] else list(fold_results[0].keys())[0]
        fold_scores = [f[primary_metric] for f in fold_results]
        best_fold = np.argmax(fold_scores)
        worst_fold = np.argmin(fold_scores)
        
        return ValidationResults(
            fold_results=fold_results,
            avg_metrics=avg_metrics,
            std_metrics=std_metrics,
            predictions=predictions_df,
            best_fold=best_fold,
            worst_fold=worst_fold
        )
    
    def _get_cv_method(self, method: str, **kwargs):
        """Get cross-validation method instance"""
        if method == 'purged_kfold':
            return PurgedKFold(**kwargs)
        elif method == 'combinatorial':
            return CombinatorialPurgedKFold(**kwargs)
        elif method == 'walk_forward':
            return WalkForwardValidator(**kwargs)
        else:
            raise ValueError(f"Unknown CV method: {method}")
    
    def _calculate_metrics(self, 
                         y_true: pd.Series, 
                         y_pred: np.ndarray,
                         primary_metric: str) -> Dict[str, float]:
        """Calculate various metrics for predictions"""
        metrics = {}
        
        # Classification metrics
        if primary_metric in ['accuracy', 'precision', 'recall', 'f1']:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Regression metrics
        elif primary_metric in ['mse', 'mae', 'r2', 'rmse']:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
        
        # Trading-specific metrics
        if hasattr(y_pred, '__len__'):
            # Calculate returns-based metrics
            if len(y_true) > 1:
                # Assuming y represents returns or signals
                if primary_metric == 'sharpe':
                    returns = y_true * y_pred  # Signal * actual returns
                    metrics['sharpe'] = self._calculate_sharpe(returns)
                    metrics['total_return'] = returns.sum()
                    metrics['win_rate'] = (returns > 0).mean()
                    metrics['max_drawdown'] = self._calculate_max_drawdown(returns.cumsum())
        
        return metrics
    
    def _calculate_sharpe(self, returns: pd.Series, periods: int = 252) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        return np.sqrt(periods) * returns.mean() / returns.std()
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(cumulative_returns) == 0:
            return 0
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / (running_max + 1e-8)
        return drawdown.min()
    
    def _aggregate_metrics(self, 
                         fold_results: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Aggregate metrics across folds"""
        # Get all metric names
        all_metrics = set()
        for fold in fold_results:
            all_metrics.update(fold.keys())
        
        # Remove fold identifier
        all_metrics.discard('fold')
        
        # Calculate mean and std
        avg_metrics = {}
        std_metrics = {}
        
        for metric in all_metrics:
            values = [fold.get(metric, np.nan) for fold in fold_results]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                avg_metrics[metric] = np.mean(values)
                std_metrics[metric] = np.std(values)
            else:
                avg_metrics[metric] = np.nan
                std_metrics[metric] = np.nan
        
        return avg_metrics, std_metrics
    
    def plot_validation_results(self, 
                              results: ValidationResults,
                              save_path: Optional[str] = None):
        """Plot validation results across folds"""
        import matplotlib.pyplot as plt
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        # 1. Metrics across folds
        fold_df = pd.DataFrame(results.fold_results)
        metrics_to_plot = [col for col in fold_df.columns if col != 'fold' and not col.startswith('test_')][:4]
        
        if metrics_to_plot:
            ax = axes[0]
            fold_df[metrics_to_plot].plot(kind='bar', ax=ax)
            ax.set_xlabel('Fold')
            ax.set_ylabel('Score')
            ax.set_title('Validation Metrics Across Folds')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        # 2. Average metrics with error bars
        if results.avg_metrics and results.std_metrics:
            ax = axes[1]
            metrics = list(results.avg_metrics.keys())[:6]
            means = [results.avg_metrics[m] for m in metrics]
            stds = [results.std_metrics[m] for m in metrics]
            
            x_pos = np.arange(len(metrics))
            ax.bar(x_pos, means, yerr=stds, capsize=5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metrics, rotation=45)
            ax.set_ylabel('Score')
            ax.set_title('Average Metrics with Standard Deviation')
            ax.grid(True, alpha=0.3)
        
        # 3. Predictions vs Actual (if classification)
        if results.predictions is not None and len(results.predictions) > 0:
            ax = axes[2]
            val_preds = results.predictions[results.predictions['set'] == 'validation']
            
            if len(val_preds) > 0:
                # Sample for visibility
                sample_size = min(1000, len(val_preds))
                sample = val_preds.sample(sample_size)
                
                ax.scatter(sample.index, sample['y_true'], alpha=0.5, label='Actual', s=10)
                ax.scatter(sample.index, sample['y_pred'], alpha=0.5, label='Predicted', s=10)
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Value')
                ax.set_title('Predictions vs Actual (Sample)')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 4. Performance timeline (for walk-forward)
        if 'test_accuracy' in fold_df.columns or 'test_sharpe' in fold_df.columns:
            ax = axes[3]
            test_metrics = [col for col in fold_df.columns if col.startswith('test_')][:2]
            
            if test_metrics:
                for metric in test_metrics:
                    ax.plot(fold_df.index, fold_df[metric], marker='o', label=metric)
                ax.set_xlabel('Fold (Time)')
                ax.set_ylabel('Score')
                ax.set_title('Test Performance Over Time')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def generate_validation_report(self, 
                                 results: ValidationResults,
                                 model_name: str,
                                 output_path: str = "reports/validation_report.md"):
        """Generate comprehensive validation report
        
        Args:
            results: ValidationResults object
            model_name: Name of the model
            output_path: Path to save report
        """
        report_lines = [
            f"# Time Series Validation Report - {model_name}",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Summary Statistics\n"
        ]
        
        # Overall metrics
        report_lines.append("### Average Performance Across Folds\n")
        report_lines.append("| Metric | Mean | Std Dev |")
        report_lines.append("|--------|------|---------|")
        
        for metric in sorted(results.avg_metrics.keys()):
            if not metric.startswith('test_'):
                mean_val = results.avg_metrics[metric]
                std_val = results.std_metrics[metric]
                report_lines.append(f"| {metric} | {mean_val:.4f} | {std_val:.4f} |")
        
        # Test set performance (if available)
        test_metrics = {k: v for k, v in results.avg_metrics.items() if k.startswith('test_')}
        if test_metrics:
            report_lines.extend([
                "\n### Test Set Performance\n",
                "| Metric | Mean | Std Dev |",
                "|--------|------|---------|"
            ])
            
            for metric in sorted(test_metrics.keys()):
                mean_val = results.avg_metrics[metric]
                std_val = results.std_metrics[metric]
                clean_name = metric.replace('test_', '')
                report_lines.append(f"| {clean_name} | {mean_val:.4f} | {std_val:.4f} |")
        
        # Fold-by-fold results
        report_lines.extend([
            "\n## Detailed Fold Results\n",
            "### Validation Performance by Fold\n"
        ])
        
        # Create fold results table
        fold_df = pd.DataFrame(results.fold_results)
        report_lines.append(fold_df.to_markdown())
        
        # Best and worst folds
        report_lines.extend([
            f"\n### Performance Analysis\n",
            f"- **Best Fold**: {results.best_fold}",
            f"- **Worst Fold**: {results.worst_fold}",
            f"- **Performance Range**: {fold_df.iloc[:, 1:].max().max():.4f} - {fold_df.iloc[:, 1:].min().min():.4f}"
        ])
        
        # Stability analysis
        cv_scores = []
        for metric in ['accuracy', 'f1', 'sharpe']:
            if metric in results.std_metrics:
                cv = results.std_metrics[metric] / results.avg_metrics[metric] if results.avg_metrics[metric] != 0 else 0
                cv_scores.append(f"{metric}: {cv:.2%}")
        
        if cv_scores:
            report_lines.extend([
                "\n### Stability Analysis (Coefficient of Variation)\n",
                "Lower values indicate more stable performance across folds:"
            ])
            for cv_score in cv_scores:
                report_lines.append(f"- {cv_score}")
        
        # Recommendations
        report_lines.extend([
            "\n## Recommendations\n"
        ])
        
        # Check for overfitting
        if test_metrics:
            val_perf = results.avg_metrics.get('accuracy', results.avg_metrics.get('sharpe', 0))
            test_perf = results.avg_metrics.get('test_accuracy', results.avg_metrics.get('test_sharpe', 0))
            
            if val_perf - test_perf > 0.05:
                report_lines.append("- ⚠️ **Potential Overfitting**: Validation performance significantly exceeds test performance")
            else:
                report_lines.append("- ✅ **Good Generalization**: Consistent performance between validation and test sets")
        
        # Check stability
        avg_cv = np.mean([results.std_metrics[m] / results.avg_metrics[m] 
                         for m in results.avg_metrics if results.avg_metrics[m] != 0])
        
        if avg_cv < 0.1:
            report_lines.append("- ✅ **Stable Model**: Low variance across folds")
        elif avg_cv < 0.2:
            report_lines.append("- ⚠️ **Moderate Stability**: Some variance across folds")
        else:
            report_lines.append("- ❌ **Unstable Model**: High variance across folds - consider ensemble methods")
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Validation report saved to {output_path}")
    
    def compare_models(self,
                      models: Dict[str, BaseEstimator],
                      X: pd.DataFrame,
                      y: pd.Series,
                      cv_method: str = 'purged_kfold',
                      scoring: str = 'accuracy',
                      **cv_params) -> pd.DataFrame:
        """Compare multiple models using same validation strategy
        
        Args:
            models: Dictionary of model_name -> model
            X: Features
            y: Target
            cv_method: Cross-validation method
            scoring: Scoring metric
            **cv_params: CV parameters
            
        Returns:
            DataFrame with comparison results
        """
        comparison_results = []
        
        for model_name, model in models.items():
            self.logger.info(f"Validating {model_name}")
            
            results = self.validate_model(
                model, X, y, cv_method, scoring, **cv_params
            )
            
            # Extract key metrics
            model_summary = {
                'model': model_name,
                **{f'avg_{k}': v for k, v in results.avg_metrics.items()},
                **{f'std_{k}': v for k, v in results.std_metrics.items()}
            }
            
            comparison_results.append(model_summary)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        
        # Sort by primary metric
        sort_col = f'avg_{scoring}'
        if sort_col in comparison_df.columns:
            comparison_df = comparison_df.sort_values(sort_col, ascending=False)
        
        return comparison_df