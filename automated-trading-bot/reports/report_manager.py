"""
Report Manager - Handles performance report versioning
Only keeps reports if they show improvement over previous versions
"""

import json
import os
import shutil
from datetime import datetime
from typing import Dict, Optional

class ReportManager:
    """Manages performance reports with version control"""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        self.metrics_file = os.path.join(reports_dir, "performance_metrics.json")
        self.archive_dir = os.path.join(reports_dir, "archive")
        
        # Create directories if not exist
        os.makedirs(self.archive_dir, exist_ok=True)
        
        # Key metrics to track for comparison
        self.comparison_metrics = [
            "win_rate",
            "sharpe_ratio", 
            "max_drawdown",
            "false_positive_rate",
            "total_return"
        ]
        
    def load_current_metrics(self) -> Optional[Dict]:
        """Load current performance metrics"""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_metrics(self, metrics: Dict):
        """Save performance metrics"""
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['version'] = self._get_next_version()
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def compare_performance(self, new_metrics: Dict, old_metrics: Dict) -> Dict[str, bool]:
        """Compare new metrics against old metrics"""
        comparison = {}
        
        # Win rate - higher is better
        comparison['win_rate'] = new_metrics.get('win_rate', 0) > old_metrics.get('win_rate', 0)
        
        # Sharpe ratio - higher is better
        comparison['sharpe_ratio'] = new_metrics.get('sharpe_ratio', 0) > old_metrics.get('sharpe_ratio', 0)
        
        # Max drawdown - lower (less negative) is better
        comparison['max_drawdown'] = new_metrics.get('max_drawdown', -100) > old_metrics.get('max_drawdown', -100)
        
        # False positive rate - lower is better
        comparison['false_positive_rate'] = new_metrics.get('false_positive_rate', 100) < old_metrics.get('false_positive_rate', 100)
        
        # Total return - higher is better
        comparison['total_return'] = new_metrics.get('total_return', 0) > old_metrics.get('total_return', 0)
        
        return comparison
    
    def is_improvement(self, comparison: Dict[str, bool]) -> bool:
        """Determine if new metrics show overall improvement"""
        # Require improvement in at least 3 out of 5 key metrics
        improvements = sum(1 for v in comparison.values() if v)
        return improvements >= 3
    
    def update_report(self, new_metrics: Dict, report_files: list) -> bool:
        """
        Update reports only if performance improves
        
        Args:
            new_metrics: New performance metrics
            report_files: List of report files to update
            
        Returns:
            bool: True if reports were updated, False otherwise
        """
        current_metrics = self.load_current_metrics()
        
        # If no previous metrics, save new ones
        if not current_metrics:
            self.save_metrics(new_metrics)
            self._copy_report_files(report_files)
            print("✓ Initial performance metrics saved")
            return True
        
        # Compare performance
        comparison = self.compare_performance(new_metrics, current_metrics)
        
        # Check if improvement
        if self.is_improvement(comparison):
            # Archive old reports
            self._archive_current_reports()
            
            # Save new metrics
            self.save_metrics(new_metrics)
            
            # Copy new report files
            self._copy_report_files(report_files)
            
            # Print improvement summary
            self._print_improvement_summary(comparison, new_metrics, current_metrics)
            
            return True
        else:
            print("✗ New report shows no significant improvement - keeping existing reports")
            self._print_comparison_details(comparison, new_metrics, current_metrics)
            return False
    
    def _get_next_version(self) -> int:
        """Get next version number"""
        current = self.load_current_metrics()
        if current:
            return current.get('version', 0) + 1
        return 1
    
    def _archive_current_reports(self):
        """Archive current reports"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_subdir = os.path.join(self.archive_dir, f"version_{timestamp}")
        os.makedirs(archive_subdir, exist_ok=True)
        
        # Move current files to archive
        for file in os.listdir(self.reports_dir):
            if file.endswith(('.png', '.md', '.json')) and file != 'performance_metrics.json':
                src = os.path.join(self.reports_dir, file)
                dst = os.path.join(archive_subdir, file)
                if os.path.isfile(src):
                    shutil.move(src, dst)
    
    def _copy_report_files(self, files: list):
        """Copy new report files to reports directory"""
        for file in files:
            if os.path.exists(file):
                filename = os.path.basename(file)
                dst = os.path.join(self.reports_dir, filename)
                shutil.copy2(file, dst)
    
    def _print_improvement_summary(self, comparison: Dict, new_metrics: Dict, old_metrics: Dict):
        """Print improvement summary"""
        print("\n✓ Performance Improved! Updating reports...")
        print("=" * 50)
        
        for metric, improved in comparison.items():
            old_val = old_metrics.get(metric, 0)
            new_val = new_metrics.get(metric, 0)
            
            if metric in ['win_rate', 'sharpe_ratio', 'total_return']:
                change = new_val - old_val
                symbol = "↑" if improved else "↓"
            else:  # For metrics where lower is better
                change = old_val - new_val
                symbol = "↓" if improved else "↑"
            
            status = "✓" if improved else "✗"
            print(f"{status} {metric}: {old_val:.2f} → {new_val:.2f} ({symbol} {abs(change):.2f})")
    
    def _print_comparison_details(self, comparison: Dict, new_metrics: Dict, old_metrics: Dict):
        """Print detailed comparison"""
        print("\nPerformance Comparison:")
        print("=" * 50)
        
        for metric, improved in comparison.items():
            old_val = old_metrics.get(metric, 0)
            new_val = new_metrics.get(metric, 0)
            status = "✓" if improved else "✗"
            print(f"{status} {metric}: {old_val:.2f} → {new_val:.2f}")


def generate_and_compare_report(metrics: Dict, report_files: list):
    """
    Generate report and compare with existing
    
    Example usage:
    metrics = {
        'win_rate': 64.7,
        'sharpe_ratio': 1.48,
        'max_drawdown': -14.3,
        'false_positive_rate': 12.0,
        'total_return': 37.2
    }
    
    report_files = [
        'training_vs_actual_performance.png',
        'detailed_performance_metrics.png',
        'PERFORMANCE_SUMMARY.md'
    ]
    
    generate_and_compare_report(metrics, report_files)
    """
    manager = ReportManager()
    updated = manager.update_report(metrics, report_files)
    
    if updated:
        print(f"\n✓ Reports updated to version {manager._get_next_version() - 1}")
    else:
        current = manager.load_current_metrics()
        print(f"\n✓ Keeping existing version {current.get('version', 1)} with better performance")
    
    return updated


if __name__ == "__main__":
    # Example usage
    test_metrics = {
        'win_rate': 64.7,
        'sharpe_ratio': 1.48,
        'max_drawdown': -14.3,
        'false_positive_rate': 12.0,
        'total_return': 37.2
    }
    
    test_files = []  # Add actual file paths when running
    
    generate_and_compare_report(test_metrics, test_files)