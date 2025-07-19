"""
Report Versioning System
Manages PNG files to keep only the best or last 2 versions
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Tuple
import hashlib


class ReportVersionManager:
    """
    Manages report versions, keeping only the best or most recent versions
    """
    
    def __init__(self, reports_dir: str = None, max_versions: int = 2):
        """
        Initialize report version manager
        
        Args:
            reports_dir: Directory containing reports
            max_versions: Maximum number of versions to keep per report type
        """
        self.reports_dir = reports_dir or os.path.dirname(os.path.abspath(__file__))
        self.max_versions = max_versions
        self.version_file = os.path.join(self.reports_dir, 'report_versions.json')
        self.archive_dir = os.path.join(self.reports_dir, 'archive')
        
        # Create archive directory if it doesn't exist
        os.makedirs(self.archive_dir, exist_ok=True)
        
        # Load existing version info
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict:
        """Load version information from file"""
        if os.path.exists(self.version_file):
            with open(self.version_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_versions(self):
        """Save version information to file"""
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def _get_file_hash(self, filepath: str) -> str:
        """Calculate file hash to detect changes"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _get_performance_score(self, report_type: str) -> float:
        """
        Get performance score for ranking reports
        Higher score = better performance
        """
        try:
            # Load performance metrics
            metrics_file = os.path.join(self.reports_dir, 'performance_metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Calculate composite score based on report type
                if 'ml_' in report_type:
                    # For ML reports, use model accuracy + sharpe ratio
                    accuracy = metrics.get('ml_accuracy', 0.5)
                    sharpe = metrics.get('sharpe_ratio', 1.0)
                    return accuracy * 0.6 + sharpe * 0.4
                else:
                    # For other reports, use win rate + sharpe ratio
                    win_rate = metrics.get('win_rate', 0.5)
                    sharpe = metrics.get('sharpe_ratio', 1.0)
                    return win_rate * 0.5 + sharpe * 0.5
        except:
            return 0.5  # Default score
    
    def add_report(self, filename: str, report_type: str = None):
        """
        Add a new report and manage versions
        
        Args:
            filename: Name of the PNG file
            report_type: Type of report (e.g., 'performance', 'ml_training')
        """
        if not filename.endswith('.png'):
            return
        
        filepath = os.path.join(self.reports_dir, filename)
        if not os.path.exists(filepath):
            return
        
        # Determine report type from filename if not provided
        if report_type is None:
            if 'ml_' in filename:
                report_type = 'ml_performance'
            elif 'training' in filename:
                report_type = 'training_comparison'
            elif 'detailed' in filename:
                report_type = 'detailed_metrics'
            else:
                report_type = 'general_performance'
        
        # Get file info
        file_hash = self._get_file_hash(filepath)
        file_stats = os.stat(filepath)
        performance_score = self._get_performance_score(report_type)
        
        # Initialize report type if not exists
        if report_type not in self.versions:
            self.versions[report_type] = []
        
        # Check if this exact file already exists
        existing = [v for v in self.versions[report_type] if v['hash'] == file_hash]
        if existing:
            return  # File already tracked
        
        # Add new version info
        version_info = {
            'filename': filename,
            'hash': file_hash,
            'created': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            'size': file_stats.st_size,
            'performance_score': performance_score
        }
        
        self.versions[report_type].append(version_info)
        
        # Sort by performance score (descending) and creation time
        self.versions[report_type].sort(
            key=lambda x: (x['performance_score'], x['created']),
            reverse=True
        )
        
        # Keep only max_versions
        if len(self.versions[report_type]) > self.max_versions:
            # Archive older versions
            for old_version in self.versions[report_type][self.max_versions:]:
                self._archive_file(old_version['filename'])
            
            # Keep only top versions
            self.versions[report_type] = self.versions[report_type][:self.max_versions]
        
        self._save_versions()
    
    def _archive_file(self, filename: str):
        """Move file to archive directory"""
        source = os.path.join(self.reports_dir, filename)
        if os.path.exists(source):
            # Create timestamped archive name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_name = f"{timestamp}_{filename}"
            destination = os.path.join(self.archive_dir, archive_name)
            
            shutil.move(source, destination)
            print(f"Archived: {filename} -> archive/{archive_name}")
    
    def cleanup_reports(self):
        """
        Clean up all PNG files according to versioning rules
        """
        # Get all PNG files in reports directory
        png_files = [f for f in os.listdir(self.reports_dir) 
                    if f.endswith('.png') and os.path.isfile(os.path.join(self.reports_dir, f))]
        
        # Track all files
        for png_file in png_files:
            self.add_report(png_file)
        
        print(f"Report cleanup complete. Kept {self.max_versions} versions per report type.")
    
    def get_current_reports(self) -> Dict[str, List[str]]:
        """Get list of current reports by type"""
        current = {}
        for report_type, versions in self.versions.items():
            current[report_type] = [v['filename'] for v in versions]
        return current
    
    def generate_report_summary(self):
        """Generate summary of current reports"""
        summary = {
            'last_updated': datetime.now().isoformat(),
            'max_versions': self.max_versions,
            'report_types': {}
        }
        
        for report_type, versions in self.versions.items():
            summary['report_types'][report_type] = {
                'count': len(versions),
                'files': [v['filename'] for v in versions],
                'best_score': versions[0]['performance_score'] if versions else 0,
                'latest': versions[0]['created'] if versions else None
            }
        
        # Save summary
        summary_file = os.path.join(self.reports_dir, 'report_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


def main():
    """Run report versioning cleanup"""
    print("="*60)
    print("REPORT VERSION MANAGEMENT")
    print("="*60)
    
    # Initialize version manager
    manager = ReportVersionManager(max_versions=2)
    
    # Clean up reports
    print("\nCleaning up report files...")
    manager.cleanup_reports()
    
    # Generate summary
    print("\nGenerating report summary...")
    summary = manager.generate_report_summary()
    
    print("\nCurrent Reports:")
    for report_type, info in summary['report_types'].items():
        print(f"\n{report_type}:")
        print(f"  Files: {info['count']}")
        for filename in info['files']:
            print(f"    - {filename}")
    
    print("\nReport versioning complete!")


if __name__ == "__main__":
    main()