# Reports Directory

This directory contains performance reports, visualizations, and analysis tools for the automated trading bot.

## Directory Structure

```
reports/
├── visualize_performance.py      # Script to generate performance charts
├── visualize_trained_vs_actual.py # ML performance comparison script  
├── report_manager.py            # Report versioning system
├── report_versioning.py         # PNG version management (keeps best/last 2)
├── performance_metrics.json     # Current best performance metrics
├── model_training_report.json   # ML training results
├── report_versions.json         # Version tracking for PNGs
├── report_summary.json          # Summary of current reports
├── PERFORMANCE_SUMMARY.md       # Latest comprehensive report
├── ML_TRAINING_SUMMARY.md       # ML training documentation
├── *.png                        # Performance visualizations (max 2 per type)
└── archive/                     # Older reports and PNG versions
    └── reports_YYYYMMDD/        # Dated archive folders
```

## Report Management

### Automated Versioning
- PNG files are automatically versioned by `report_versioning.py`
- Only the best performing and most recent versions are kept (max 2 per type)
- Older versions are moved to the archive directory

### Report Types
1. **Performance Reports**: System performance metrics and analysis
2. **ML Training Reports**: Model training results and comparisons
3. **Visualizations**: PNG charts showing various performance metrics

### Running Reports
Reports are automatically generated as part of the deployment pipeline:
```bash
python run_deployment_pipeline.py
```

Or generate reports manually:
```bash
python run_deployment_pipeline.py report
```

### Cleanup
The report directory is automatically cleaned during the deployment pipeline.
Old reports are archived with timestamps for historical reference.

---
**Last Updated**: 2025-07-19
