# Data Directory

This directory is for storing market data files.

## Purpose
- Historical market data storage
- Downloaded price data cache
- Processed datasets for ML training
- Real-time data snapshots

## Structure
```
data/
├── historical/      # Historical price data
├── realtime/        # Real-time data snapshots
├── processed/       # ML-ready processed data
└── cache/           # Temporary data cache
```

## Notes
- Files in this directory are gitignored
- Large CSV/parquet files should be stored here
- Use subdirectories to organize by data type