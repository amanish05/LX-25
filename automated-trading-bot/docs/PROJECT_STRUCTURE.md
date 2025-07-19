# Project Structure Documentation

This document provides a comprehensive overview of the project structure, explaining the purpose of each directory and important files.

## 🎯 **Core Principle: No Duplication**
Each directory has a specific, unique purpose. No functionality should be duplicated across directories.

## 📁 Root Directory Structure

```
automated-trading-bot/
├── src/                    # Source code - implementation only (NO scripts)
├── config/                 # Configuration files (JSON settings)
├── tests/                  # All testing code (unit, integration, validation)
├── reports/               # Performance reports & visualizations (versioned)
├── docs/                  # All documentation (except README.md)
├── notebooks/             # Jupyter notebooks for analysis
├── data/                  # Market data storage (currently empty, future use)
├── models/                # Trained ML models (*.pkl files, NOT code)
├── logs/                  # Application logs
├── db/                    # SQLite database files
├── scripts/               # Setup and utility scripts only
└── dev_temp/              # TEMPORARY development work (DELETE when done)
    ├── scripts/           # Debug/optimization scripts
    ├── configs/           # Test configurations
    └── reports/           # Analysis reports
```

## 🔍 Detailed Directory Breakdown

### `/src` - Source Code
The heart of the application containing all trading logic. NO standalone scripts here - all implementation code only.

```
src/
├── bots/                          # Trading bot implementations
│   ├── base_bot.py               # Abstract base class for all bots
│   ├── momentum_rider_bot.py     # Option-buying: Quick momentum trades
│   ├── volatility_expander_bot.py # Option-buying: IV expansion trades
│   ├── short_straddle_bot.py     # Option-selling: Premium collection
│   └── iron_condor_bot.py        # Option-selling: Range-bound strategy
│
├── indicators/                    # Technical indicators
│   ├── momentum.py               # Basic momentum calculations
│   ├── volatility.py             # Volatility indicators (Bollinger, ATR)
│   ├── reversal.py               # Reversal pattern detection
│   ├── rsi_advanced.py           # RSI with divergence detection (NEW)
│   ├── oscillator_matrix.py      # Multi-oscillator analysis (NEW)
│   ├── advanced_confirmation.py   # Multi-layer signal confirmation
│   ├── signal_validator.py       # False positive filtering
│   ├── market_structure.py       # LuxAlgo market structure analysis
│   ├── order_blocks.py           # Institutional order flow detection
│1   ├── fair_value_gaps.py        # Price inefficiency identification
│   ├── liquidity_zones.py        # Liquidity concentration areas
│   ├── pattern_recognition.py    # Chart pattern detection
│   └── price_action_composite.py # Unified price action signals
│
├── analysis/                      # Performance analysis tools
│   └── indicator_performance_analyzer.py  # Analyzes indicator effectiveness
│       # Purpose: Tests indicators across different market scenarios
│       # - Evaluates indicators in trending/ranging/volatile markets
│       # - Generates performance heatmaps
│       # - Finds optimal indicator combinations
│       # - Used by optimization system to improve parameters
│
├── optimization/                  # Parameter optimization and ML training
│   ├── bot_parameter_optimizer.py # Bot parameter optimization logic
│   ├── run_optimization.py       # Parameter optimization script
│   └── model_training_pipeline.py # ML model training (called by deployment)
│
├── config/                        # Configuration management
│   ├── __init__.py               # Config module initialization
│   └── constants.py              # System-wide constants
│
├── models/                        # Database ORM models (SQLAlchemy)
│   ├── trade.py                  # Trade data model
│   ├── position.py               # Position tracking model
│   └── performance.py            # Performance metrics model
│   # NOTE: ML models are stored in /models directory at root
│
├── services/                      # External service integrations
│   ├── openalgo_service.py       # OpenAlgo broker integration
│   ├── data_service.py           # Market data service
│   └── notification_service.py   # Alert/notification service
│
├── api/                          # REST API endpoints
│   ├── models.py                 # Pydantic models for API (NOT ML models)
│   ├── bot_routes.py             # Bot control endpoints
│   ├── performance_routes.py     # Performance data endpoints
│   └── config_routes.py          # Configuration endpoints
│
├── utils/                        # Utility functions
│   ├── logger.py                 # Logging configuration
│   ├── risk_manager.py           # Risk management utilities
│   └── market_utils.py           # Market-related utilities
│
└── main.py                       # Application entry point
```

### `/config` - Configuration Files

```
config/
├── trading_config.json           # Main trading configuration
├── optimized_trading_params.json # Production-ready parameters
├── optimal_bot_parameters.json   # Bot-specific optimal parameters (NEW)
├── price_action_fine_tuned.json  # Optimized price action parameters
└── price_action_optimized.json   # Price action configuration
```

**Purpose**: Centralized configuration management
- `trading_config.json`: System settings, API endpoints, logging
- `optimized_trading_params.json`: Trading parameters by bot/symbol
- `optimal_bot_parameters.json`: Results from parameter optimization

### `/tests` - Test Suite

```
tests/
├── unit/                         # Unit tests for individual components
│   ├── test_indicators.py        # Test indicator calculations
│   ├── test_bots.py             # Test bot logic
│   ├── test_rsi_advanced.py     # Test RSI indicator
│   ├── test_oscillator_matrix.py # Test Oscillator Matrix
│   ├── test_market_structure.py  # Test Market Structure (NEW)
│   ├── test_order_blocks.py      # Test Order Blocks (NEW)
│   ├── test_fair_value_gaps.py   # Test Fair Value Gaps (NEW)
│   ├── test_price_action_composite.py # Test PA Composite (NEW)
│   └── test_utils.py            # Test utility functions
│
├── integration/                  # Integration tests
│   ├── test_bot_manager.py      # Test bot orchestration
│   ├── test_openalgo.py         # Test broker integration
│   └── test_full_system.py      # End-to-end tests
│
├── performance/                  # Performance tests
│   └── test_enhanced_system.py   # Compare basic vs enhanced system
│
├── scripts/                      # Test utility scripts (NEW)
│   └── run_tests.sh             # Comprehensive test runner (MOVED)
│
└── validation/                   # System validation tests (MOVED from src)
    └── validate_system.py        # System validation script
```

**Purpose**: Ensure code quality and reliability
- Unit tests: Test individual functions/classes
- Integration tests: Test component interactions
- Performance tests: Compare system improvements

### `/reports` - Performance Reports

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
└── archive/                     # Older PNG versions (auto-archived)
```

**File Purposes**:
- `visualize_performance.py`: Generates basic performance charts
- `visualize_trained_vs_actual.py`: Generates ML comparison charts
- `report_versioning.py`: Manages PNG files (keeps best/last 2 versions)
- `report_manager.py`: Handles report archiving based on performance
- PNG files: Limited to 2 versions per type (auto-archived)

### `/docs` - Documentation

```
docs/
├── BOT_MECHANISMS.md            # Detailed bot working principles
├── PROJECT_STRUCTURE.md         # This file - directory guide
├── CLAUDE.md                    # System context for AI assistance (MOVED from root)
├── INSTALLATION.md              # Installation and setup guide (MOVED from root)
├── SCRIPT_REORGANIZATION.md     # Script organization documentation
├── API_DOCUMENTATION.md         # API endpoint documentation
└── DEPLOYMENT_GUIDE.md          # Production deployment guide
```

**Purpose**: Comprehensive project documentation
- `BOT_MECHANISMS.md`: How each bot works internally
- `PROJECT_STRUCTURE.md`: Directory and file purposes (this file)
- `CLAUDE.md`: Complete system understanding for AI assistance
- `INSTALLATION.md`: Step-by-step setup instructions
- `SCRIPT_REORGANIZATION.md`: Documentation of script organization
- Keep all docs up-to-date with changes

### `/notebooks` - Jupyter Notebooks

```
notebooks/
├── live_trading_enhanced.ipynb  # Live trading analysis
├── backtest_analysis.ipynb      # Backtest result analysis
├── indicator_research.ipynb     # Indicator development/testing
└── performance_analysis.ipynb   # Performance deep-dive
```

**Purpose**: Interactive analysis and research
- Data exploration
- Strategy prototyping
- Performance analysis

## 📄 Important Root Files

### Core Files
- `main.py`: Application entry point
- `requirements.txt`: Python dependencies (consolidated with sections)
- `README.md`: Project overview and setup
- `.gitignore`: Git ignore rules
- `.env`: Environment configuration (API keys, settings)

### Deployment & Operations Scripts
- `run_deployment_pipeline.py`: **MAIN DEPLOYMENT SCRIPT** - Consolidated workflow (NEW)

### Utility Scripts Directory (`/scripts`)
- `setup.py`: Package setup configuration (MOVED)

### Container & Deployment
- `Dockerfile`: Container configuration
- `docker-compose.yml`: Multi-container setup

## 🔄 Deployment Pipeline Integration

The `run_deployment_pipeline.py` orchestrates everything:

```
1. Optimization → src/optimization/run_optimization.py
2. Validation → tests/validation/validate_system.py  
3. Testing → tests/scripts/run_tests.sh
4. ML Training → src/optimization/model_training_pipeline.py
5. Reports → reports/visualize_*.py + report versioning
```

**Everything runs through deployment pipeline - no standalone execution needed.**

## 📝 File Workflow

### 1. **Main Deployment Flow (RECOMMENDED)**
```
Development Changes → ./run_deployment_pipeline.py → Production Ready
```
**Pipeline Steps**: Optimization → Validation → Tests → Training → Reports

### 2. Development Flow
```
src/indicators/new_indicator.py → tests/unit/test_new_indicator.py → docs/update
```

### 3. Individual Component Flows
```
# Optimization Flow
src/optimization/run_optimization.py → config/optimal_bot_parameters.json → reports/bot_optimization_report.md

# Validation Flow  
tests/validation/validate_system.py → reports/system_validation_report.md

# Testing Flow
./tests/scripts/run_tests.sh → Test Results → Coverage Reports

# Performance Tracking Flow
tests/performance/ → reports/visualize_performance.py → reports/PERFORMANCE_SUMMARY.md
```

### 4. Report Management Flow
```
New Report → report_manager.py → Compare with performance_metrics.json → Keep/Archive
```

## 🔑 Key Distinctions

### Model Files Locations
- `/src/api/models.py` → API request/response models (Pydantic)
- `/src/models/` → Database ORM models (SQLAlchemy)
- `/models/` → Trained ML models (pickle files)

### Optimization vs Data
- `/src/optimization/` → Code for optimization and training
- `/data/` → Storage for market data (future use)
- `/models/` → Output of training (serialized models)

### Testing Organization
- `/tests/validation/` → System validation (moved from src)
- `/tests/scripts/` → Test utilities (run_tests.sh)
- `/tests/unit/` → Unit tests for all components

### Report Management
- Maximum 2 PNG files per report type
- Older versions auto-archived to `/reports/archive/`
- Versioning tracked in `report_versions.json`

## 📋 File Naming Conventions

### Code Files
- Snake_case: `momentum_rider_bot.py`
- Descriptive names: `signal_validator.py`
- Test prefix: `test_<module_name>.py`

### Configuration Files
- JSON format: `trading_config.json`
- Environment: `.env`, `.env.example`

### Reports
- Markdown summaries: `UPPERCASE_SUMMARY.md`
- Versioned reports: `report_v1.md`
- Timestamped archives: `version_20250719/`

### Data Files
- CSV format: `data_YYYYMMDD.csv`
- JSON for structured data: `parameters.json`

## 🚫 What NOT to Create

1. **Files in Root Directory**: Never create debug/optimization files in root
2. **Duplicate Reports**: Use `report_manager.py` to version
3. **Redundant Configs**: Keep one source of truth
4. **Test Data in src/**: Keep test data in tests/
5. **Hardcoded Credentials**: Use environment variables

## ⚠️ CRITICAL: Temporary Work Protocol

**ALWAYS use `dev_temp/` for**:
- Debug scripts (`diagnose_*.py`, `debug_*.py`)
- Optimization attempts (`optimize_*.py`, `test_*.py`)
- Intermediate configs (`*_backup.json`, `*_temp.json`)
- Analysis reports (`*_results.md`, `*_analysis.md`)

**NEVER leave temporary files in root directory!**

**After completing work**:
1. Move ONLY final working files to proper locations
2. **DELETE entire `dev_temp/` directory**
3. Update documentation with learnings

## 🔧 Maintenance Guidelines

### Daily
- Check logs/ for errors
- Monitor report generation

### Weekly
- Review and clean reports/
- Update BOT_MECHANISMS.md if changes made
- Run test suite

### Monthly
- Archive old reports
- Update documentation
- Review file structure

## 📊 Quick Reference

| What | Where | Purpose |
|------|-------|---------|
| ML Training Code | `/src/optimization/model_training_pipeline.py` | Implementation |
| Trained ML Models | `/models/*.pkl` | Serialized models |
| API Data Models | `/src/api/models.py` | Pydantic models |
| Database Models | `/src/models/*.py` | SQLAlchemy ORM |
| Market Data | `/data/` | Future storage |
| Config Files | `/config/*.json` | All settings |
| Reports/Charts | `/reports/*.png` | Max 2 per type |
| Old Reports | `/reports/archive/` | Auto-archived |
| Test Scripts | `/tests/scripts/` | Test utilities |
| Validation | `/tests/validation/` | System checks |

## ❌ Common Anti-Patterns to Avoid

1. **DON'T**: Create ML model classes in `/src/api/models.py`
   - **DO**: Use `/src/api/models.py` for API models only

2. **DON'T**: Put training scripts in root directory
   - **DO**: Put them in `/src/optimization/`

3. **DON'T**: Store data in `/src/data/`
   - **DO**: Use `/data/` directory in root

4. **DON'T**: Mix test scripts with implementation
   - **DO**: Keep all tests under `/tests/`

5. **DON'T**: Create duplicate model directories
   - **DO**: Use `/models/` for ML models, `/src/models/` for ORM

## 📋 Protocol Maintenance History

### Script Reorganization (2025-07-19) ✅

**Scripts moved from root to organized locations**:

| **Old Location** | **New Location** | **Reason** |
|-----------------|------------------|------------|
| `run_optimization.py` | `src/optimization/run_optimization.py` | Groups with optimization logic |
| `validate_system.py` | `tests/validation/validate_system.py` | Validation is a form of testing |
| `run_tests.sh` | `tests/scripts/run_tests.sh` | Groups with test utilities |
| `setup.py` | `scripts/setup.py` | Utility/setup scripts |

### Documentation Consolidation (2025-07-19) ✅

**Documentation moved to `/docs`**:
- `INSTALLATION.md` moved from root to `/docs/`
- `CLAUDE.md` moved from root to `/docs/`
- Added detailed `/analysis/` directory explanation
- Consolidated all maintenance protocols into this file

### Environment Configuration (2025-07-19) ✅

**Changes made**:
- Added `OPENALGO_API_SECRET` to `.env`
- Removed `TRADING_MODE` (controlled in OpenAlgo)
- Updated `.env.example` template
- Note: Trading mode (paper/live) is controlled in OpenAlgo, not here

## 🔧 Protocols to Maintain

1. **No Scripts in Root**: Only `main.py` and `run_deployment_pipeline.py` remain
2. **Documentation in /docs**: All .md files except README.md go in docs/
3. **Testing in /tests**: All testing-related scripts including validation
4. **Source in /src**: All implementation code including optimization
5. **Clean Separation**: Implementation vs Testing vs Documentation
6. **Report Management**: Auto-cleanup keeps only best/last 2 PNGs per type
7. **Deployment Pipeline**: Always use `./run_deployment_pipeline.py` for deployments

## 📋 Future Considerations

1. **Trading Mode Cleanup**: Consider removing `paper_trading_mode` references from:
   - `src/config/config_manager.py`
   - `src/config/trading_params.py`

2. **Test Dependencies**: Fix import issues in test suite for full pipeline integration

3. **Performance Dashboard**: Create comprehensive performance tracking dashboard

---

**Last Updated**: 2025-07-19
**Maintained By**: Development Team
**Review Frequency**: Monthly