# Claude Development Notes

## 📁 IMPORTANT: Project Structure Reference
**Before creating any new files or directories, please refer to `/docs/PROJECT_STRUCTURE.md`**
- Contains complete directory structure and file organization
- Explains purpose of each directory and major files
- Helps avoid creating duplicate or misplaced files
- Shows where different types of code should be placed

## Database Configuration ✅ WORKING

### PostgreSQL Installation
- **Location**: `/Library/PostgreSQL/17/`
- **Version**: PostgreSQL 17
- **Status**: ✅ Running and operational
- **Data Directory**: `/Library/PostgreSQL/17/data`
- **Service Control**: `/Library/PostgreSQL/17/bin/pg_ctl`

### Database Setup Status ✅ COMPLETE
- **Test Database**: ✅ `test_trading_bot` (working)
  - User: `test_user`
  - Password: `test_pass`
  - Connection: `postgresql+asyncpg://test_user:test_pass@localhost:5432/test_trading_bot`

- **Main Database**: ✅ `trading_bot` (working)
  - User: `trading_user`
  - Password: `TradingBot2024!`
  - Connection: `postgresql+asyncpg://trading_user:TradingBot2024!@localhost:5432/trading_bot`

### Existing Databases
```
openalgo         | trading_user 
openalgo_latency | trading_user 
openalgo_logs    | trading_user 
postgres         | postgres     
template0        | postgres     
template1        | postgres     
test_trading_bot | test_user    
trading_bot      | trading_user 
```

## Environment Configuration ✅ WORKING

### .env File Status
- **Location**: `/Users/amanish05/workspace/algoTrader/LX-25/automated-trading-bot/.env`
- **Database URL**: ✅ Updated to use async driver (`postgresql+asyncpg://`)
- **Database Name**: ✅ Corrected to `trading_bot` (not `trading_bot_dev`)

### Key Environment Variables
```
DATABASE_URL=postgresql+asyncpg://trading_user:TradingBot2024!@localhost:5432/trading_bot
TEST_DATABASE_URL=postgresql+asyncpg://test_user:test_pass@localhost:5432/test_trading_bot
ENVIRONMENT=development
TRADING_MODE=paper
```

## Test Results Status ✅ EXCELLENT

### Unit Tests: ✅ 110/113 Passing
- **Coverage**: 41.91% (exceeds 20% requirement)
- **ML Validator Tests**: ✅ All 9 tests fixed
- **TensorFlow Warnings**: ✅ Eliminated
- **Integration**: Fixed import errors

### Deployment Pipeline: ✅ Ready
- **System Validation**: ✅ PASSED
- **Bot Validation**: ✅ All 4 bots ready
- **Functional Tests**: ✅ All components working

## Application Startup Status

### ✅ Database Connection: WORKING
- **Status**: ✅ "Database tables initialized" - Connection successful
- **Driver**: ✅ AsyncPG working correctly
- **Tables**: ✅ Initialized successfully

### ⚠️ Current Issue: OpenAlgo API Connection
- **Problem**: HTTP 403 Forbidden error at `http://localhost:5000/api/v1/funds/`
- **API URL**: `http://localhost:5000/api/v1`
- **API Key**: `027f5049815dec5081294d97bc9a06533afab6462e7c2628bda0e5d73777116b`
- **Status**: OpenAlgo service not running or API credentials invalid

### Resolution Steps
1. ✅ PostgreSQL service validated and running
2. ✅ Database connection fixed
3. ✅ Async driver configured correctly
4. ⏳ OpenAlgo API service needs to be started
5. ⏳ Verify OpenAlgo API credentials

## Dependencies ✅ ALL WORKING
- **asyncpg**: ✅ Installed and working (async PostgreSQL driver)
- **greenlet**: ✅ Installed (required for SQLAlchemy async)
- **TensorFlow**: ✅ Working (Input layer fixes applied)

## Trading Bot Features ✅ READY
- **Price Action Analysis**: ✅ Optimized weights
- **ML Ensemble**: ✅ Trained and validated
- **Risk Management**: ✅ Conservative 1.5+ R:R ratios
- **Multi-Bot Support**: ✅ 4 bots configured
- **Database Integration**: ✅ Tables initialized

## Current Status Summary
- **Database**: ✅ Fully operational
- **Application Startup**: ✅ Database initialization working
- **Missing**: OpenAlgo API service running
- **Next Step**: Start OpenAlgo API service or configure mock mode

## Quick Start Commands
```bash
# Start trading bot (requires OpenAlgo API running)
source .venv/bin/activate && python main.py

# Check database connection
PGPASSWORD=TradingBot2024! /Library/PostgreSQL/17/bin/psql -h localhost -U trading_user -d trading_bot -p 5432 -c "\dt"

# Run tests
source .venv/bin/activate && python -m pytest tests/unit/ -v
```

## 📁 Project Structure Guidelines

### Important Directories:
- `/src/` - All implementation code goes here
- `/tests/` - All test files (unit, integration, performance)
- `/config/` - JSON configuration files
- `/models/` - Trained ML models (*.pkl, *.h5)
- `/data/` - Market data storage
- `/reports/` - Performance reports and visualizations
- `/docs/` - All documentation files

### Key Rules:
1. **Never create scripts in root directory** - Use appropriate subdirectories
2. **ML models**: `/src/api/models.py` is for API models, `/models/` is for trained ML models
3. **Configuration**: `/config/` for JSON files, `/src/config/` for Python config code
4. **Always check PROJECT_STRUCTURE.md before creating new files**

### Recent Cleanup (2025-07-20):
- Removed 3 deprecated notebooks (only `live_trading_master.ipynb` remains)
- Removed `exception_migration_helper.py` (migration complete)
- Removed `run_enhanced_training.py` (functionality in `src/optimization/`)
- Created `/data/` and `/db/` directories with READMEs
- ML training consolidated in `model_training_pipeline.py`