# Automated Trading Bot System

A production-ready multi-strategy trading bot system with ML-enhanced signal generation that integrates with OpenAlgo for executing trades across Indian markets.

## System Architecture

### Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   Automated Trading Bot System               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐     ┌──────────────┐    ┌──────────────┐ │
│  │   FastAPI   │     │  Bot Manager │    │  WebSocket   │ │
│  │   REST API  │◄────┤              │────►│   Client     │ │
│  │  Port 8080  │     │              │    │              │ │
│  └─────────────┘     └──────┬───────┘    └──────────────┘ │
│                             │                              │
│  ┌─────────────────────────┴────────────────────────────┐ │
│  │                    Trading Bots                       │ │
│  ├──────────────┬──────────────┬────────────┬──────────┤ │
│  │Short Straddle│ Iron Condor  │ Volatility │ Momentum │ │
│  │     Bot      │     Bot      │  Expander  │  Rider   │ │
│  └──────┬───────┴──────┬───────┴─────┬──────┴────┬─────┘ │
│         │              │             │           │       │
│  ┌──────┴──────────────┴─────────────┴───────────┴─────┐ │
│  │              Core Infrastructure                     │ │
│  ├────────────┬─────────────┬───────────┬─────────────┤ │
│  │  Database  │  Indicators │ OpenAlgo  │   Config    │ │
│  │  Manager   │   System    │  Client   │  Manager    │ │
│  └────────────┴─────────────┴───────────┴─────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   OpenAlgo API   │
                    │   Port 5000      │
                    └──────────────────┘
```

### Architecture Overview

This system runs as a separate application that communicates with OpenAlgo via its REST API and WebSocket connections. It manages multiple trading bots running different strategies concurrently.

## Features

### Implemented Features ✅

- **Multiple Trading Strategies**:
  - ✅ Short Straddle Bot (Options Selling)
  - ✅ Iron Condor Bot (Options Selling)  
  - ✅ Volatility Expander Bot (Options Buying)
  - ✅ Momentum Rider Bot (Directional Trading)

- **Core Infrastructure**:
  - ✅ Async Bot Manager for concurrent bot execution
  - ✅ Database module with async SQLAlchemy
  - ✅ Live WebSocket feed subscription
  - ✅ Modular indicator system (20+ indicators)
  - ✅ Base bot framework with state management
  - ✅ REST API with real-time monitoring
  - ✅ Centralized configuration system
  - ✅ Risk management framework
  - ✅ Performance tracking and metrics
  - ✅ Comprehensive integration test suite
  - ✅ Jupyter notebook for testing and simulation

- **Technical Indicators**:
  - ✅ Trend: EMA, SMA, MACD, ADX
  - ✅ Momentum: RSI, Stochastic, Williams %R, ROC
  - ✅ Volatility: ATR, Bollinger Bands, Keltner Channels
  - ✅ Volume: OBV, MFI, Volume MA, VWAP
  - ✅ Price Action: Market Structure, Order Blocks, FVGs, Liquidity Zones

- **ML Enhancement Features** (NEW):
  - ✅ ML Ensemble System with Individual Indicator Intelligence
  - ✅ RSI LSTM Model for pattern prediction
  - ✅ Pattern CNN for chart pattern recognition
  - ✅ Adaptive Thresholds RL for dynamic optimization
  - ✅ ML-Enhanced Price Action Validation
  - ✅ Advanced Confirmation System with ML weights
  - ✅ Signal Validator with adaptive thresholds
  - ✅ Integrated confirmation and validation pipeline
  - ✅ Smart bot selection based on market regime
  - ✅ Performance-based weight adjustment
  - ✅ Model training pipeline with ensemble support

### Features Under Development 🚧

- ❌ Historical data loader for training (2020-present)
- ✅ Machine learning model training pipeline (Implemented)
- ❌ Backtesting engine with walk-forward analysis
- ✅ Parameter optimization framework (Enhanced with ML)
- ✅ Real-time model inference engine (Implemented)
- ❌ Advanced position sizing algorithms
- ❌ Multi-timeframe analysis
- ✅ Market regime detection (Implemented)
- ❌ Correlation analysis between bots
- ❌ Advanced Greeks management for options

## System Requirements

- Python 3.10+
- OpenAlgo running on http://localhost:5000
- Access to OpenAlgo database
- Market data subscription

## Installation

```bash
# Clone the repository
cd /Users/amanish05/workspace/algoTrader/OpenAlgo/automated-trading-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Copy the example configuration:
```bash
cp config/config.example.json config/config.json
```

2. Update the configuration with your settings:
- OpenAlgo API credentials
- Database connection details
- Bot capital allocations
- Risk parameters

## Running the System

```bash
# Start the trading bot system
python main.py

# Or run with specific config
python main.py --config config/production.json

# Run in development mode with hot reload
python main.py --dev
```

## API Endpoints

The trading bot system exposes its own API on port 8080:

- `GET /api/status` - System status and health check
- `GET /api/bots` - List all bots and their status
- `GET /api/bots/{bot_name}` - Get specific bot details
- `POST /api/bots/{bot_name}/start` - Start a bot
- `POST /api/bots/{bot_name}/stop` - Stop a bot
- `GET /api/performance` - Get performance metrics
- `GET /api/positions` - Get all open positions
- `GET /api/signals` - Get recent signals

## Project Structure

```
automated-trading-bot/
├── config/                       # Configuration files
│   ├── config.example.json      # Example app configuration
│   ├── trading_params.example.json  # Example trading parameters
│   └── settings.example.json    # Example user settings
├── src/
│   ├── core/                    # Core components
│   │   ├── bot_manager.py      # Manages multiple bots
│   │   ├── database.py         # Async database operations
│   │   └── __init__.py
│   ├── bots/                    # Trading bot implementations
│   │   ├── base_bot.py         # Abstract base bot class
│   │   ├── short_straddle_bot.py
│   │   ├── iron_condor_bot.py
│   │   ├── volatility_expander_bot.py
│   │   └── momentum_rider_bot.py
│   ├── indicators/              # Technical indicators
│   │   ├── base.py             # Base indicator class
│   │   ├── trend.py            # Trend indicators
│   │   ├── momentum.py         # Momentum indicators
│   │   ├── volatility.py       # Volatility indicators
│   │   ├── volume.py           # Volume indicators
│   │   └── composite.py        # Composite indicator manager
│   ├── integrations/            # External integrations
│   │   └── openalgo_client.py  # OpenAlgo API client
│   ├── api/                     # REST API
│   │   ├── app.py              # FastAPI application
│   │   ├── models.py           # Pydantic models
│   │   └── endpoints.py        # API endpoints
│   ├── config/                  # Configuration system
│   │   ├── constants.py        # All constants
│   │   ├── app_config.py       # Application config
│   │   ├── trading_params.py   # Trading parameters
│   │   ├── settings.py         # User settings
│   │   └── config_manager.py   # Config manager
│   └── utils/                   # Utilities
│       ├── logger.py           # Logging setup
│       └── helpers.py          # Helper functions
├── tests/                       # Test suite
│   ├── conftest.py             # Test fixtures
│   ├── integration/            # Integration tests
│   │   ├── test_bot_manager_integration.py
│   │   ├── test_database_integration.py
│   │   ├── test_openalgo_integration.py
│   │   ├── test_config_integration.py
│   │   ├── test_api_integration.py
│   │   ├── test_short_straddle_integration.py
│   │   ├── test_all_bots_integration.py
│   │   ├── test_full_system_integration.py
│   │   └── test_indicator_integration.py
│   └── test_short_straddle_bot.py  # Unit tests
├── notebooks/                   # Jupyter notebooks
│   └── live_trading_test.ipynb # Live trading simulation
├── docs/                        # Documentation
│   ├── API_DOCUMENTATION.md    # API docs
│   └── CONFIGURATION_GUIDE.md  # Config guide
├── logs/                        # Log files
├── data/                        # Local data storage
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
├── run_tests.sh               # Test runner script
└── main.py                    # Application entry point
```

## Integration with OpenAlgo

The bot system communicates with OpenAlgo through:

1. **REST API** - For placing orders, fetching positions, historical data
2. **WebSocket** - For real-time market data
3. **Database** - Direct read access for performance analytics

## Risk Management

Built-in risk controls include:
- Position size limits per bot
- Portfolio-level Greeks limits
- Maximum drawdown controls
- Correlation checks
- Daily loss limits

## Monitoring

- Real-time dashboard at http://localhost:8080/dashboard
- Prometheus metrics exposed at /metrics
- Detailed logging in logs/ directory

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src tests/

# Lint code
flake8 src/

# Format code
black src/
```

## Running the Application

### Prerequisites

1. Ensure OpenAlgo is running on http://localhost:5000
2. Configure the application:
   ```bash
   cd /Users/amanish05/workspace/algoTrader/OpenAlgo/automated-trading-bot
   cp config/config.example.json config/config.json
   cp config/trading_params.example.json config/trading_params.json
   cp config/settings.example.json config/settings.json
   ```
3. Update config files with your API credentials and preferences

### Starting the Application

```bash
# Activate virtual environment
source venv/bin/activate

# Run the application
python main.py

# The API will be available at http://localhost:8080
```

### Using Jupyter to Activate Bots

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open `notebooks/live_trading_test.ipynb`

3. Run the cells to:
   - Connect to the trading system
   - Activate specific bots
   - Monitor real-time performance
   - Feed historical data for training

### Testing with Historical Data

The system supports feeding historical data (from 2020) for training and last month's data for live testing:

```python
# In Jupyter notebook
from src.data.historical_loader import HistoricalDataLoader

loader = HistoricalDataLoader()

# Load training data (2020 to June 2024)
training_data = await loader.load_training_data("NIFTY", "2020-01-01", "2024-06-30")

# Load recent data for testing (last month)
test_data = await loader.load_recent_data("NIFTY", days=30)

# Feed to bot for parameter optimization
await bot_manager.train_bot("ShortStraddleBot", training_data)
```

## Performance Monitoring

Access the performance dashboard to see how each bot is performing:

- API Dashboard: http://localhost:8080/docs
- Performance Metrics: http://localhost:8080/api/performance
- Bot Status: http://localhost:8080/api/bots

## Enhancements Roadmap

### In Progress
- Historical data loader implementation
- Model training pipeline
- Enhanced Jupyter notebook with backtesting

### Planned Enhancements
1. **Machine Learning Integration**
   - Feature engineering pipeline
   - Model training with TensorFlow/PyTorch
   - Real-time inference engine
   - A/B testing framework

2. **Advanced Risk Management**
   - Portfolio-level VaR calculation
   - Stress testing scenarios
   - Dynamic position sizing
   - Correlation-based hedging

3. **Enhanced Trading Strategies**
   - Market regime detection
   - Multi-timeframe analysis
   - Sentiment analysis integration
   - Options Greeks optimization

4. **Infrastructure Improvements**
   - Kubernetes deployment configs
   - Prometheus/Grafana monitoring
   - Redis caching layer
   - Message queue integration

5. **Data Pipeline**
   - Real-time data warehouse
   - Feature store implementation
   - Automated data quality checks
   - Historical data API

## Deployment

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for production deployment guidelines.

## License

Proprietary - See LICENSE file