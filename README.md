# OpenAlgo Trading Platform

This repository contains two main applications for algorithmic trading using the OpenAlgo platform.

## Applications

### 1. OpenAlgo Platform (`openalgo/`)

**Purpose**: Web-based trading platform that provides API endpoints for executing trades, managing positions, and retrieving market data.

**Key Features**:
- Web-based dashboard for trading operations
- REST API for programmatic trading
- WebSocket support for real-time data
- Multi-broker support
- User authentication and session management
- Trade execution and position management

**Technology Stack**:
- **Backend**: Python Flask with SocketIO
- **Database**: PostgreSQL
- **Frontend**: HTML/CSS/JavaScript with DaisyUI

**Database Configuration**:
- **Main Database**: `openalgo` - Stores user data, trades, positions
- **Latency Database**: `openalgo_latency` - Performance monitoring
- **Logs Database**: `openalgo_logs` - Application logs and audit trails

### 2. Automated Trading Bot (`automated-trading-bot/`)

**Purpose**: Automated trading system that connects to OpenAlgo platform to execute trading strategies programmatically.

**Key Features**:
- Automated trading strategy execution
- Real-time market data processing
- Risk management and position sizing
- Performance tracking and analytics
- Multi-bot architecture support
- WebSocket integration for live data

**Technology Stack**:
- **Backend**: Python FastAPI with async/await
- **Database**: PostgreSQL with async SQLAlchemy
- **Integration**: OpenAlgo REST API and WebSocket

**Database Configuration**:
- **Main Database**: `trading_bot` - Bot positions, trades, performance metrics, signals

## System Architecture

```
┌─────────────────┐    HTTP/WebSocket    ┌─────────────────┐
│                 │ ◄─────────────────► │                 │
│  Trading Bot    │                     │   OpenAlgo      │
│  (Port 8080)    │                     │  (Port 5000)    │
│                 │                     │                 │
└─────────────────┘                     └─────────────────┘
         │                                       │
         ▼                                       ▼
┌─────────────────┐                     ┌─────────────────┐
│   PostgreSQL    │                     │   PostgreSQL    │
│   trading_bot   │                     │ openalgo (3 DBs)│
└─────────────────┘                     └─────────────────┘
```

## Prerequisites

- Python 3.9+
- PostgreSQL 17+
- Virtual environment (recommended)

## Database Setup

### PostgreSQL Configuration

1. **Install PostgreSQL 17** (if not already installed)
2. **Create databases**:
   ```sql
   CREATE DATABASE openalgo;
   CREATE DATABASE openalgo_latency;
   CREATE DATABASE openalgo_logs;
   CREATE DATABASE trading_bot;
   ```

3. **Create user and grant permissions** (configure credentials in .env files):
   ```sql
   CREATE USER your_trading_user WITH PASSWORD 'your_secure_password';
   GRANT ALL PRIVILEGES ON DATABASE openalgo TO your_trading_user;
   GRANT ALL PRIVILEGES ON DATABASE openalgo_latency TO your_trading_user;
   GRANT ALL PRIVILEGES ON DATABASE openalgo_logs TO your_trading_user;
   GRANT ALL PRIVILEGES ON DATABASE trading_bot TO your_trading_user;
   ```

## Installation & Setup

### 1. OpenAlgo Platform

```bash
cd openalgo
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Environment Configuration** (`.env`):
```bash
DATABASE_URL=postgresql://your_user:your_password@localhost:5432/openalgo
LATENCY_DATABASE_URL=postgresql://your_user:your_password@localhost:5432/openalgo_latency
LOGS_DATABASE_URL=postgresql://your_user:your_password@localhost:5432/openalgo_logs
```

**Start OpenAlgo**:
```bash
python app.py
```
Access at: http://127.0.0.1:5000

### 2. Automated Trading Bot

```bash
cd automated-trading-bot
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Environment Configuration** (`.env`):
```bash
# OpenAlgo API Configuration
OPENALGO_API_KEY=your_openalgo_api_key_here
OPENALGO_HOST=http://127.0.0.1:5000
OPENALGO_API_URL=http://127.0.0.1:5000/api/v1
OPENALGO_WEBSOCKET_URL=ws://127.0.0.1:8765

# PostgreSQL Database Configuration
DATABASE_URL=postgresql+asyncpg://your_user:your_password@localhost:5432/trading_bot

# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=8080
LOG_LEVEL=DEBUG
MAX_CAPITAL=1000000
ENV=development
```

**Start Trading Bot**:
```bash
python main.py
```
Access at: http://0.0.0.0:8080

## API Documentation

- **OpenAlgo API**: http://127.0.0.1:5000/api/v1/ (Swagger UI)
- **Trading Bot API**: http://0.0.0.0:8080/docs (FastAPI docs)

## Key Dependencies

### OpenAlgo
- Flask 3.1.2
- Flask-SocketIO
- SQLAlchemy
- PostgreSQL adapter (psycopg2)
- WebSocket support

### Automated Trading Bot
- FastAPI
- SQLAlchemy (async)
- asyncpg (PostgreSQL async driver)
- httpx (async HTTP client)
- websockets
- python-dotenv

## Database Schema

### OpenAlgo Databases
- **openalgo**: Users, sessions, broker connections, trade history
- **openalgo_latency**: Performance metrics and latency tracking
- **openalgo_logs**: Application logs and audit trails

### Trading Bot Database
- **bot_positions**: Active and historical positions
- **bot_trades**: Individual trade executions
- **bot_performance**: Daily performance metrics
- **bot_signals**: Trading signals generated
- **bot_capital**: Capital allocation per bot
- **market_data_cache**: Cached market data with TTL

## Trading Configuration

All trades execute on live market data. The trading bot places orders through OpenAlgo, which handles the actual broker integration and order execution. Risk management and trade control are managed at the OpenAlgo platform level.

## Security Notes

- API keys are stored in environment variables
- Database credentials use environment configuration
- No hardcoded sensitive values in code
- HTTPS recommended for production deployment

## Monitoring

- **Trading Bot**: Built-in performance tracking and logging
- **OpenAlgo**: Web dashboard with real-time status
- **Database**: PostgreSQL performance monitoring recommended
- **Logs**: Structured logging with configurable levels

## Support

For issues or questions:
- OpenAlgo Documentation: https://docs.openalgo.in
- GitHub Issues: Create issues in respective repositories