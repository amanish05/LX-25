# Installation Guide

This guide provides step-by-step instructions for setting up the Automated Trading Bot system.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git
- macOS/Linux/Windows with WSL

## Step 1: System Dependencies

### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install TA-Lib C library
brew install ta-lib

# Install other dependencies
brew install python3 git
```

### Ubuntu/Debian
```bash
# Update package list
sudo apt-get update

# Install TA-Lib C library
sudo apt-get install -y ta-lib

# If ta-lib not found, install from source:
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Install Python and pip
sudo apt-get install -y python3 python3-pip git
```

### Windows (WSL)
Follow the Ubuntu instructions above in WSL.

## Step 2: Clone Repository

```bash
git clone https://github.com/your-org/automated-trading-bot.git
cd automated-trading-bot
```

## Step 3: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

## Step 4: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### If TA-Lib Installation Fails

If you encounter issues with TA-Lib, you can use the pure Python alternative:

```bash
# Remove TA-Lib from requirements
pip uninstall TA-Lib

# Install pandas-ta instead
pip install pandas-ta

# Update imports in code to use pandas-ta
```

## Step 5: Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

Required environment variables:
```env
# API Configuration
OPENALGO_API_KEY=your_api_key_here
OPENALGO_API_SECRET=your_secret_here
OPENALGO_HOST=http://localhost:5000

# Database
DATABASE_URL=sqlite:///db/trading.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading.log

# Trading Configuration
TRADING_MODE=paper  # paper or live
MAX_CAPITAL=1000000
```

## Step 6: Database Setup

```bash
# Create database directory
mkdir -p db

# Initialize database
python scripts/init_db.py
```

## Step 7: Verify Installation

```bash
# Run tests
pytest tests/

# Check specific components
python -c "import pandas; print('Pandas:', pandas.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import talib; print('TA-Lib:', talib.__version__)"
```

## Step 8: Run the System

```bash
# Run in development mode
python main.py --mode development

# Run optimization
python run_optimization.py

# Start API server
uvicorn src.api.main:app --reload
```

## Troubleshooting

### TA-Lib Import Error
```bash
# macOS: Ensure TA-Lib is in the correct path
export TA_LIBRARY_PATH=/opt/homebrew/lib
export TA_INCLUDE_PATH=/opt/homebrew/include

# Then reinstall
pip uninstall TA-Lib
pip install TA-Lib
```

### Permission Errors
```bash
# Use --user flag for user installation
pip install --user -r requirements.txt
```

### Memory Issues
```bash
# Install dependencies one by one
cat requirements.txt | xargs -n 1 pip install
```

## Docker Installation (Alternative)

```bash
# Build Docker image
docker build -t trading-bot .

# Run container
docker-compose up -d
```

## Next Steps

1. Configure your broker API credentials
2. Run parameter optimization: `python run_optimization.py`
3. Start with paper trading mode
4. Monitor logs in `logs/` directory
5. Check reports in `reports/` directory

## Updating

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade
```

---

**Note**: Always test in paper trading mode before using real money!