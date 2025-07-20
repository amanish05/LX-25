#!/bin/bash
# Setup development database for automated trading bot

echo "Setting up development database for automated trading bot..."

# Check if PostgreSQL is running
if ! pg_isready -q; then
    echo "❌ PostgreSQL is not running. Please start PostgreSQL first."
    echo ""
    echo "To install PostgreSQL on macOS:"
    echo "  brew install postgresql@14"
    echo "  brew services start postgresql@14"
    echo ""
    echo "To install on Ubuntu:"
    echo "  sudo apt update"
    echo "  sudo apt install postgresql postgresql-contrib"
    echo "  sudo systemctl start postgresql"
    exit 1
fi

# Database configuration
DEV_DB_NAME="trading_bot_dev"
DEV_DB_USER="trading_bot_user"
DEV_DB_PASS="trading_bot_pass"

# Check if we can connect
if [ -z "$PGPASSWORD" ]; then
    if ! psql -U postgres -c '\q' 2>/dev/null; then
        echo "❌ Cannot connect to PostgreSQL. Please either:"
        echo "   1. Set PGPASSWORD environment variable"
        echo "   2. Configure PostgreSQL for trust authentication"
        echo "   3. Run with sudo: sudo -u postgres $0"
        exit 1
    fi
fi

# Create development user if not exists
echo "Creating development user..."
psql -U postgres -tc "SELECT 1 FROM pg_roles WHERE rolname='$DEV_DB_USER'" | grep -q 1 || \
psql -U postgres -c "CREATE USER $DEV_DB_USER WITH PASSWORD '$DEV_DB_PASS';"

# Create development database if not exists
echo "Creating development database..."
psql -U postgres -tc "SELECT 1 FROM pg_database WHERE datname='$DEV_DB_NAME'" | grep -q 1 || \
psql -U postgres -c "CREATE DATABASE $DEV_DB_NAME OWNER $DEV_DB_USER;"

# Grant privileges
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE $DEV_DB_NAME TO $DEV_DB_USER;"

# Create schema
export DATABASE_URL="postgresql://$DEV_DB_USER:$DEV_DB_PASS@localhost:5432/$DEV_DB_NAME"

echo "✅ Development database setup complete!"
echo ""
echo "Database Configuration:"
echo "  Database: $DEV_DB_NAME"
echo "  User: $DEV_DB_USER"
echo "  Password: $DEV_DB_PASS"
echo ""
echo "Add to your .env file:"
echo "  DATABASE_URL=postgresql://$DEV_DB_USER:$DEV_DB_PASS@localhost:5432/$DEV_DB_NAME"
echo ""
echo "For async connections:"
echo "  DATABASE_URL=postgresql+asyncpg://$DEV_DB_USER:$DEV_DB_PASS@localhost:5432/$DEV_DB_NAME"