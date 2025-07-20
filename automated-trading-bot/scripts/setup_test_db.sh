#!/bin/bash
# Setup test database for automated trading bot

echo "Setting up test database for automated trading bot..."

# Check if PostgreSQL is running
if ! pg_isready -q; then
    echo "❌ PostgreSQL is not running. Please start PostgreSQL first."
    exit 1
fi

# Database configuration
TEST_DB_NAME="test_trading_bot"
TEST_DB_USER="test_user"
TEST_DB_PASS="test_pass"

# Set default PostgreSQL password for test setup
export PGPASSWORD="${PGPASSWORD:-427182}"

# Test PostgreSQL connection
if ! psql -U postgres -c '\q' 2>/dev/null; then
    echo "❌ Cannot connect to PostgreSQL with provided credentials."
    echo "   Make sure PostgreSQL is running and password is correct."
    echo "   Current PGPASSWORD: ${PGPASSWORD}"
    exit 1
fi

# Drop existing test database if it exists
echo "Dropping existing test database if exists..."
psql -U postgres -c "DROP DATABASE IF EXISTS $TEST_DB_NAME;" 2>/dev/null || true

# Drop existing test user if exists
psql -U postgres -c "DROP USER IF EXISTS $TEST_DB_USER;" 2>/dev/null || true

# Create test user
echo "Creating test user..."
psql -U postgres -c "CREATE USER $TEST_DB_USER WITH PASSWORD '$TEST_DB_PASS';" || {
    echo "❌ Failed to create test user"
    exit 1
}

# Create test database
echo "Creating test database..."
psql -U postgres -c "CREATE DATABASE $TEST_DB_NAME OWNER $TEST_DB_USER;" || {
    echo "❌ Failed to create test database"
    exit 1
}

# Grant all privileges
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE $TEST_DB_NAME TO $TEST_DB_USER;" || {
    echo "❌ Failed to grant privileges"
    exit 1
}

# Export test database URL
export TEST_DATABASE_URL="postgresql://$TEST_DB_USER:$TEST_DB_PASS@localhost:5432/$TEST_DB_NAME"

echo "✅ Test database setup complete!"
echo ""
echo "Test Database Configuration:"
echo "  Database: $TEST_DB_NAME"
echo "  User: $TEST_DB_USER"
echo "  Password: $TEST_DB_PASS"
echo ""
echo "To use this database in tests, set:"
echo "  export DATABASE_URL='$TEST_DATABASE_URL'"
echo ""
echo "Or for asyncpg:"
echo "  export DATABASE_URL='postgresql+asyncpg://$TEST_DB_USER:$TEST_DB_PASS@localhost:5432/$TEST_DB_NAME'"