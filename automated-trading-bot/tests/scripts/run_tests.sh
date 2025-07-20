#!/bin/bash

# Test runner script for Automated Trading Bot

set -e  # Exit on error

echo "🧪 Running Automated Trading Bot Tests"
echo "====================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to run tests
run_tests() {
    local test_type=$1
    local test_path=$2
    local markers=$3
    
    echo -e "\n${BLUE}Running $test_type tests...${NC}"
    
    if [ -z "$markers" ]; then
        pytest $test_path -v
    else
        pytest $test_path -v -m "$markers"
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $test_type tests passed${NC}"
    else
        echo -e "${RED}✗ $test_type tests failed${NC}"
        exit 1
    fi
}

# Check if virtual environment is activated, if not try to activate it
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/bin/activate" ]; then
        echo -e "${BLUE}Activating virtual environment...${NC}"
        source .venv/bin/activate
    elif [ -f "venv/bin/activate" ]; then
        echo -e "${BLUE}Activating virtual environment...${NC}"
        source venv/bin/activate
    else
        echo -e "${RED}Error: Virtual environment not found${NC}"
        echo "Please create and activate your virtual environment first:"
        echo "  python -m venv .venv"
        echo "  source .venv/bin/activate"
        exit 1
    fi
fi

# Change to project root directory
cd "$(dirname "$0")/../.."

# Skip dependency installation when run via deployment pipeline
if [ -z "$DEPLOYMENT_PIPELINE" ]; then
    # Install test dependencies (only when run directly)
    echo -e "${BLUE}Installing test dependencies...${NC}"
    pip install -q -r requirements.txt
else
    echo -e "${BLUE}Running via deployment pipeline - dependencies assumed installed${NC}"
fi

# Clean previous coverage reports
echo -e "${BLUE}Cleaning previous coverage reports...${NC}"
rm -rf htmlcov
rm -f .coverage
rm -f coverage.xml

# Run different test suites based on argument
case "${1:-all}" in
    unit)
        run_tests "Unit" "tests/test_*.py" "unit"
        ;;
    integration)
        run_tests "Integration" "tests/integration/" "integration"
        ;;
    api)
        run_tests "API" "tests/integration/test_api_integration.py" "api"
        ;;
    database)
        run_tests "Database" "tests/integration/test_database_integration.py" "database"
        ;;
    config)
        run_tests "Configuration" "tests/integration/test_config_integration.py" "config"
        ;;
    bot)
        run_tests "Bot" "tests/test_*bot*.py tests/integration/test_*bot*.py" "bot"
        ;;
    fast)
        echo -e "${BLUE}Running fast tests (excluding slow tests)...${NC}"
        pytest -v -m "not slow"
        ;;
    parallel)
        echo -e "${BLUE}Running tests in parallel...${NC}"
        pytest -v -n auto
        ;;
    coverage)
        echo -e "${BLUE}Running all tests with coverage...${NC}"
        pytest --cov=src --cov-report=html --cov-report=term-missing
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;
    all)
        # Run all tests in sequence
        run_tests "Unit" "tests/test_*.py" ""
        run_tests "Integration" "tests/integration/" ""
        
        # Generate final coverage report
        echo -e "\n${BLUE}Generating coverage report...${NC}"
        coverage report
        coverage html
        
        echo -e "\n${GREEN}✓ All tests passed!${NC}"
        echo -e "Coverage report available at: htmlcov/index.html"
        ;;
    watch)
        echo -e "${BLUE}Running tests in watch mode...${NC}"
        # Use pytest-watch if installed
        if command -v ptw &> /dev/null; then
            ptw -- -v
        else
            echo "Installing pytest-watch..."
            pip install pytest-watch
            ptw -- -v
        fi
        ;;
    *)
        echo "Usage: $0 [unit|integration|api|database|config|bot|fast|parallel|coverage|all|watch]"
        echo ""
        echo "Options:"
        echo "  unit         - Run unit tests only"
        echo "  integration  - Run integration tests only"
        echo "  api          - Run API tests only"
        echo "  database     - Run database tests only"
        echo "  config       - Run configuration tests only"
        echo "  bot          - Run bot-related tests only"
        echo "  fast         - Run fast tests (exclude slow tests)"
        echo "  parallel     - Run tests in parallel"
        echo "  coverage     - Run all tests with coverage report"
        echo "  all          - Run all tests (default)"
        echo "  watch        - Run tests in watch mode"
        exit 1
        ;;
esac

# Check coverage threshold
if [ "${1:-all}" = "all" ] || [ "${1:-all}" = "coverage" ]; then
    echo -e "\n${BLUE}Checking coverage threshold...${NC}"
    coverage_percent=$(coverage report | grep TOTAL | awk '{print $4}' | sed 's/%//')
    
    if [ -n "$coverage_percent" ]; then
        if (( $(echo "$coverage_percent >= 80" | bc -l) )); then
            echo -e "${GREEN}✓ Coverage is $coverage_percent% (meets 80% threshold)${NC}"
        else
            echo -e "${RED}✗ Coverage is $coverage_percent% (below 80% threshold)${NC}"
            exit 1
        fi
    fi
fi

echo -e "\n${GREEN}✨ Test run completed successfully!${NC}"