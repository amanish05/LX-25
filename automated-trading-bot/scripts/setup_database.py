#!/usr/bin/env python3
"""
PostgreSQL Database Setup for Automated Trading Bot
Requires PostgreSQL to be installed and running
"""

import os
import sys
import asyncio
import psycopg2
from psycopg2 import sql
from pathlib import Path
from urllib.parse import urlparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_postgresql():
    """Check if PostgreSQL is available"""
    try:
        # Try to connect to PostgreSQL default database
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="postgres",
            user="postgres"
        )
        conn.close()
        return True
    except Exception as e:
        return False


def create_database_if_not_exists(db_name, user, password):
    """Create database and user if they don't exist"""
    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="postgres",
            user="postgres"
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s",
            (user,)
        )
        if not cursor.fetchone():
            # Create user
            cursor.execute(
                sql.SQL("CREATE USER {} WITH PASSWORD %s").format(
                    sql.Identifier(user)
                ),
                (password,)
            )
            print(f"✅ Created user: {user}")
        
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (db_name,)
        )
        if not cursor.fetchone():
            # Create database
            cursor.execute(
                sql.SQL("CREATE DATABASE {} OWNER {}").format(
                    sql.Identifier(db_name),
                    sql.Identifier(user)
                )
            )
            print(f"✅ Created database: {db_name}")
        
        # Grant privileges
        cursor.execute(
            sql.SQL("GRANT ALL PRIVILEGES ON DATABASE {} TO {}").format(
                sql.Identifier(db_name),
                sql.Identifier(user)
            )
        )
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return False


async def setup_database_schema():
    """Setup database schema using SQLAlchemy"""
    from src.core.database import DatabaseManager, Base
    
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        print("❌ DATABASE_URL not set!")
        print("Please set DATABASE_URL in your .env file")
        return False
    
    if not database_url.startswith('postgresql'):
        print("❌ Only PostgreSQL is supported!")
        print(f"Invalid DATABASE_URL: {database_url}")
        return False
    
    # Convert to async URL
    async_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')
    
    try:
        db_manager = DatabaseManager(database_url=async_url)
        await db_manager.init_db()
        print("✅ Database schema created successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to create schema: {e}")
        return False


def setup_from_env():
    """Setup databases from environment variables"""
    # Load .env file if it exists
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get database URLs
    dev_url = os.getenv('DATABASE_URL')
    test_url = os.getenv('TEST_DATABASE_URL')
    
    if not dev_url:
        print("❌ DATABASE_URL not found in environment!")
        print("Please create a .env file from .env.example")
        return False
    
    # Parse development database URL
    dev_parsed = urlparse(dev_url)
    dev_db = dev_parsed.path.lstrip('/')
    dev_user = dev_parsed.username
    dev_pass = dev_parsed.password
    
    print(f"Setting up development database: {dev_db}")
    if not create_database_if_not_exists(dev_db, dev_user, dev_pass):
        return False
    
    # Setup test database if URL provided
    if test_url:
        test_parsed = urlparse(test_url)
        test_db = test_parsed.path.lstrip('/')
        test_user = test_parsed.username
        test_pass = test_parsed.password
        
        print(f"Setting up test database: {test_db}")
        if not create_database_if_not_exists(test_db, test_user, test_pass):
            return False
    
    return True


def main():
    """Main setup function"""
    print("=" * 60)
    print("Automated Trading Bot - PostgreSQL Database Setup")
    print("=" * 60)
    
    # Check PostgreSQL
    print("\n1. Checking PostgreSQL...")
    if not check_postgresql():
        print("❌ PostgreSQL is not running or not accessible!")
        print("\nTo install PostgreSQL:")
        print("  macOS:  brew install postgresql@14 && brew services start postgresql@14")
        print("  Ubuntu: sudo apt install postgresql && sudo systemctl start postgresql")
        sys.exit(1)
    print("✅ PostgreSQL is running")
    
    # Setup databases
    print("\n2. Setting up databases...")
    if not setup_from_env():
        sys.exit(1)
    
    # Create schema
    print("\n3. Creating database schema...")
    success = asyncio.run(setup_database_schema())
    
    if success:
        print("\n✅ Database setup complete!")
        print("\nYou can now run the trading bot:")
        print("  python main.py")
    else:
        print("\n❌ Database setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()