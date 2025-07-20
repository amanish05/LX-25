# Database Directory

This directory is for local database files (when using SQLite).

## Purpose
- SQLite database files
- Database backups
- Migration scripts
- Database dumps

## Structure
```
db/
├── trading_bot.db       # Main SQLite database (if not using PostgreSQL)
├── test_trading_bot.db  # Test database
├── backups/             # Database backups
└── migrations/          # SQL migration scripts
```

## Notes
- Files in this directory are gitignored
- Only used when PostgreSQL is not available
- Regular backups recommended