"""Initial database schema for automated trading bot

Revision ID: 001
Revises: 
Create Date: 2025-01-19 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import os

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def get_json_type():
    """Get appropriate JSON type based on database"""
    # Check if we're using PostgreSQL
    db_url = os.getenv('DATABASE_URL', '')
    if 'postgresql' in db_url or 'postgres' in db_url:
        return postgresql.JSONB
    else:
        return sa.JSON


def upgrade() -> None:
    """Create all tables for automated trading bot"""
    
    json_type = get_json_type()
    
    # Create bot_positions table
    op.create_table('bot_positions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('bot_name', sa.String(50), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('entry_price', sa.Float(), nullable=False),
        sa.Column('current_price', sa.Float(), nullable=True),
        sa.Column('unrealized_pnl', sa.Float(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('position_type', sa.String(10), nullable=False),
        sa.Column('metadata', json_type, nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_positions_bot_symbol', 'bot_positions', ['bot_name', 'symbol'])
    op.create_index('idx_positions_status', 'bot_positions', ['status'])
    
    # Create bot_trades table
    op.create_table('bot_trades',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('bot_name', sa.String(50), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('action', sa.String(10), nullable=False),
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('order_id', sa.String(50), nullable=True),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('executed_price', sa.Float(), nullable=True),
        sa.Column('fees', sa.Float(), nullable=True),
        sa.Column('pnl', sa.Float(), nullable=True),
        sa.Column('metadata', json_type, nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('execution_time', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_trades_bot_symbol', 'bot_trades', ['bot_name', 'symbol'])
    op.create_index('idx_trades_timestamp', 'bot_trades', ['timestamp'])
    op.create_index('idx_trades_order_id', 'bot_trades', ['order_id'])
    
    # Create bot_performance table
    op.create_table('bot_performance',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('bot_name', sa.String(50), nullable=False),
        sa.Column('total_trades', sa.Integer(), nullable=False),
        sa.Column('winning_trades', sa.Integer(), nullable=False),
        sa.Column('losing_trades', sa.Integer(), nullable=False),
        sa.Column('total_pnl', sa.Float(), nullable=False),
        sa.Column('win_rate', sa.Float(), nullable=False),
        sa.Column('average_win', sa.Float(), nullable=False),
        sa.Column('average_loss', sa.Float(), nullable=False),
        sa.Column('sharpe_ratio', sa.Float(), nullable=True),
        sa.Column('max_drawdown', sa.Float(), nullable=True),
        sa.Column('metadata', json_type, nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_performance_bot_timestamp', 'bot_performance', ['bot_name', 'timestamp'])
    
    # Create bot_signals table
    op.create_table('bot_signals',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('bot_name', sa.String(50), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('signal_type', sa.String(20), nullable=False),
        sa.Column('strength', sa.Float(), nullable=False),
        sa.Column('entry_price', sa.Float(), nullable=True),
        sa.Column('stop_loss', sa.Float(), nullable=True),
        sa.Column('take_profit', sa.Float(), nullable=True),
        sa.Column('metadata', json_type, nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('expiry', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_signals_bot_symbol', 'bot_signals', ['bot_name', 'symbol'])
    op.create_index('idx_signals_timestamp', 'bot_signals', ['timestamp'])
    
    # Create market_data_cache table
    op.create_table('market_data_cache',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('data_type', sa.String(20), nullable=False),
        sa.Column('data', json_type, nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('ttl', sa.Integer(), nullable=False),
        sa.Column('expiry', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'data_type', name='uq_cache_symbol_type')
    )
    op.create_index('idx_cache_expiry', 'market_data_cache', ['expiry'])
    
    # Create bot_capital table
    op.create_table('bot_capital',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('bot_name', sa.String(50), nullable=False),
        sa.Column('allocated_capital', sa.Float(), nullable=False),
        sa.Column('used_capital', sa.Float(), nullable=False),
        sa.Column('available_capital', sa.Float(), nullable=False),
        sa.Column('locked_capital', sa.Float(), nullable=False),
        sa.Column('total_pnl', sa.Float(), nullable=False),
        sa.Column('metadata', json_type, nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('bot_name', name='uq_capital_bot')
    )
    
    # PostgreSQL-specific optimizations
    if 'postgresql' in os.getenv('DATABASE_URL', ''):
        # Create extensions
        op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")  # For text search
        
        # Add JSONB indexes for metadata columns
        op.create_index('idx_positions_metadata', 'bot_positions', ['metadata'], 
                       postgresql_using='gin')
        op.create_index('idx_trades_metadata', 'bot_trades', ['metadata'], 
                       postgresql_using='gin')
        
        # Add partial indexes for active positions
        op.create_index('idx_positions_active', 'bot_positions', ['bot_name', 'symbol'],
                       postgresql_where=sa.text("status = 'ACTIVE'"))
        
        # Add indexes for time-series queries
        op.create_index('idx_trades_timestamp_desc', 'bot_trades', [sa.text('timestamp DESC')])
        op.create_index('idx_performance_timestamp_desc', 'bot_performance', [sa.text('timestamp DESC')])


def downgrade() -> None:
    """Drop all tables"""
    op.drop_table('bot_capital')
    op.drop_table('market_data_cache')
    op.drop_table('bot_signals')
    op.drop_table('bot_performance')
    op.drop_table('bot_trades')
    op.drop_table('bot_positions')