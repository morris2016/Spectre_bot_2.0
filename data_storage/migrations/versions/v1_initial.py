"""
Initial database schema

Revision ID: 5a8d9c0e7f31
Revises: 
Create Date: 2023-08-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '5a8d9c0e7f31'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial database schema."""
    
    # Create market_data table for storing OHLCV and other market data
    op.create_table(
        'market_data',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('platform', sa.String(length=50), nullable=False),
        sa.Column('symbol', sa.String(length=50), nullable=False),
        sa.Column('timeframe', sa.String(length=20), nullable=False),
        sa.Column('timestamp', sa.BigInteger(), nullable=False),
        sa.Column('open', sa.Float(), nullable=False),
        sa.Column('high', sa.Float(), nullable=False),
        sa.Column('low', sa.Float(), nullable=False),
        sa.Column('close', sa.Float(), nullable=False),
        sa.Column('volume', sa.Float(), nullable=False),
        sa.Column('num_trades', sa.Integer(), nullable=True),
        sa.Column('taker_buy_volume', sa.Float(), nullable=True),
        sa.Column('taker_sell_volume', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('platform', 'symbol', 'timeframe', 'timestamp', name='uix_market_data_platform_symbol_timeframe_timestamp'),
        sa.Index('ix_market_data_platform_symbol', 'platform', 'symbol'),
        sa.Index('ix_market_data_timeframe', 'timeframe'),
        sa.Index('ix_market_data_timestamp', 'timestamp')
    )
    
    # Create order_book_snapshot table for order book snapshots
    op.create_table(
        'order_book_snapshot',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('platform', sa.String(length=50), nullable=False),
        sa.Column('symbol', sa.String(length=50), nullable=False),
        sa.Column('timestamp', sa.BigInteger(), nullable=False),
        sa.Column('bids', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('asks', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('sequence_id', sa.BigInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_order_book_snapshot_platform_symbol', 'platform', 'symbol'),
        sa.Index('ix_order_book_snapshot_timestamp', 'timestamp')
    )
    
    # Create trade_data table for storing individual trades
    op.create_table(
        'trade_data',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('platform', sa.String(length=50), nullable=False),
        sa.Column('symbol', sa.String(length=50), nullable=False),
        sa.Column('trade_id', sa.String(length=100), nullable=False),
        sa.Column('timestamp', sa.BigInteger(), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('quantity', sa.Float(), nullable=False),
        sa.Column('is_buyer_maker', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('platform', 'symbol', 'trade_id', name='uix_trade_data_platform_symbol_trade_id'),
        sa.Index('ix_trade_data_platform_symbol', 'platform', 'symbol'),
        sa.Index('ix_trade_data_timestamp', 'timestamp')
    )
    
    # Create user_profile table
    op.create_table(
        'user_profile',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('username', sa.String(length=100), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=True),
        sa.Column('preferences', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username', name='uix_user_profile_username')
    )
    
    # Create api_key table for securely storing encrypted API keys
    op.create_table(
        'api_key',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('platform', sa.String(length=50), nullable=False),
        sa.Column('description', sa.String(length=255), nullable=True),
        sa.Column('encrypted_key', sa.Text(), nullable=False),
        sa.Column('encrypted_secret', sa.Text(), nullable=False),
        sa.Column('additional_params', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_used', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user_profile.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_api_key_user_id_platform', 'user_id', 'platform')
    )
    
    # Create strategy_config table for storing strategy configurations
    op.create_table(
        'strategy_config',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('strategy_type', sa.String(length=50), nullable=False),
        sa.Column('platform', sa.String(length=50), nullable=False),
        sa.Column('symbol', sa.String(length=50), nullable=False),
        sa.Column('timeframe', sa.String(length=20), nullable=False),
        sa.Column('parameters', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user_profile.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'name', name='uix_strategy_config_user_id_name'),
        sa.Index('ix_strategy_config_strategy_type', 'strategy_type'),
        sa.Index('ix_strategy_config_platform_symbol', 'platform', 'symbol')
    )
    
    # Create strategy_performance table for storing strategy performance metrics
    op.create_table(
        'strategy_performance',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('strategy_id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.BigInteger(), nullable=False),
        sa.Column('equity', sa.Float(), nullable=False),
        sa.Column('balance', sa.Float(), nullable=False),
        sa.Column('open_positions', sa.Integer(), nullable=False),
        sa.Column('win_count', sa.Integer(), nullable=False),
        sa.Column('loss_count', sa.Integer(), nullable=False),
        sa.Column('profit_factor', sa.Float(), nullable=True),
        sa.Column('metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['strategy_id'], ['strategy_config.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_strategy_performance_strategy_id', 'strategy_id'),
        sa.Index('ix_strategy_performance_timestamp', 'timestamp')
    )
    
    # Create trading_signal table for storing generated trading signals
    op.create_table(
        'trading_signal',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('strategy_id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.BigInteger(), nullable=False),
        sa.Column('platform', sa.String(length=50), nullable=False),
        sa.Column('symbol', sa.String(length=50), nullable=False),
        sa.Column('timeframe', sa.String(length=20), nullable=False),
        sa.Column('signal_type', sa.String(length=20), nullable=False),
        sa.Column('direction', sa.String(length=10), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('executed', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['strategy_id'], ['strategy_config.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_trading_signal_strategy_id', 'strategy_id'),
        sa.Index('ix_trading_signal_timestamp', 'timestamp'),
        sa.Index('ix_trading_signal_platform_symbol', 'platform', 'symbol')
    )
    
    # Create trading_position table for tracking active and historical positions
    op.create_table(
        'trading_position',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('strategy_id', sa.Integer(), nullable=False),
        sa.Column('platform', sa.String(length=50), nullable=False),
        sa.Column('symbol', sa.String(length=50), nullable=False),
        sa.Column('position_id', sa.String(length=100), nullable=True),
        sa.Column('direction', sa.String(length=10), nullable=False),
        sa.Column('entry_price', sa.Float(), nullable=False),
        sa.Column('exit_price', sa.Float(), nullable=True),
        sa.Column('quantity', sa.Float(), nullable=False),
        sa.Column('entry_time', sa.BigInteger(), nullable=False),
        sa.Column('exit_time', sa.BigInteger(), nullable=True),
        sa.Column('profit_loss', sa.Float(), nullable=True),
        sa.Column('profit_loss_pct', sa.Float(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['strategy_id'], ['strategy_config.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_trading_position_strategy_id', 'strategy_id'),
        sa.Index('ix_trading_position_platform_symbol', 'platform', 'symbol'),
        sa.Index('ix_trading_position_status', 'status')
    )
    
    # Create news_item table for storing market news
    op.create_table(
        'news_item',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('source', sa.String(length=100), nullable=False),
        sa.Column('title', sa.String(length=500), nullable=False),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('url', sa.String(length=1000), nullable=True),
        sa.Column('published_at', sa.BigInteger(), nullable=False),
        sa.Column('sentiment_score', sa.Float(), nullable=True),
        sa.Column('entities', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('categories', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_news_item_published_at', 'published_at'),
        sa.Index('ix_news_item_source', 'source')
    )
    
    # Create news_asset_relation table to link news items to assets
    op.create_table(
        'news_asset_relation',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('news_id', sa.BigInteger(), nullable=False),
        sa.Column('platform', sa.String(length=50), nullable=False),
        sa.Column('symbol', sa.String(length=50), nullable=False),
        sa.Column('relevance_score', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['news_id'], ['news_item.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_news_asset_relation_news_id', 'news_id'),
        sa.Index('ix_news_asset_relation_platform_symbol', 'platform', 'symbol')
    )
    
    # Create market_regime table for storing market regime classifications
    op.create_table(
        'market_regime',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('platform', sa.String(length=50), nullable=False),
        sa.Column('symbol', sa.String(length=50), nullable=False),
        sa.Column('timeframe', sa.String(length=20), nullable=False),
        sa.Column('timestamp', sa.BigInteger(), nullable=False),
        sa.Column('regime_type', sa.String(length=50), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_market_regime_platform_symbol', 'platform', 'symbol'),
        sa.Index('ix_market_regime_timestamp', 'timestamp')
    )
    
    # Create detected_pattern table for storing identified chart patterns
    op.create_table(
        'detected_pattern',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('platform', sa.String(length=50), nullable=False),
        sa.Column('symbol', sa.String(length=50), nullable=False),
        sa.Column('timeframe', sa.String(length=20), nullable=False),
        sa.Column('timestamp', sa.BigInteger(), nullable=False),
        sa.Column('pattern_type', sa.String(length=50), nullable=False),
        sa.Column('direction', sa.String(length=10), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('start_time', sa.BigInteger(), nullable=False),
        sa.Column('end_time', sa.BigInteger(), nullable=False),
        sa.Column('target_price', sa.Float(), nullable=True),
        sa.Column('stop_price', sa.Float(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_detected_pattern_platform_symbol', 'platform', 'symbol'),
        sa.Index('ix_detected_pattern_timestamp', 'timestamp'),
        sa.Index('ix_detected_pattern_pattern_type', 'pattern_type')
    )
    
    # Create system_log table for storing system events and errors
    op.create_table(
        'system_log',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('timestamp', sa.BigInteger(), nullable=False),
        sa.Column('service', sa.String(length=100), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_system_log_timestamp', 'timestamp'),
        sa.Index('ix_system_log_service', 'service'),
        sa.Index('ix_system_log_severity', 'severity')
    )
    
    # Create system_metric table for storing performance metrics
    op.create_table(
        'system_metric',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('timestamp', sa.BigInteger(), nullable=False),
        sa.Column('service', sa.String(length=100), nullable=False),
        sa.Column('metric_name', sa.String(length=100), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_system_metric_timestamp', 'timestamp'),
        sa.Index('ix_system_metric_service', 'service'),
        sa.Index('ix_system_metric_metric_name', 'metric_name')
    )
    
    # Create feature_data table for storing calculated features
    op.create_table(
        'feature_data',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('platform', sa.String(length=50), nullable=False),
        sa.Column('symbol', sa.String(length=50), nullable=False),
        sa.Column('timeframe', sa.String(length=20), nullable=False),
        sa.Column('timestamp', sa.BigInteger(), nullable=False),
        sa.Column('feature_set', sa.String(length=50), nullable=False),
        sa.Column('features', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('platform', 'symbol', 'timeframe', 'timestamp', 'feature_set', name='uix_feature_data_platform_symbol_timeframe_timestamp_feature_set'),
        sa.Index('ix_feature_data_platform_symbol', 'platform', 'symbol'),
        sa.Index('ix_feature_data_timestamp', 'timestamp'),
        sa.Index('ix_feature_data_feature_set', 'feature_set')
    )
    
    # Create ml_model table for tracking machine learning models
    op.create_table(
        'ml_model',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('model_type', sa.String(length=50), nullable=False),
        sa.Column('platform', sa.String(length=50), nullable=True),
        sa.Column('symbol', sa.String(length=50), nullable=True),
        sa.Column('timeframe', sa.String(length=20), nullable=True),
        sa.Column('parameters', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('performance_metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name', 'version', name='uix_ml_model_name_version'),
        sa.Index('ix_ml_model_model_type', 'model_type'),
        sa.Index('ix_ml_model_platform_symbol', 'platform', 'symbol'),
        sa.Index('ix_ml_model_is_active', 'is_active')
    )


def downgrade() -> None:
    """Drop all tables created in the upgrade function."""
    op.drop_table('ml_model')
    op.drop_table('feature_data')
    op.drop_table('system_metric')
    op.drop_table('system_log')
    op.drop_table('detected_pattern')
    op.drop_table('market_regime')
    op.drop_table('news_asset_relation')
    op.drop_table('news_item')
    op.drop_table('trading_position')
    op.drop_table('trading_signal')
    op.drop_table('strategy_performance')
    op.drop_table('strategy_config')
    op.drop_table('api_key')
    op.drop_table('user_profile')
    op.drop_table('trade_data')
    op.drop_table('order_book_snapshot')
    op.drop_table('market_data')
