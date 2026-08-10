#!/usr/bin/env python3
"""
News Articles Schema Migration
Author: GeoAirQuality AI Engineering

Revision ID: 004_news_articles
Revises: 003_geography_indexes
Create Date: 2026-08-10 14:00:00.000000

Adds the news_articles table powering the news intelligence layer:
official alerts + media articles, enriched (category/severity/relevance),
geo-tagged to the spatial grid, with optimized query indexes.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from geoalchemy2 import Geometry
import datetime

# revision identifiers
revision = '004_news_articles'
down_revision = '003_geography_indexes'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the news_articles table."""
    op.create_table(
        'news_articles',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('external_id', sa.String(length=200), nullable=False),
        sa.Column('source_name', sa.String(length=100), nullable=False),
        sa.Column('source_type', sa.String(length=20), nullable=False),
        sa.Column('title', sa.String(length=500), nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('url', sa.String(length=1000), nullable=True),
        sa.Column('published_at', sa.DateTime(), nullable=False),
        sa.Column('fetched_at', sa.DateTime(), nullable=False, default=datetime.datetime.utcnow),
        sa.Column('latitude', sa.Float(), nullable=True),
        sa.Column('longitude', sa.Float(), nullable=True),
        sa.Column('location', Geometry('POINT', srid=4326, spatial_index=True), nullable=True),
        sa.Column('grid_cell_id', sa.Integer(), nullable=True),
        sa.Column('event_category', sa.String(length=30), nullable=False, server_default='general'),
        sa.Column('severity', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('respiratory_relevance', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('is_active', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('raw_metadata', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=datetime.datetime.utcnow),
        sa.CheckConstraint(
            'severity >= 0 AND severity <= 100',
            name='valid_severity'
        ),
        sa.CheckConstraint(
            'respiratory_relevance >= 0 AND respiratory_relevance <= 100',
            name='valid_respiratory_relevance'
        ),
        sa.CheckConstraint(
            'latitude IS NULL OR (latitude >= -90 AND latitude <= 90)',
            name='valid_latitude'
        ),
        sa.CheckConstraint(
            'longitude IS NULL OR (longitude >= -180 AND longitude <= 180)',
            name='valid_longitude'
        ),
        sa.ForeignKeyConstraint(['grid_cell_id'], ['spatial_grids.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('external_id')
    )

    op.create_index('idx_news_articles_external_id', 'news_articles', ['external_id'])
    op.create_index('idx_news_published', 'news_articles', ['published_at'])
    op.create_index('idx_news_active_published', 'news_articles', ['is_active', 'published_at'])
    op.create_index('idx_news_grid_published', 'news_articles', ['grid_cell_id', 'published_at'])
    op.execute(
        'CREATE INDEX idx_news_metadata_gin ON news_articles USING gin (raw_metadata)'
    )


def downgrade() -> None:
    """Drop the news_articles table."""
    op.drop_table('news_articles')
