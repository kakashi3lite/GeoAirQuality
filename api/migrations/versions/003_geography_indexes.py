#!/usr/bin/env python3
"""
Geography Expression Indexes Migration
Author: GeoAirQuality AI Engineering

Revision ID: 003_geography_indexes
Revises: 002_patient_safety_schema
Create Date: 2026-08-10 12:00:00.000000

Adds GiST expression indexes on (location::geography) so meter-accurate
spherical ST_DWithin queries (ST_DWithin(location::geography, point, meters))
remain index-accelerated instead of falling back to a sequential scan.

Without these indexes the geography cast in the radius queries would scan
the whole table; with them, PostGIS uses the GiST index for the bbox pre-
filter — keeping both accuracy (meters) and latency (indexed) intact.
"""

from alembic import op

# revision identifiers
revision = '003_geography_indexes'
down_revision = '002_patient_safety_schema'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create geography expression indexes for meter-accurate radius queries."""
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_aq_location_geog
        ON air_quality_readings USING gist ((location::geography))
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_weather_location_geog
        ON weather_readings USING gist ((location::geography))
    """)


def downgrade() -> None:
    """Drop the geography expression indexes."""
    op.execute("DROP INDEX IF EXISTS idx_aq_location_geog")
    op.execute("DROP INDEX IF EXISTS idx_weather_location_geog")
