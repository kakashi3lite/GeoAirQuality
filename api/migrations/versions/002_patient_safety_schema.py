#!/usr/bin/env python3
"""
Patient Safety Schema Migration
Author: GeoAirQuality AI Engineering

Revision ID: 002_patient_safety_schema
Revises: 001_initial_postgis_schema
Create Date: 2026-08-10 00:00:00.000000

Adds patient health profiles and symptom logs powering the personalized
safety scoring engine:
  * patient_profiles - conditions, personalized thresholds, home location
  * symptom_logs    - symptom events with environmental snapshots
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from geoalchemy2 import Geometry
import datetime

# revision identifiers
revision = '002_patient_safety_schema'
down_revision = '001_initial_postgis_schema'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create patient profile and symptom log tables."""

    op.create_table(
        'patient_profiles',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.String(length=100), nullable=False),
        sa.Column(
            'conditions',
            postgresql.ARRAY(sa.String(length=50)),
            nullable=False,
            server_default='{asthma}'
        ),
        sa.Column('aqi_threshold', sa.Integer(), nullable=False, server_default='100'),
        sa.Column('pm25_threshold', sa.Float(), nullable=False, server_default='35.4'),
        sa.Column('o3_threshold', sa.Float(), nullable=False, server_default='70.0'),
        sa.Column('no2_threshold', sa.Float(), nullable=False, server_default='100.0'),
        sa.Column('alert_radius_km', sa.Float(), nullable=False, server_default='25.0'),
        sa.Column('home_lat', sa.Float(), nullable=False, server_default='40.7128'),
        sa.Column('home_lon', sa.Float(), nullable=False, server_default='-74.0060'),
        sa.Column(
            'home_location',
            Geometry('POINT', srid=4326, spatial_index=True),
            nullable=True
        ),
        sa.Column('notification_preferences', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=datetime.datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), default=datetime.datetime.utcnow),
        sa.CheckConstraint('aqi_threshold >= 0', name='positive_aqi_threshold'),
        sa.CheckConstraint('pm25_threshold >= 0', name='positive_pm25_threshold'),
        sa.CheckConstraint('o3_threshold >= 0', name='positive_o3_threshold'),
        sa.CheckConstraint('no2_threshold >= 0', name='positive_no2_threshold'),
        sa.CheckConstraint(
            'alert_radius_km > 0 AND alert_radius_km <= 200',
            name='valid_alert_radius'
        ),
        sa.CheckConstraint('home_lat >= -90 AND home_lat <= 90', name='valid_home_lat'),
        sa.CheckConstraint('home_lon >= -180 AND home_lon <= 180', name='valid_home_lon'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id')
    )

    op.create_index('idx_patient_profiles_user_id', 'patient_profiles', ['user_id'])
    op.create_index('idx_patient_profiles_conditions', 'patient_profiles', ['conditions'])

    op.create_table(
        'symptom_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('patient_id', sa.Integer(), nullable=False),
        sa.Column('symptom_type', sa.String(length=50), nullable=False),
        sa.Column('severity', sa.Integer(), nullable=False),
        sa.Column('lat', sa.Float(), nullable=False),
        sa.Column('lon', sa.Float(), nullable=False),
        sa.Column(
            'location',
            Geometry('POINT', srid=4326, spatial_index=True),
            nullable=True
        ),
        sa.Column('weather_snapshot', postgresql.JSONB(), nullable=True),
        sa.Column('logged_at', sa.DateTime(), nullable=False, default=datetime.datetime.utcnow),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=datetime.datetime.utcnow),
        sa.CheckConstraint('severity >= 1 AND severity <= 5', name='valid_severity'),
        sa.CheckConstraint('lat >= -90 AND lat <= 90', name='valid_lat'),
        sa.CheckConstraint('lon >= -180 AND lon <= 180', name='valid_lon'),
        sa.ForeignKeyConstraint(['patient_id'], ['patient_profiles.id']),
        sa.PrimaryKeyConstraint('id')
    )

    op.create_index('idx_symptom_logs_patient_id', 'symptom_logs', ['patient_id'])
    op.create_index('idx_symptom_logs_logged_at', 'symptom_logs', ['logged_at'])
    op.create_index('idx_symptom_patient_time', 'symptom_logs', ['patient_id', 'logged_at'])
    op.create_index('idx_symptom_type_time', 'symptom_logs', ['symptom_type', 'logged_at'])


def downgrade() -> None:
    """Drop patient safety tables."""
    op.drop_table('symptom_logs')
    op.drop_table('patient_profiles')
