#!/usr/bin/env python3
"""
Dask-GeoPandas Air Quality Data Ingestion Pipeline
Author: Captain Aurelia "Skyforge" Stratos

High-performance parallel processing of air quality and weather data
with spatial join capabilities for real-time environmental monitoring.
"""

import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
from dask.distributed import Client
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AirQualityIngestor:
    """Scalable air quality data ingestion with spatial processing."""
    
    def __init__(self, n_workers: int = 4, memory_limit: str = '2GB'):
        """Initialize Dask client for distributed processing."""
        self.client = Client(n_workers=n_workers, memory_limit=memory_limit)
        logger.info(f"Dask client initialized: {self.client.dashboard_link}")
    
    def load_air_quality_data(self, file_path: str) -> dd.DataFrame:
        """Load air quality CSV with Dask for parallel processing."""
        logger.info(f"Loading air quality data from {file_path}")
        
        # Define data types for optimization
        dtype_map = {
            'CO(GT)': 'float32',
            'PT08.S1(CO)': 'int32',
            'NMHC(GT)': 'float32',
            'C6H6(GT)': 'float32',
            'PT08.S2(NMHC)': 'int32',
            'NOx(GT)': 'float32',
            'PT08.S3(NOx)': 'int32',
            'NO2(GT)': 'float32',
            'PT08.S4(NO2)': 'int32',
            'PT08.S5(O3)': 'int32',
            'T': 'float32',
            'RH': 'float32',
            'AH': 'float32'
        }
        
        # Load with Dask for parallel processing
        df = dd.read_csv(
            file_path,
            sep=';',
            dtype=dtype_map,
            parse_dates=['Date'],
            blocksize='64MB'  # Optimize chunk size
        )
        
        # Clean and preprocess
        df = self._clean_air_quality_data(df)
        return df
    
    def load_weather_data(self, file_path: str) -> dd.DataFrame:
        """Load global weather repository with geospatial coordinates."""
        logger.info(f"Loading weather data from {file_path}")
        
        dtype_map = {
            'latitude': 'float64',
            'longitude': 'float64',
            'temperature_celsius': 'float32',
            'humidity': 'int16',
            'pressure_mb': 'float32',
            'wind_kph': 'float32',
            'air_quality_PM2.5': 'float32',
            'air_quality_PM10': 'float32',
            'air_quality_Carbon_Monoxide': 'float32',
            'air_quality_Ozone': 'float32',
            'air_quality_Nitrogen_dioxide': 'float32'
        }
        
        df = dd.read_csv(
            file_path,
            dtype=dtype_map,
            parse_dates=['last_updated'],
            blocksize='128MB'
        )
        
        return self._clean_weather_data(df)
    
    def _clean_air_quality_data(self, df: dd.DataFrame) -> dd.DataFrame:
        """Clean and normalize air quality sensor data."""
        # Replace -200 values (sensor errors) with NaN
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace(-200.0, np.nan)
        
        # Create datetime index
        df['datetime'] = dd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'])
        df = df.drop(['Date', 'Time'], axis=1)
        
        # Forward fill missing values (sensor continuity)
        df = df.fillna(method='ffill')
        
        return df
    
    def _clean_weather_data(self, df: dd.DataFrame) -> dd.DataFrame:
        """Clean and validate weather station data."""
        # Filter valid coordinates
        df = df[
            (df['latitude'].between(-90, 90)) & 
            (df['longitude'].between(-180, 180))
        ]
        
        # Remove outliers using IQR method
        for col in ['temperature_celsius', 'air_quality_PM2.5', 'air_quality_PM10']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def spatial_join_analysis(self, weather_df: dd.DataFrame, 
                            grid_resolution: float = 0.1) -> dd.DataFrame:
        """Perform spatial aggregation on weather data using grid cells."""
        logger.info(f"Performing spatial join with {grid_resolution}° resolution")
        
        # Create spatial grid cells
        weather_df['grid_lat'] = (weather_df['latitude'] / grid_resolution).round() * grid_resolution
        weather_df['grid_lon'] = (weather_df['longitude'] / grid_resolution).round() * grid_resolution
        
        # Aggregate by grid cell and time
        spatial_agg = weather_df.groupby(['grid_lat', 'grid_lon', 'last_updated']).agg({
            'temperature_celsius': 'mean',
            'air_quality_PM2.5': 'mean',
            'air_quality_PM10': 'mean',
            'air_quality_Carbon_Monoxide': 'mean',
            'air_quality_Ozone': 'mean',
            'humidity': 'mean',
            'pressure_mb': 'mean',
            'wind_kph': 'mean'
        }).reset_index()
        
        return spatial_agg
    
    def compute_air_quality_index(self, df: dd.DataFrame) -> dd.DataFrame:
        """Calculate composite Air Quality Index using EPA standards."""
        logger.info("Computing Air Quality Index")
        
        def calculate_aqi(pm25, pm10, co, o3, no2):
            """EPA AQI calculation formula."""
            # Simplified AQI calculation (production would use full EPA breakpoints)
            pm25_aqi = (pm25 / 35.4) * 100  # PM2.5 standard
            pm10_aqi = (pm10 / 154) * 100    # PM10 standard
            co_aqi = (co / 9000) * 100       # CO standard (μg/m³)
            o3_aqi = (o3 / 160) * 100        # O3 standard
            no2_aqi = (no2 / 100) * 100      # NO2 standard
            
            return np.maximum.reduce([pm25_aqi, pm10_aqi, co_aqi, o3_aqi, no2_aqi])
        
        df['aqi'] = df.apply(
            lambda row: calculate_aqi(
                row.get('air_quality_PM2.5', 0),
                row.get('air_quality_PM10', 0),
                row.get('air_quality_Carbon_Monoxide', 0),
                row.get('air_quality_Ozone', 0),
                row.get('air_quality_Nitrogen_dioxide', 0)
            ),
            axis=1,
            meta=('aqi', 'float32')
        )
        
        return df
    
    def export_to_parquet(self, df: dd.DataFrame, output_path: str) -> None:
        """Export processed data to Parquet for efficient storage."""
        logger.info(f"Exporting to Parquet: {output_path}")
        
        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',
            write_index=False
        )
        
        logger.info("Export completed successfully")
    
    def close(self):
        """Clean up Dask client."""
        self.client.close()
        logger.info("Dask client closed")

# Example usage and pipeline execution
if __name__ == "__main__":
    # Initialize ingestor
    ingestor = AirQualityIngestor(n_workers=4)
    
    try:
        # Load datasets
        air_quality_df = ingestor.load_air_quality_data("../AirQuality.csv")
        weather_df = ingestor.load_weather_data("../GlobalWeatherRepository.csv")
        
        # Perform spatial analysis
        spatial_weather = ingestor.spatial_join_analysis(weather_df)
        
        # Calculate AQI
        weather_with_aqi = ingestor.compute_air_quality_index(spatial_weather)
        
        # Export processed data
        ingestor.export_to_parquet(weather_with_aqi, "../processed_data/weather_aqi.parquet")
        
        logger.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        ingestor.close()