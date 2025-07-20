#!/usr/bin/env python3
"""
Simple Air Quality Data Ingestion Pipeline (Dask-free)
Minimal version for stable operation

This is a simplified version that processes air quality data
without Dask dependencies to ensure stable operation.
"""

import pandas as pd
import numpy as np
import logging
import time
import os
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleAirQualityIngestor:
    """Simple air quality data ingestion without distributed processing."""
    
    def __init__(self):
        """Initialize simple processor."""
        logger.info("Initializing Simple Air Quality Ingestor")
        self.data_dir = Path("/app/data")
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir = self.data_dir / "raw"
        
        # Create directories if they don't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_sample_data(self):
        """Generate sample air quality data for demonstration."""
        logger.info("Generating sample air quality data...")
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1H'),
            'location_id': np.random.choice(['LOC_001', 'LOC_002', 'LOC_003', 'LOC_004'], n_samples),
            'pm2_5': np.random.normal(25, 10, n_samples).clip(0, 100),
            'pm10': np.random.normal(35, 15, n_samples).clip(0, 150),
            'no2': np.random.normal(30, 8, n_samples).clip(0, 80),
            'o3': np.random.normal(40, 12, n_samples).clip(0, 120),
            'temperature': np.random.normal(20, 10, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'wind_speed': np.random.exponential(5, n_samples).clip(0, 30),
            'latitude': np.random.uniform(40.0, 41.0, n_samples),
            'longitude': np.random.uniform(-74.5, -73.5, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Save raw data
        raw_file = self.raw_dir / "air_quality_sample.csv"
        df.to_csv(raw_file, index=False)
        logger.info(f"Generated sample data saved to {raw_file}")
        
        return df
    
    def process_data(self, df):
        """Process the air quality data."""
        logger.info("Processing air quality data...")
        
        # Basic processing
        df_processed = df.copy()
        
        # Calculate AQI (simplified)
        df_processed['aqi_pm2_5'] = (df_processed['pm2_5'] / 35.4 * 100).round()
        df_processed['aqi_pm10'] = (df_processed['pm10'] / 154 * 100).round()
        df_processed['aqi'] = df_processed[['aqi_pm2_5', 'aqi_pm10']].max(axis=1)
        
        # Add quality categories
        def get_aqi_category(aqi):
            if aqi <= 50:
                return 'Good'
            elif aqi <= 100:
                return 'Moderate'
            elif aqi <= 150:
                return 'Unhealthy for Sensitive Groups'
            elif aqi <= 200:
                return 'Unhealthy'
            elif aqi <= 300:
                return 'Very Unhealthy'
            else:
                return 'Hazardous'
        
        df_processed['aqi_category'] = df_processed['aqi'].apply(get_aqi_category)
        
        # Add processing timestamp
        df_processed['processed_at'] = datetime.now()
        
        return df_processed
    
    def save_processed_data(self, df):
        """Save processed data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_file = self.processed_dir / f"air_quality_processed_{timestamp}.csv"
        
        df.to_csv(processed_file, index=False)
        logger.info(f"Processed data saved to {processed_file}")
        
        # Also save as latest
        latest_file = self.processed_dir / "air_quality_latest.csv"
        df.to_csv(latest_file, index=False)
        logger.info(f"Latest data saved to {latest_file}")
    
    def run_pipeline(self):
        """Run the complete pipeline."""
        logger.info("Starting Simple Air Quality Ingestion Pipeline")
        
        try:
            # Generate or load data
            df = self.generate_sample_data()
            
            # Process data
            df_processed = self.process_data(df)
            
            # Save processed data
            self.save_processed_data(df_processed)
            
            logger.info(f"Pipeline completed successfully. Processed {len(df_processed)} records.")
            
            # Print summary
            print(f"\n=== Pipeline Summary ===")
            print(f"Records processed: {len(df_processed)}")
            print(f"Date range: {df_processed['timestamp'].min()} to {df_processed['timestamp'].max()}")
            print(f"Locations: {df_processed['location_id'].nunique()}")
            print(f"Average AQI: {df_processed['aqi'].mean():.1f}")
            print(f"AQI Categories:")
            print(df_processed['aqi_category'].value_counts())
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False

def main():
    """Main function to run the pipeline continuously."""
    ingestor = SimpleAirQualityIngestor()
    
    # Run pipeline once
    success = ingestor.run_pipeline()
    
    if success:
        logger.info("Pipeline completed successfully")
        
        # For demonstration, run in a loop with delays
        logger.info("Starting continuous monitoring mode...")
        while True:
            try:
                time.sleep(60)  # Wait 1 minute
                logger.info("Running periodic data refresh...")
                ingestor.run_pipeline()
            except KeyboardInterrupt:
                logger.info("Pipeline stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous mode: {e}")
                time.sleep(30)  # Wait 30 seconds before retry
    else:
        logger.error("Pipeline failed to start")

if __name__ == "__main__":
    main()
