#!/usr/bin/env python3
"""
Unit Tests for Dask-GeoPandas Air Quality Ingestion Pipeline
Author: Captain Aurelia "Skyforge" Stratos

Comprehensive test suite for spatial joins and data processing validation.
"""

import pytest
import pandas as pd
import dask.dataframe as dd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os
from pathlib import Path

# Import our ingest module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data-pipeline'))
from ingest import AirQualityIngestor

class TestAirQualityIngestor:
    """Test suite for AirQualityIngestor class."""
    
    @pytest.fixture
    def sample_air_quality_data(self):
        """Create sample air quality data for testing."""
        data = {
            'Date': ['10/03/2004', '10/03/2004', '10/03/2004'],
            'Time': ['18.00.00', '19.00.00', '20.00.00'],
            'CO(GT)': [2.6, 2.0, 2.2],
            'PT08.S1(CO)': [1360, 1292, 1402],
            'NMHC(GT)': [150.0, 112.0, 88.0],
            'C6H6(GT)': [11.9, 9.4, 9.0],
            'PT08.S2(NMHC)': [1046, 955, 939],
            'NOx(GT)': [166.0, 103.0, 131.0],
            'PT08.S3(NOx)': [1056, 1174, 1140],
            'NO2(GT)': [113.0, 92.0, 114.0],
            'PT08.S4(NO2)': [1692, 1559, 1555],
            'PT08.S5(O3)': [1268, 972, 1074],
            'T': [13.6, 13.3, 11.9],
            'RH': [48.9, 47.7, 54.0],
            'AH': [0.7578, 0.7255, 0.7502]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_weather_data(self):
        """Create sample weather data with geospatial coordinates."""
        data = {
            'country': ['USA', 'USA', 'Canada'],
            'location_name': ['New York', 'Los Angeles', 'Toronto'],
            'latitude': [40.7128, 34.0522, 43.6532],
            'longitude': [-74.0060, -118.2437, -79.3832],
            'last_updated': ['2024-05-16 13:15', '2024-05-16 13:15', '2024-05-16 13:15'],
            'temperature_celsius': [20.5, 25.3, 18.2],
            'humidity': [65, 45, 70],
            'pressure_mb': [1013.2, 1015.8, 1012.5],
            'wind_kph': [15.2, 8.7, 12.3],
            'air_quality_PM2.5': [12.5, 18.3, 8.9],
            'air_quality_PM10': [25.1, 32.7, 15.4],
            'air_quality_Carbon_Monoxide': [230.5, 340.2, 180.7],
            'air_quality_Ozone': [85.3, 95.7, 78.2],
            'air_quality_Nitrogen_dioxide': [25.8, 35.2, 20.1]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def ingestor(self):
        """Create AirQualityIngestor instance for testing."""
        with patch('ingest.Client') as mock_client:
            mock_client.return_value.dashboard_link = 'http://localhost:8787'
            ingestor = AirQualityIngestor(n_workers=2, memory_limit='1GB')
            yield ingestor
            ingestor.close()
    
    def test_initialization(self, ingestor):
        """Test proper initialization of AirQualityIngestor."""
        assert ingestor.client is not None
        assert hasattr(ingestor, 'load_air_quality_data')
        assert hasattr(ingestor, 'spatial_join_analysis')
    
    def test_clean_air_quality_data(self, ingestor, sample_air_quality_data):
        """Test air quality data cleaning functionality."""
        # Add some -200 values (sensor errors)
        sample_air_quality_data.loc[0, 'CO(GT)'] = -200.0
        sample_air_quality_data.loc[1, 'T'] = -200.0
        
        # Convert to Dask DataFrame
        ddf = dd.from_pandas(sample_air_quality_data, npartitions=2)
        
        # Clean the data
        cleaned_df = ingestor._clean_air_quality_data(ddf)
        
        # Compute result
        result = cleaned_df.compute()
        
        # Verify -200 values are replaced with NaN
        assert pd.isna(result.iloc[0]['CO(GT)'])
        assert pd.isna(result.iloc[1]['T'])
        
        # Verify datetime column is created
        assert 'datetime' in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result['datetime'])
        
        # Verify original Date and Time columns are dropped
        assert 'Date' not in result.columns
        assert 'Time' not in result.columns
    
    def test_clean_weather_data(self, ingestor, sample_weather_data):
        """Test weather data cleaning and validation."""
        # Add invalid coordinates
        invalid_data = sample_weather_data.copy()
        invalid_data.loc[len(invalid_data)] = {
            'country': 'Invalid',
            'location_name': 'Invalid Location',
            'latitude': 95.0,  # Invalid latitude
            'longitude': -200.0,  # Invalid longitude
            'last_updated': '2024-05-16 13:15',
            'temperature_celsius': 20.0,
            'humidity': 50,
            'pressure_mb': 1013.0,
            'wind_kph': 10.0,
            'air_quality_PM2.5': 15.0,
            'air_quality_PM10': 30.0,
            'air_quality_Carbon_Monoxide': 250.0,
            'air_quality_Ozone': 80.0,
            'air_quality_Nitrogen_dioxide': 25.0
        }
        
        # Convert to Dask DataFrame
        ddf = dd.from_pandas(invalid_data, npartitions=2)
        
        # Clean the data
        cleaned_df = ingestor._clean_weather_data(ddf)
        result = cleaned_df.compute()
        
        # Verify invalid coordinates are filtered out
        assert len(result) == 3  # Original 3 valid records
        assert all(result['latitude'].between(-90, 90))
        assert all(result['longitude'].between(-180, 180))
    
    def test_spatial_join_analysis(self, ingestor, sample_weather_data):
        """Test spatial grid aggregation functionality."""
        # Convert to Dask DataFrame
        ddf = dd.from_pandas(sample_weather_data, npartitions=2)
        
        # Perform spatial join with 1.0 degree resolution
        spatial_result = ingestor.spatial_join_analysis(ddf, grid_resolution=1.0)
        result = spatial_result.compute()
        
        # Verify grid columns are created
        assert 'grid_lat' in result.columns
        assert 'grid_lon' in result.columns
        
        # Verify aggregation occurred
        assert len(result) <= len(sample_weather_data)
        
        # Verify grid coordinates are properly rounded
        for _, row in result.iterrows():
            assert row['grid_lat'] % 1.0 == 0.0
            assert row['grid_lon'] % 1.0 == 0.0
    
    def test_compute_air_quality_index(self, ingestor, sample_weather_data):
        """Test AQI calculation functionality."""
        # Convert to Dask DataFrame
        ddf = dd.from_pandas(sample_weather_data, npartitions=2)
        
        # Compute AQI
        aqi_result = ingestor.compute_air_quality_index(ddf)
        result = aqi_result.compute()
        
        # Verify AQI column is created
        assert 'aqi' in result.columns
        
        # Verify AQI values are reasonable (should be positive)
        assert all(result['aqi'] >= 0)
        
        # Verify AQI calculation logic
        for _, row in result.iterrows():
            expected_pm25_aqi = (row['air_quality_PM2.5'] / 35.4) * 100
            expected_pm10_aqi = (row['air_quality_PM10'] / 154) * 100
            # AQI should be at least as high as the highest component
            assert row['aqi'] >= expected_pm25_aqi or row['aqi'] >= expected_pm10_aqi
    
    def test_export_to_parquet(self, ingestor, sample_weather_data):
        """Test Parquet export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_output.parquet')
            
            # Convert to Dask DataFrame
            ddf = dd.from_pandas(sample_weather_data, npartitions=2)
            
            # Export to Parquet
            ingestor.export_to_parquet(ddf, output_path)
            
            # Verify file was created
            assert os.path.exists(output_path)
            
            # Verify we can read the file back
            read_back = dd.read_parquet(output_path)
            result = read_back.compute()
            
            # Verify data integrity
            assert len(result) == len(sample_weather_data)
            assert list(result.columns) == list(sample_weather_data.columns)
    
    def test_coordinate_validation(self, ingestor):
        """Test coordinate validation edge cases."""
        edge_case_data = pd.DataFrame({
            'latitude': [-90, 90, 0, -91, 91],  # Include invalid values
            'longitude': [-180, 180, 0, -181, 181],  # Include invalid values
            'temperature_celsius': [20, 25, 22, 18, 30],
            'air_quality_PM2.5': [10, 15, 12, 8, 20]
        })
        
        ddf = dd.from_pandas(edge_case_data, npartitions=2)
        cleaned_df = ingestor._clean_weather_data(ddf)
        result = cleaned_df.compute()
        
        # Should only keep the first 3 rows (valid coordinates)
        assert len(result) == 3
        assert all(result['latitude'].between(-90, 90))
        assert all(result['longitude'].between(-180, 180))
    
    def test_outlier_removal(self, ingestor):
        """Test outlier removal using IQR method."""
        # Create data with obvious outliers
        outlier_data = pd.DataFrame({
            'latitude': [40.7] * 10,
            'longitude': [-74.0] * 10,
            'temperature_celsius': [20, 21, 22, 23, 24, 25, 26, 27, 100, -50],  # Last two are outliers
            'air_quality_PM2.5': [10, 11, 12, 13, 14, 15, 16, 17, 1000, -100]  # Last two are outliers
        })
        
        ddf = dd.from_pandas(outlier_data, npartitions=2)
        cleaned_df = ingestor._clean_weather_data(ddf)
        result = cleaned_df.compute()
        
        # Outliers should be removed
        assert len(result) < len(outlier_data)
        assert result['temperature_celsius'].max() < 100
        assert result['temperature_celsius'].min() > -50
        assert result['air_quality_PM2.5'].max() < 1000
        assert result['air_quality_PM2.5'].min() > -100

# Performance and integration tests
class TestPerformance:
    """Performance and integration test suite."""
    
    @pytest.mark.slow
    def test_large_dataset_processing(self):
        """Test processing performance with larger datasets."""
        # Create a larger synthetic dataset
        n_records = 10000
        large_data = pd.DataFrame({
            'latitude': np.random.uniform(-90, 90, n_records),
            'longitude': np.random.uniform(-180, 180, n_records),
            'temperature_celsius': np.random.normal(20, 10, n_records),
            'air_quality_PM2.5': np.random.exponential(15, n_records),
            'air_quality_PM10': np.random.exponential(25, n_records),
            'air_quality_Carbon_Monoxide': np.random.exponential(250, n_records),
            'air_quality_Ozone': np.random.exponential(80, n_records),
            'air_quality_Nitrogen_dioxide': np.random.exponential(25, n_records),
            'last_updated': pd.date_range('2024-01-01', periods=n_records, freq='H')
        })
        
        with patch('ingest.Client') as mock_client:
            mock_client.return_value.dashboard_link = 'http://localhost:8787'
            ingestor = AirQualityIngestor(n_workers=2)
            
            try:
                # Convert to Dask DataFrame
                ddf = dd.from_pandas(large_data, npartitions=10)
                
                # Time the spatial analysis
                import time
                start_time = time.time()
                
                spatial_result = ingestor.spatial_join_analysis(ddf, grid_resolution=0.5)
                aqi_result = ingestor.compute_air_quality_index(spatial_result)
                final_result = aqi_result.compute()
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Verify processing completed successfully
                assert len(final_result) > 0
                assert 'aqi' in final_result.columns
                
                # Performance assertion (should process 10k records in reasonable time)
                assert processing_time < 30  # 30 seconds max
                
                print(f"Processed {n_records} records in {processing_time:.2f} seconds")
                
            finally:
                ingestor.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])