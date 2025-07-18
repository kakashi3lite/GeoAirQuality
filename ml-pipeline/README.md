# Machine Learning Pipeline Implementation

## Overview
This document provides comprehensive implementation for the GeoAirQuality ML pipeline using MLflow, Ray, and advanced time-series forecasting models.

## Architecture
- **MLflow**: Model registry, experiment tracking, and deployment
- **Ray**: Distributed training and hyperparameter tuning  
- **TensorFlow/PyTorch**: Deep learning models (LSTM, Transformer)
- **Prophet/XGBoost**: Classical time-series and gradient boosting models
- **Apache Airflow**: Pipeline orchestration
- **Feature Store**: Centralized feature management

---

## MLflow Setup and Configuration

### MLflow Server Configuration
```python
# ml-pipeline/mlflow/server_config.py
import os
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

class MLflowServer:
    def __init__(self, tracking_uri: str, registry_uri: str = None):
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri or tracking_uri
        
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        
        self.client = MlflowClient()
    
    def create_experiment(self, name: str, artifact_location: str = None) -> str:
        """Create MLflow experiment if it doesn't exist"""
        try:
            experiment_id = mlflow.create_experiment(
                name=name,
                artifact_location=artifact_location
            )
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment_id = mlflow.get_experiment_by_name(name).experiment_id
        
        return experiment_id
    
    def log_model_performance(self, 
                            model_name: str,
                            metrics: dict,
                            parameters: dict,
                            model_artifacts: dict,
                            tags: dict = None):
        """Log model performance and artifacts"""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(parameters)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log tags
            if tags:
                mlflow.set_tags(tags)
            
            # Log model artifacts
            for artifact_name, artifact_data in model_artifacts.items():
                if artifact_name.endswith('.pkl'):
                    mlflow.sklearn.log_model(artifact_data, artifact_name)
                elif artifact_name.endswith('.h5'):
                    mlflow.tensorflow.log_model(artifact_data, artifact_name)
            
            return mlflow.active_run().info.run_id

# Initialize MLflow server
mlflow_server = MLflowServer(
    tracking_uri="http://mlflow-server:5000",
    registry_uri="http://mlflow-server:5000"
)
```

### Model Registry Management
```python
# ml-pipeline/mlflow/model_registry.py
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient
from typing import Dict, List, Optional
import mlflow

class ModelRegistryManager:
    def __init__(self, client: MlflowClient):
        self.client = client
    
    def register_model(self, 
                      model_name: str,
                      run_id: str,
                      artifact_path: str,
                      description: str = None) -> ModelVersion:
        """Register a model version"""
        # Create registered model if it doesn't exist
        try:
            self.client.create_registered_model(
                name=model_name,
                description=description
            )
        except mlflow.exceptions.MlflowException:
            pass  # Model already exists
        
        # Create model version
        model_uri = f"runs:/{run_id}/{artifact_path}"
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        return model_version
    
    def promote_model(self, 
                     model_name: str,
                     version: str,
                     stage: str) -> ModelVersion:
        """Promote model to different stage (Staging, Production)"""
        return self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
    
    def get_latest_model_version(self, 
                               model_name: str,
                               stage: str = "Production") -> Optional[ModelVersion]:
        """Get latest model version in specified stage"""
        versions = self.client.get_latest_versions(
            name=model_name,
            stages=[stage]
        )
        return versions[0] if versions else None
    
    def load_production_model(self, model_name: str):
        """Load the production version of a model"""
        model_version = self.get_latest_model_version(model_name, "Production")
        if model_version:
            model_uri = f"models:/{model_name}/{model_version.version}"
            return mlflow.pyfunc.load_model(model_uri)
        else:
            raise ValueError(f"No production model found for {model_name}")
    
    def compare_models(self, 
                      model_name: str,
                      metric_name: str,
                      stages: List[str] = ["Staging", "Production"]) -> Dict:
        """Compare model performance across stages"""
        comparison = {}
        
        for stage in stages:
            version = self.get_latest_model_version(model_name, stage)
            if version:
                run = self.client.get_run(version.run_id)
                metric_value = run.data.metrics.get(metric_name)
                comparison[stage] = {
                    "version": version.version,
                    "metric_value": metric_value,
                    "run_id": version.run_id
                }
        
        return comparison

# Initialize registry manager
registry_manager = ModelRegistryManager(mlflow_server.client)
```

---

## Time Series Forecasting Models

### LSTM Neural Network Implementation
```python
# ml-pipeline/models/lstm_forecaster.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import mlflow
import mlflow.tensorflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from typing import Tuple, Dict, Any

class LSTMAirQualityForecaster:
    def __init__(self, 
                 sequence_length: int = 24,
                 forecast_horizon: int = 12,
                 features: List[str] = None):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.features = features or ['aqi', 'pm25', 'pm10', 'temperature', 'humidity', 'wind_speed']
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def create_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Create LSTM model architecture"""
        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            
            # Second LSTM layer
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            # Third LSTM layer
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers for output
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(self.forecast_horizon, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_sequences(self, 
                         data: pd.DataFrame,
                         target_column: str = 'aqi') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series sequences for training"""
        # Select features
        feature_data = data[self.features].values
        target_data = data[target_column].values.reshape(-1, 1)
        
        # Scale features and target
        X_scaled = self.scaler_X.fit_transform(feature_data)
        y_scaled = self.scaler_y.fit_transform(target_data)
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(X_scaled) - self.forecast_horizon + 1):
            # Input sequence
            X.append(X_scaled[i-self.sequence_length:i])
            
            # Output sequence (next forecast_horizon values)
            y.append(y_scaled[i:i+self.forecast_horizon].flatten())
        
        return np.array(X), np.array(y)
    
    def train(self, 
              train_data: pd.DataFrame,
              val_data: pd.DataFrame,
              target_column: str = 'aqi',
              epochs: int = 100,
              batch_size: int = 32) -> Dict[str, Any]:
        """Train the LSTM model"""
        # Prepare training data
        X_train, y_train = self.prepare_sequences(train_data, target_column)
        X_val, y_val = self.prepare_sequences(val_data, target_column)
        
        # Create model
        self.model = self.create_model((X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                "sequence_length": self.sequence_length,
                "forecast_horizon": self.forecast_horizon,
                "epochs": epochs,
                "batch_size": batch_size,
                "features": self.features
            })
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            train_loss = self.model.evaluate(X_train, y_train, verbose=0)[0]
            val_loss = self.model.evaluate(X_val, y_val, verbose=0)[0]
            
            # Generate predictions for metrics
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)
            
            # Inverse transform predictions
            train_pred_inv = self.scaler_y.inverse_transform(train_pred)
            val_pred_inv = self.scaler_y.inverse_transform(val_pred)
            train_true_inv = self.scaler_y.inverse_transform(y_train)
            val_true_inv = self.scaler_y.inverse_transform(y_val)
            
            # Calculate metrics
            train_mae = mean_absolute_error(train_true_inv, train_pred_inv)
            val_mae = mean_absolute_error(val_true_inv, val_pred_inv)
            train_rmse = np.sqrt(mean_squared_error(train_true_inv, train_pred_inv))
            val_rmse = np.sqrt(mean_squared_error(val_true_inv, val_pred_inv))
            
            # Log metrics
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_mae": train_mae,
                "val_mae": val_mae,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse
            }
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.tensorflow.log_model(
                self.model,
                "lstm_model",
                signature=mlflow.models.infer_signature(X_train, train_pred)
            )
            
            return {
                "metrics": metrics,
                "history": history.history,
                "run_id": mlflow.active_run().info.run_id
            }
    
    def predict(self, 
                data: pd.DataFrame,
                steps_ahead: int = None) -> np.ndarray:
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        steps_ahead = steps_ahead or self.forecast_horizon
        
        # Prepare input sequence
        feature_data = data[self.features].tail(self.sequence_length).values
        X_scaled = self.scaler_X.transform(feature_data)
        X_input = X_scaled.reshape(1, self.sequence_length, len(self.features))
        
        # Make prediction
        prediction_scaled = self.model.predict(X_input)
        prediction = self.scaler_y.inverse_transform(prediction_scaled)
        
        return prediction.flatten()[:steps_ahead]
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model:
            self.model.save(filepath)
        else:
            raise ValueError("No model to save. Train the model first.")
    
    def load_model(self, filepath: str):
        """Load a saved model"""
        self.model = tf.keras.models.load_model(filepath)
```

### Prophet Model Implementation
```python
# ml-pipeline/models/prophet_forecaster.py
from prophet import Prophet
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
from typing import Dict, Any, List
import logging

# Suppress Prophet logging
logging.getLogger('prophet').setLevel(logging.WARNING)

class ProphetAirQualityForecaster:
    def __init__(self, 
                 seasonality_mode: str = 'multiplicative',
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = True,
                 changepoint_prior_scale: float = 0.05):
        
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model = None
        self.regressors = []
    
    def create_model(self) -> Prophet:
        """Create Prophet model with configuration"""
        model = Prophet(
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        
        return model
    
    def add_regressors(self, regressors: List[str]):
        """Add external regressors to the model"""
        self.regressors = regressors
        if self.model:
            for regressor in regressors:
                self.model.add_regressor(regressor)
    
    def prepare_data(self, 
                    data: pd.DataFrame,
                    target_column: str = 'aqi',
                    timestamp_column: str = 'timestamp') -> pd.DataFrame:
        """Prepare data in Prophet format"""
        prophet_data = pd.DataFrame({
            'ds': pd.to_datetime(data[timestamp_column]),
            'y': data[target_column]
        })
        
        # Add regressors
        for regressor in self.regressors:
            if regressor in data.columns:
                prophet_data[regressor] = data[regressor]
        
        return prophet_data
    
    def train(self, 
              train_data: pd.DataFrame,
              val_data: pd.DataFrame,
              target_column: str = 'aqi',
              timestamp_column: str = 'timestamp',
              regressors: List[str] = None) -> Dict[str, Any]:
        """Train the Prophet model"""
        
        # Set up regressors
        if regressors:
            self.add_regressors(regressors)
        
        # Create model
        self.model = self.create_model()
        
        # Add regressors to model
        for regressor in self.regressors:
            self.model.add_regressor(regressor)
        
        # Prepare training data
        train_prophet = self.prepare_data(train_data, target_column, timestamp_column)
        val_prophet = self.prepare_data(val_data, target_column, timestamp_column)
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                "seasonality_mode": self.seasonality_mode,
                "yearly_seasonality": self.yearly_seasonality,
                "weekly_seasonality": self.weekly_seasonality,
                "daily_seasonality": self.daily_seasonality,
                "changepoint_prior_scale": self.changepoint_prior_scale,
                "regressors": self.regressors
            })
            
            # Fit model
            self.model.fit(train_prophet)
            
            # Make predictions on training and validation sets
            train_forecast = self.model.predict(train_prophet)
            val_forecast = self.model.predict(val_prophet)
            
            # Calculate metrics
            train_mae = mean_absolute_error(train_prophet['y'], train_forecast['yhat'])
            val_mae = mean_absolute_error(val_prophet['y'], val_forecast['yhat'])
            train_rmse = np.sqrt(mean_squared_error(train_prophet['y'], train_forecast['yhat']))
            val_rmse = np.sqrt(mean_squared_error(val_prophet['y'], val_forecast['yhat']))
            
            # Calculate additional Prophet-specific metrics
            train_mape = np.mean(np.abs((train_prophet['y'] - train_forecast['yhat']) / train_prophet['y'])) * 100
            val_mape = np.mean(np.abs((val_prophet['y'] - val_forecast['yhat']) / val_prophet['y'])) * 100
            
            metrics = {
                "train_mae": train_mae,
                "val_mae": val_mae,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "train_mape": train_mape,
                "val_mape": val_mape
            }
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(
                self.model,
                "prophet_model"
            )
            
            return {
                "metrics": metrics,
                "train_forecast": train_forecast,
                "val_forecast": val_forecast,
                "run_id": mlflow.active_run().info.run_id
            }
    
    def predict(self, 
                future_data: pd.DataFrame,
                periods: int = 24) -> pd.DataFrame:
        """Make predictions for future periods"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create future dataframe
        if future_data is not None:
            future = future_data.copy()
        else:
            future = self.model.make_future_dataframe(periods=periods, freq='H')
        
        # Make predictions
        forecast = self.model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def get_component_plots(self) -> Dict[str, Any]:
        """Generate component plots for analysis"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # This would typically return matplotlib figures
        # For now, return component importance
        components = {
            "trend": True,
            "yearly": self.yearly_seasonality,
            "weekly": self.weekly_seasonality,
            "daily": self.daily_seasonality
        }
        
        return components
```

### XGBoost Model Implementation
```python
# ml-pipeline/models/xgboost_forecaster.py
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.xgboost
from typing import Dict, Any, List, Tuple

class XGBoostAirQualityForecaster:
    def __init__(self, 
                 n_estimators: int = 1000,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 0.1):
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def create_model(self) -> xgb.XGBRegressor:
        """Create XGBoost model with configuration"""
        model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=42,
            n_jobs=-1
        )
        
        return model
    
    def create_features(self, 
                       data: pd.DataFrame,
                       target_column: str = 'aqi',
                       lag_features: List[int] = [1, 2, 3, 6, 12, 24],
                       rolling_windows: List[int] = [6, 12, 24]) -> pd.DataFrame:
        """Create time series features for XGBoost"""
        df = data.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Lag features
        for lag in lag_features:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        
        # Rolling statistics
        for window in rolling_windows:
            df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window).mean()
            df[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(window).std()
            df[f'{target_column}_rolling_min_{window}'] = df[target_column].rolling(window).min()
            df[f'{target_column}_rolling_max_{window}'] = df[target_column].rolling(window).max()
        
        # Polynomial features for continuous variables
        continuous_cols = ['temperature', 'humidity', 'wind_speed', 'pressure']
        for col in continuous_cols:
            if col in df.columns:
                df[f'{col}_squared'] = df[col] ** 2
                df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
        
        # Interaction features
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        
        # Target encoding for categorical variables
        if 'season' in df.columns:
            season_encoding = df.groupby('season')[target_column].mean()
            df['season_encoded'] = df['season'].map(season_encoding)
        
        return df
    
    def prepare_data(self, 
                    data: pd.DataFrame,
                    target_column: str = 'aqi') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for training"""
        # Create features
        df_features = self.create_features(data, target_column)
        
        # Remove rows with NaN values
        df_clean = df_features.dropna()
        
        # Separate features and target
        target = df_clean[target_column].values
        features = df_clean.drop(columns=[target_column])
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(features.values)
        
        return X_scaled, target
    
    def train(self, 
              train_data: pd.DataFrame,
              val_data: pd.DataFrame,
              target_column: str = 'aqi') -> Dict[str, Any]:
        """Train the XGBoost model"""
        
        # Prepare data
        X_train, y_train = self.prepare_data(train_data, target_column)
        X_val, y_val = self.prepare_data(val_data, target_column)
        
        # Create model
        self.model = self.create_model()
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda,
                "n_features": len(self.feature_names)
            })
            
            # Train model with early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_metric='rmse',
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Make predictions
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, train_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            
            # Feature importance
            feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
            
            metrics = {
                "train_mae": train_mae,
                "val_mae": val_mae,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "best_iteration": self.model.best_iteration
            }
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log feature importance as artifacts
            importance_df = pd.DataFrame([
                {"feature": k, "importance": v} 
                for k, v in feature_importance.items()
            ]).sort_values("importance", ascending=False)
            
            mlflow.log_text(
                importance_df.to_string(index=False),
                "feature_importance.txt"
            )
            
            # Log model
            mlflow.xgboost.log_model(
                self.model,
                "xgboost_model",
                signature=mlflow.models.infer_signature(X_train, train_pred)
            )
            
            return {
                "metrics": metrics,
                "feature_importance": feature_importance,
                "run_id": mlflow.active_run().info.run_id
            }
    
    def predict(self, data: pd.DataFrame, target_column: str = 'aqi') -> np.ndarray:
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        df_features = self.create_features(data, target_column)
        features = df_features[self.feature_names]
        
        # Handle missing values
        features = features.fillna(features.mean())
        
        # Scale features
        X_scaled = self.scaler.transform(features.values)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance ranking"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
```

---

## Distributed Training with Ray

### Ray Training Configuration
```python
# ml-pipeline/distributed/ray_trainer.py
import ray
from ray import tune
from ray.train import Trainer
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig
import torch
import torch.nn as nn
from typing import Dict, Any
import mlflow

@ray.remote
class DistributedMLTrainer:
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        
    def setup_ray_cluster(self, num_workers: int = 4):
        """Setup Ray cluster for distributed training"""
        if not ray.is_initialized():
            ray.init(
                num_cpus=num_workers * 2,
                object_store_memory=2000000000,  # 2GB
                dashboard_host="0.0.0.0"
            )
    
    def hyperparameter_tuning(self, 
                            model_class,
                            train_data,
                            val_data,
                            search_space: Dict,
                            num_samples: int = 10) -> Dict[str, Any]:
        """Perform distributed hyperparameter tuning"""
        
        def train_func(config):
            # Initialize model with current hyperparameters
            model = model_class(**config)
            
            # Train model
            results = model.train(train_data, val_data)
            
            # Report metrics to Ray Tune
            session.report({
                "val_mae": results["metrics"]["val_mae"],
                "val_rmse": results["metrics"]["val_rmse"],
                "train_mae": results["metrics"]["train_mae"]
            })
        
        # Configure search algorithm
        search_alg = tune.search.optuna.OptunaSearch(
            metric="val_mae",
            mode="min"
        )
        
        # Configure scheduler
        scheduler = tune.schedulers.ASHAScheduler(
            metric="val_mae",
            mode="min",
            max_t=100,
            grace_period=10,
            reduction_factor=2
        )
        
        # Run hyperparameter tuning
        tuner = tune.Tuner(
            train_func,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=num_samples
            ),
            run_config=ray.air.RunConfig(
                stop={"training_iteration": 100},
                local_dir="./ray_results"
            )
        )
        
        results = tuner.fit()
        best_result = results.get_best_result(metric="val_mae", mode="min")
        
        return {
            "best_config": best_result.config,
            "best_metrics": best_result.metrics,
            "all_results": results
        }

# Example usage for LSTM hyperparameter tuning
def lstm_hyperparameter_search():
    """Example hyperparameter search for LSTM model"""
    trainer = DistributedMLTrainer.remote({})
    
    search_space = {
        "sequence_length": tune.choice([12, 24, 48]),
        "forecast_horizon": tune.choice([6, 12, 24]),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
        "lstm_units": tune.choice([32, 64, 128, 256]),
        "dropout_rate": tune.uniform(0.1, 0.5)
    }
    
    # Load your training data here
    train_data = load_training_data()
    val_data = load_validation_data()
    
    results = ray.get(trainer.hyperparameter_tuning.remote(
        LSTMAirQualityForecaster,
        train_data,
        val_data,
        search_space,
        num_samples=20
    ))
    
    return results
```

---

## Model Ensemble and A/B Testing

### Model Ensemble Implementation
```python
# ml-pipeline/ensemble/model_ensemble.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
from concurrent.futures import ThreadPoolExecutor
import asyncio

class AirQualityModelEnsemble:
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float] = None):
        self.models = models
        self.weights = weights or {name: 1.0/len(models) for name in models.keys()}
        self.performance_history = {}
        
    def validate_weights(self):
        """Ensure weights sum to 1.0"""
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def predict_single_model(self, model_name: str, data: pd.DataFrame) -> np.ndarray:
        """Get prediction from a single model"""
        try:
            model = self.models[model_name]
            prediction = model.predict(data)
            return prediction
        except Exception as e:
            print(f"Error in model {model_name}: {e}")
            return np.zeros(len(data))  # Return zeros if model fails
    
    def predict_weighted_average(self, data: pd.DataFrame) -> np.ndarray:
        """Get weighted average prediction from all models"""
        self.validate_weights()
        
        predictions = {}
        
        # Get predictions from all models in parallel
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            future_to_model = {
                executor.submit(self.predict_single_model, name, data): name
                for name in self.models.keys()
            }
            
            for future in future_to_model:
                model_name = future_to_model[future]
                try:
                    predictions[model_name] = future.result()
                except Exception as e:
                    print(f"Model {model_name} failed: {e}")
                    predictions[model_name] = np.zeros(len(data))
        
        # Calculate weighted average
        ensemble_prediction = np.zeros(len(data))
        for model_name, pred in predictions.items():
            weight = self.weights[model_name]
            ensemble_prediction += weight * pred
        
        return ensemble_prediction
    
    def predict_stacking(self, data: pd.DataFrame, meta_model) -> np.ndarray:
        """Use stacking ensemble with meta-model"""
        # Get predictions from base models
        base_predictions = []
        
        for model_name in self.models.keys():
            pred = self.predict_single_model(model_name, data)
            base_predictions.append(pred)
        
        # Stack predictions as features for meta-model
        stacked_features = np.column_stack(base_predictions)
        
        # Get final prediction from meta-model
        final_prediction = meta_model.predict(stacked_features)
        
        return final_prediction
    
    def adaptive_weighting(self, 
                          val_data: pd.DataFrame,
                          target_column: str = 'aqi',
                          window_size: int = 100) -> Dict[str, float]:
        """Adaptively update model weights based on recent performance"""
        model_errors = {}
        
        # Get recent predictions and calculate errors
        for model_name in self.models.keys():
            predictions = self.predict_single_model(model_name, val_data)
            true_values = val_data[target_column].values
            
            # Use recent window for error calculation
            recent_pred = predictions[-window_size:]
            recent_true = true_values[-window_size:]
            
            mae = mean_absolute_error(recent_true, recent_pred)
            model_errors[model_name] = mae
        
        # Calculate inverse error weights (better models get higher weights)
        max_error = max(model_errors.values())
        inverse_errors = {
            name: max_error - error + 1e-6 
            for name, error in model_errors.items()
        }
        
        # Normalize to get weights
        total_inverse = sum(inverse_errors.values())
        new_weights = {
            name: inv_err / total_inverse 
            for name, inv_err in inverse_errors.items()
        }
        
        self.weights = new_weights
        
        return new_weights
    
    def evaluate_ensemble(self, 
                         test_data: pd.DataFrame,
                         target_column: str = 'aqi') -> Dict[str, Any]:
        """Evaluate ensemble performance against individual models"""
        
        results = {}
        true_values = test_data[target_column].values
        
        # Evaluate individual models
        for model_name in self.models.keys():
            predictions = self.predict_single_model(model_name, test_data)
            mae = mean_absolute_error(true_values, predictions)
            rmse = np.sqrt(mean_squared_error(true_values, predictions))
            
            results[model_name] = {
                "mae": mae,
                "rmse": rmse,
                "predictions": predictions
            }
        
        # Evaluate ensemble
        ensemble_pred = self.predict_weighted_average(test_data)
        ensemble_mae = mean_absolute_error(true_values, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(true_values, ensemble_pred))
        
        results["ensemble"] = {
            "mae": ensemble_mae,
            "rmse": ensemble_rmse,
            "predictions": ensemble_pred,
            "weights": self.weights.copy()
        }
        
        # Log results to MLflow
        with mlflow.start_run():
            mlflow.log_params({"ensemble_weights": self.weights})
            
            for model_name, metrics in results.items():
                mlflow.log_metrics({
                    f"{model_name}_mae": metrics["mae"],
                    f"{model_name}_rmse": metrics["rmse"]
                })
        
        return results

# A/B Testing Framework
class ModelABTester:
    def __init__(self, model_a, model_b, traffic_split: float = 0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.metrics_a = []
        self.metrics_b = []
        
    def route_request(self, request_id: str) -> str:
        """Route request to model A or B based on traffic split"""
        # Simple hash-based routing for consistent assignment
        hash_value = hash(request_id) % 100
        return "model_a" if hash_value < (self.traffic_split * 100) else "model_b"
    
    def predict_with_routing(self, data: pd.DataFrame, request_id: str) -> Tuple[np.ndarray, str]:
        """Make prediction with A/B routing"""
        model_choice = self.route_request(request_id)
        
        if model_choice == "model_a":
            prediction = self.model_a.predict(data)
        else:
            prediction = self.model_b.predict(data)
        
        return prediction, model_choice
    
    def collect_metrics(self, 
                       prediction: np.ndarray,
                       actual: np.ndarray,
                       model_used: str,
                       response_time: float):
        """Collect metrics for A/B test analysis"""
        mae = mean_absolute_error(actual, prediction)
        rmse = np.sqrt(mean_squared_error(actual, prediction))
        
        metrics = {
            "mae": mae,
            "rmse": rmse,
            "response_time": response_time,
            "timestamp": pd.Timestamp.now()
        }
        
        if model_used == "model_a":
            self.metrics_a.append(metrics)
        else:
            self.metrics_b.append(metrics)
    
    def analyze_ab_test(self) -> Dict[str, Any]:
        """Analyze A/B test results"""
        if not self.metrics_a or not self.metrics_b:
            return {"error": "Insufficient data for analysis"}
        
        # Convert to DataFrames
        df_a = pd.DataFrame(self.metrics_a)
        df_b = pd.DataFrame(self.metrics_b)
        
        # Statistical analysis
        from scipy import stats
        
        mae_ttest = stats.ttest_ind(df_a['mae'], df_b['mae'])
        rmse_ttest = stats.ttest_ind(df_a['rmse'], df_b['rmse'])
        response_time_ttest = stats.ttest_ind(df_a['response_time'], df_b['response_time'])
        
        analysis = {
            "model_a_stats": {
                "mae_mean": df_a['mae'].mean(),
                "rmse_mean": df_a['rmse'].mean(),
                "response_time_mean": df_a['response_time'].mean(),
                "sample_size": len(df_a)
            },
            "model_b_stats": {
                "mae_mean": df_b['mae'].mean(),
                "rmse_mean": df_b['rmse'].mean(),
                "response_time_mean": df_b['response_time'].mean(),
                "sample_size": len(df_b)
            },
            "statistical_tests": {
                "mae_pvalue": mae_ttest.pvalue,
                "rmse_pvalue": rmse_ttest.pvalue,
                "response_time_pvalue": response_time_ttest.pvalue
            }
        }
        
        # Determine winner
        if mae_ttest.pvalue < 0.05:
            winner = "model_a" if df_a['mae'].mean() < df_b['mae'].mean() else "model_b"
            analysis["winner"] = winner
            analysis["significance"] = "significant"
        else:
            analysis["winner"] = "no_clear_winner"
            analysis["significance"] = "not_significant"
        
        return analysis
```

This comprehensive ML pipeline implementation provides:

1. **MLflow Integration**: Complete model tracking, registry, and deployment
2. **Advanced Models**: LSTM, Prophet, and XGBoost with production-ready implementations
3. **Distributed Training**: Ray-powered hyperparameter tuning and distributed training
4. **Model Ensemble**: Sophisticated ensemble methods with adaptive weighting
5. **A/B Testing**: Production-ready A/B testing framework for model comparison
6. **Feature Engineering**: Comprehensive time-series feature creation
7. **Performance Monitoring**: Detailed metrics tracking and model performance analysis

The implementation is production-ready and integrates seamlessly with the existing GeoAirQuality infrastructure.
