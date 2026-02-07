"""
Helper utilities for the framework.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import json
import warnings
logger = logging.getLogger(__name__)
class DataValidator:
    """Data validation utilities."""
    @staticmethod
    def validate_parameter_value(value: Any, param_name: str, 
                                min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Validate parameter value.
        Args:
            value: Parameter value
            param_name: Parameter name for logging
            min_val: Minimum valid value
            max_val: Maximum valid value
        Returns:
            Validated value
        """
        try:
            # Convert to float
            float_value = float(value)
            # Check for NaN or infinity
            if np.isnan(float_value) or np.isinf(float_value):
                logger.warning(f"Invalid value for {param_name}: {value}")
                return np.clip(0.5, min_val, max_val)
            # Clip to valid range
            clipped = np.clip(float_value, min_val, max_val)
            # Log if value was clipped
            if clipped != float_value:
                logger.debug(f"Value clipped for {param_name}: {float_value} -> {clipped}")
            return clipped
        except (ValueError, TypeError):
            logger.warning(f"Could not convert value for {param_name}: {value}")
            return np.clip(0.5, min_val, max_val)
    @staticmethod
    def validate_parameter_dict(param_dict: Dict[str, Any], 
                               required_keys: List[str] = None) -> Dict[str, float]:
        """
        Validate parameter dictionary.
        Args:
            param_dict: Parameter dictionary
            required_keys: List of required keys
        Returns:
            Validated dictionary with float values
        """
        validated = {}
        if required_keys:
            for key in required_keys:
                if key in param_dict:
                    validated[key] = DataValidator.validate_parameter_value(
                        param_dict[key], key
                    )
                else:
                    logger.warning(f"Missing required parameter: {key}")
                    validated[key] = 0.5
        # Add any additional parameters
        for key, value in param_dict.items():
            if key not in validated:
                validated[key] = DataValidator.validate_parameter_value(value, key)
        return validated
class TimeSeriesProcessor:
    """Time series processing utilities."""
    @staticmethod
    def resample_time_series(data: pd.DataFrame, 
                            freq: str = '1H',
                            method: str = 'linear') -> pd.DataFrame:
        """
        Resample time series data.
        Args:
            data: DataFrame with datetime index
            freq: Resampling frequency
            method: Interpolation method
        Returns:
            Resampled DataFrame
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        # Resample
        resampled = data.resample(freq).mean()
        # Interpolate missing values
        if method == 'linear':
            resampled = resampled.interpolate(method='linear', limit_direction='both')
        elif method == 'time':
            resampled = resampled.interpolate(method='time', limit_direction='both')
        elif method == 'nearest':
            resampled = resampled.interpolate(method='nearest', limit_direction='both')
        else:
            resampled = resampled.ffill().bfill()
        return resampled
    @staticmethod
    def calculate_rolling_statistics(data: pd.Series,
                                    window: Union[int, str] = '24H',
                                    statistics: List[str] = None) -> pd.DataFrame:
        """
        Calculate rolling statistics for time series.
        Args:
            data: Time series data
            window: Rolling window size
            statistics: List of statistics to calculate
        Returns:
            DataFrame with rolling statistics
        """
        if statistics is None:
            statistics = ['mean', 'std', 'min', 'max', 'median']
        results = {}
        for stat in statistics:
            if stat == 'mean':
                results[f'rolling_{stat}'] = data.rolling(window).mean()
            elif stat == 'std':
                results[f'rolling_{stat}'] = data.rolling(window).std()
            elif stat == 'min':
                results[f'rolling_{stat}'] = data.rolling(window).min()
            elif stat == 'max':
                results[f'rolling_{stat}'] = data.rolling(window).max()
            elif stat == 'median':
                results[f'rolling_{stat}'] = data.rolling(window).median()
            elif stat == 'sum':
                results[f'rolling_{stat}'] = data.rolling(window).sum()
            elif stat == 'count':
                results[f'rolling_{stat}'] = data.rolling(window).count()
        return pd.DataFrame(results)
    @staticmethod
    def detect_anomalies(data: pd.Series,
                        window: int = 48,
                        threshold: float = 3.0) -> pd.Series:
        """
        Detect anomalies in time series.
        Args:
            data: Time series data
            window: Window for rolling statistics
            threshold: Z-score threshold
        Returns:
            Boolean series indicating anomalies
        """
        # Calculate rolling statistics
        rolling_mean = data.rolling(window=window, center=True, min_periods=1).mean()
        rolling_std = data.rolling(window=window, center=True, min_periods=1).std()
        # Calculate z-scores
        z_scores = (data - rolling_mean) / (rolling_std + 1e-10)
        # Detect anomalies
        anomalies = np.abs(z_scores) > threshold
        return anomalies
class ConfigManager:
    """Configuration management utilities."""
    @staticmethod
    def load_config(config_file: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        Args:
            config_file: Path to configuration file
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_file}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_file}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing config file: {e}")
            return {}
    @staticmethod
    def save_config(config: Dict[str, Any], config_file: str):
        """
        Save configuration to JSON file.
        Args:
            config: Configuration dictionary
            config_file: Path to configuration file
        """
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    @staticmethod
    def merge_configs(base_config: Dict[str, Any],
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configurations.
        Args:
            base_config: Base configuration
            override_config: Configuration with overrides
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        for key, value in override_config.items():
            if (key in merged and isinstance(merged[key], dict) 
                and isinstance(value, dict)):
                # Recursively merge dictionaries
                merged[key] = ConfigManager.merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged
class AlertManager:
    """Alert management utilities."""
    ALERT_LEVELS = {
        'normal': {
            'color': 'green',
            'priority': 0,
            'description': 'Normal conditions'
        },
        'elevated': {
            'color': 'yellow',
            'priority': 1,
            'description': 'Elevated activity'
        },
        'watch': {
            'color': 'orange',
            'priority': 2,
            'description': 'Watch - increased monitoring required'
        },
        'warning': {
            'color': 'red',
            'priority': 3,
            'description': 'Warning - immediate action required'
        }
    }
    @staticmethod
    def create_alert(alert_level: str, 
                    message: str,
                    parameters: Dict[str, float],
                    location: Optional[str] = None,
                    timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Create an alert dictionary.
        Args:
            alert_level: Alert level (normal, elevated, watch, warning)
            message: Alert message
            parameters: Triggering parameters
            location: Optional location
            timestamp: Optional timestamp
        Returns:
            Alert dictionary
        """
        if alert_level not in AlertManager.ALERT_LEVELS:
            alert_level = 'normal'
        alert_info = AlertManager.ALERT_LEVELS[alert_level]
        return {
            'timestamp': timestamp or datetime.now().isoformat(),
            'alert_level': alert_level,
            'message': message,
            'location': location,
            'parameters': parameters,
            'color': alert_info['color'],
            'priority': alert_info['priority'],
            'description': alert_info['description']
        }
    @staticmethod
    def should_escalate(current_level: str, new_level: str) -> bool:
        """
        Check if alert should be escalated.
        Args:
            current_level: Current alert level
            new_level: New alert level
        Returns:
            True if escalation is needed
        """
        levels = ['normal', 'elevated', 'watch', 'warning']
        if current_level not in levels:
            current_idx = 0
        else:
            current_idx = levels.index(current_level)
        if new_level not in levels:
            new_idx = 0
        else:
            new_idx = levels.index(new_level)
        return new_idx > current_idx
    @staticmethod
    def format_alert_message(alert: Dict[str, Any]) -> str:
        """
        Format alert as human-readable message.
        Args:
            alert: Alert dictionary
        Returns:
            Formatted message
        """
        timestamp = alert.get('timestamp', 'Unknown time')
        level = alert.get('alert_level', 'normal').upper()
        message = alert.get('message', 'No message')
        location = alert.get('location', 'Unknown location')
        formatted = f"[{timestamp}] {level} ALERT at {location}\n"
        formatted += f"Message: {message}\n"
        if 'parameters' in alert:
            formatted += "Parameters:\n"
            for param, value in alert['parameters'].items():
                formatted += f"  {param}: {value:.3f}\n"
        return formatted
class PerformanceMetrics:
    """Performance metrics calculation."""
    @staticmethod
    def calculate_metrics(predictions: np.ndarray,
                         actuals: np.ndarray,
                         threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate performance metrics.
        Args:
            predictions: Predicted probabilities
            actuals: Actual binary values (0 or 1)
            threshold: Classification threshold
        Returns:
            Dictionary of metrics
        """
        # Convert probabilities to binary predictions
        binary_preds = (predictions >= threshold).astype(int)
        # Calculate confusion matrix
        tp = np.sum((binary_preds == 1) & (actuals == 1))
        fp = np.sum((binary_preds == 1) & (actuals == 0))
        tn = np.sum((binary_preds == 0) & (actuals == 0))
        fn = np.sum((binary_preds == 0) & (actuals == 1))
        # Calculate metrics
        metrics = {}
        # Basic metrics
        metrics['true_positives'] = float(tp)
        metrics['false_positives'] = float(fp)
        metrics['true_negatives'] = float(tn)
        metrics['false_negatives'] = float(fn)
        # Derived metrics
        if tp + fn > 0:
            metrics['sensitivity'] = float(tp / (tp + fn))
        else:
            metrics['sensitivity'] = 0.0
        if tn + fp > 0:
            metrics['specificity'] = float(tn / (tn + fp))
        else:
            metrics['specificity'] = 0.0
        if tp + fp > 0:
            metrics['precision'] = float(tp / (tp + fp))
        else:
            metrics['precision'] = 0.0
        if (tp + fn) > 0 and (tn + fp) > 0:
            metrics['accuracy'] = float((tp + tn) / (tp + tn + fp + fn))
        else:
            metrics['accuracy'] = 0.0
        if metrics['precision'] + metrics['sensitivity'] > 0:
            metrics['f1_score'] = float(
                2 * (metrics['precision'] * metrics['sensitivity']) /
                (metrics['precision'] + metrics['sensitivity'])
            )
        else:
            metrics['f1_score'] = 0.0
        # ROC AUC (simplified)
        from sklearn.metrics import roc_auc_score
        try:
            metrics['roc_auc'] = float(roc_auc_score(actuals, predictions))
        except:
            metrics['roc_auc'] = 0.5
        return metrics
    @staticmethod
    def calculate_uncertainty_metrics(predictions: np.ndarray,
                                     uncertainties: np.ndarray) -> Dict[str, float]:
        """
        Calculate uncertainty-related metrics.
        Args:
            predictions: Predicted values
            uncertainties: Uncertainty estimates
        Returns:
            Dictionary of uncertainty metrics
        """
        metrics = {}
        if len(predictions) == 0:
            return metrics
        metrics['average_uncertainty'] = float(np.mean(uncertainties))
        metrics['max_uncertainty'] = float(np.max(uncertainties))
        metrics['min_uncertainty'] = float(np.min(uncertainties))
        metrics['uncertainty_std'] = float(np.std(uncertainties))
        # Uncertainty vs prediction correlation
        if len(predictions) > 1:
            corr = np.corrcoef(predictions, uncertainties)[0, 1]
            metrics['uncertainty_correlation'] = float(corr if not np.isnan(corr) else 0.0)
        return metrics
def setup_logging(log_file: Optional[str] = None,
                 log_level: str = 'INFO',
                 console: bool = True) -> logging.Logger:
    """
    Set up logging configuration.
    Args:
        log_file: Optional log file path
        log_level: Logging level
        console: Whether to log to console
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('seismo_framework')
    logger.setLevel(getattr(logging, log_level.upper()))
    # Clear existing handlers
    logger.handlers.clear()
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
def print_summary(results: Dict[str, Any], title: str = "Results Summary"):
    """
    Print results summary in readable format.
    Args:
        results: Results dictionary
        title: Summary title
    """
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60)
    def print_dict(data: Dict, indent: int = 0):
        for key, value in data.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                print_dict(value, indent + 1)
            elif isinstance(value, (int, float)):
                if 0 <= value <= 1:
                    print("  " * indent + f"{key}: {value:.3f}")
                else:
                    print("  " * indent + f"{key}: {value:.2f}")
            elif isinstance(value, list):
                if len(value) <= 5:
                    print("  " * indent + f"{key}: {value}")
                else:
                    print("  " * indent + f"{key}: [{value[0]}, ..., {value[-1]}] ({len(value)} items)")
            else:
                print("  " * indent + f"{key}: {value}")
    print_dict(results)
    print("="*60)