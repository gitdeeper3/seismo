"""
Magnetic Anomalies (M) parameter module.
This module handles magnetic field analysis including:
- Geomagnetic field variations
- Magnetic anomaly detection
- Secular variation monitoring
- Magnetic susceptibility changes
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import warnings
logger = logging.getLogger(__name__)
class MagneticAnalyzer:
    """
    Analyzer for magnetic anomalies parameter (M).
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._setup_defaults()
    def _setup_defaults(self):
        defaults = {
            'anomaly_threshold_nT': 10.0,
            'gradient_threshold': 1.0,  # nT/km
            'diurnal_correction': True,
            'noise_level_nT': 0.1,
        }
        defaults.update(self.config)
        self.config = defaults
    def calculate_parameters(self, magnetic_data: pd.DataFrame,
                            baseline_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Calculate magnetic parameters.
        Parameters
        ----------
        magnetic_data : pd.DataFrame
            Magnetic field measurements
        baseline_data : pd.DataFrame, optional
            Baseline/reference data
        Returns
        -------
        dict
            Dictionary of calculated parameters
        """
        results = {}
        try:
            # 1. Basic magnetic field analysis
            if not magnetic_data.empty:
                mag_params = self._analyze_magnetic_field(magnetic_data)
                results.update(mag_params)
            # 2. Anomaly detection
            anomaly_params = self._detect_magnetic_anomalies(magnetic_data, baseline_data)
            results.update(anomaly_params)
            # 3. Gradient analysis
            gradient_params = self._analyze_magnetic_gradients(magnetic_data)
            results.update(gradient_params)
            # 4. Combined magnetic index
            if results:
                results['magnetic_index'] = self._calculate_magnetic_index(results)
            # 5. Uncertainty estimates
            results['uncertainties'] = self._estimate_uncertainties(results)
            logger.info(f"Magnetic parameters calculated: {len(results)} metrics")
        except Exception as e:
            logger.error(f"Error calculating magnetic parameters: {e}")
            return self._get_default_parameters()
        return results
    def _analyze_magnetic_field(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze magnetic field measurements."""
        results = {}
        # Find magnetic field components
        components = {}
        for comp in ['bx', 'by', 'bz', 'total', 'magnitude', 'f']:
            if comp in data.columns:
                components[comp] = data[comp].dropna().values
        if not components:
            return results
        # Analyze each component
        for comp, values in components.items():
            if len(values) > 5:
                results[f'{comp}_mean'] = float(np.mean(values))
                results[f'{comp}_std'] = float(np.std(values))
                results[f'{comp}_range'] = float(np.max(values) - np.min(values))
                # Diurnal variation removal if time data available
                if 'time' in data.columns and len(values) > 24:
                    try:
                        detrended = self._remove_diurnal_variation(data, comp)
                        if detrended is not None:
                            results[f'{comp}_detrended_std'] = float(np.std(detrended))
                    except:
                        pass
        # Total field strength
        if 'total' in components:
            results['field_strength_mean'] = results['total_mean']
            results['field_strength_std'] = results['total_std']
        elif all(c in components for c in ['bx', 'by', 'bz']):
            bx_mean = results.get('bx_mean', 0)
            by_mean = results.get('by_mean', 0)
            bz_mean = results.get('bz_mean', 0)
            total_strength = np.sqrt(bx_mean**2 + by_mean**2 + bz_mean**2)
            results['field_strength_mean'] = float(total_strength)
        return results
    def _remove_diurnal_variation(self, data: pd.DataFrame, 
                                 component: str) -> Optional[np.ndarray]:
        """Remove diurnal variation from magnetic data."""
        try:
            times = pd.to_datetime(data['time']).values
            values = data[component].values
            valid_idx = ~(np.isnan(times) | np.isnan(values))
            if np.sum(valid_idx) < 48:  # Need at least 2 days of hourly data
                return None
            times_valid = times[valid_idx]
            values_valid = values[valid_idx]
            # Convert to hours of day
            hours = np.array([t.hour + t.minute/60 for t in times_valid])
            # Calculate daily pattern (24-hour moving average)
            daily_pattern = np.zeros(24)
            hour_counts = np.zeros(24)
            for hour, value in zip(hours.astype(int), values_valid):
                if 0 <= hour < 24:
                    daily_pattern[hour] += value
                    hour_counts[hour] += 1
            # Average pattern
            hour_counts[hour_counts == 0] = 1  # Avoid division by zero
            daily_pattern /= hour_counts
            # Remove pattern
            detrended = []
            for t, value in zip(times_valid, values_valid):
                hour = t.hour
                detrended_value = value - daily_pattern[hour]
                detrended.append(detrended_value)
            return np.array(detrended)
        except:
            return None
    def _detect_magnetic_anomalies(self, data: pd.DataFrame,
                                  baseline: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Detect magnetic anomalies."""
        results = {}
        # Find total field column
        field_col = None
        for col in ['total', 'magnitude', 'f', 'bz']:
            if col in data.columns:
                field_col = col
                break
        if field_col is None:
            return results
        values = data[field_col].dropna().values
        if len(values) < 10:
            return results
        # Simple anomaly detection
        mean_value = np.mean(values)
        std_value = np.std(values)
        threshold = self.config['anomaly_threshold_nT']
        # Absolute anomalies
        abs_anomalies = np.abs(values - mean_value) > threshold
        results['absolute_anomaly_count'] = float(np.sum(abs_anomalies))
        results['absolute_anomaly_ratio'] = float(np.mean(abs_anomalies))
        # Relative anomalies (if baseline available)
        if baseline is not None and field_col in baseline.columns:
            baseline_values = baseline[field_col].dropna().values
            if len(baseline_values) > 0:
                baseline_mean = np.mean(baseline_values)
                rel_anomalies = np.abs(values - baseline_mean) > threshold
                results['relative_anomaly_count'] = float(np.sum(rel_anomalies))
                results['relative_anomaly_ratio'] = float(np.mean(rel_anomalies))
        # Spike detection
        spikes = self._detect_magnetic_spikes(values)
        results['spike_count'] = float(len(spikes))
        results['spike_ratio'] = float(len(spikes) / len(values))
        return results
    def _detect_magnetic_spikes(self, values: np.ndarray) -> np.ndarray:
        """Detect spikes in magnetic data."""
        if len(values) < 10:
            return np.array([], dtype=bool)
        # Use gradient-based spike detection
        gradients = np.abs(np.diff(values))
        if len(gradients) == 0:
            return np.array([], dtype=bool)
        median_grad = np.median(gradients)
        mad_grad = np.median(np.abs(gradients - median_grad))
        if mad_grad == 0:
            return np.array([], dtype=bool)
        # Modified Z-score for gradients
        modified_z_scores = 0.6745 * (gradients - median_grad) / mad_grad
        spike_gradients = modified_z_scores > self.config['spike_threshold_std']
        # Convert to same length as input
        spikes = np.zeros(len(values), dtype=bool)
        spikes[1:] = spike_gradients
        return spikes
    def _analyze_magnetic_gradients(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze spatial magnetic gradients."""
        results = {}
        # Check if spatial data is available
        if not all(col in data.columns for col in ['latitude', 'longitude']):
            return results
        field_col = None
        for col in ['total', 'magnitude', 'f', 'bz']:
            if col in data.columns:
                field_col = col
                break
        if field_col is None:
            return results
        # Extract valid data
        valid_idx = ~(data['latitude'].isna() | data['longitude'].isna() | 
                     data[field_col].isna())
        if np.sum(valid_idx) < 3:
            return results
        lats = data.loc[valid_idx, 'latitude'].values
        lons = data.loc[valid_idx, 'longitude'].values
        values = data.loc[valid_idx, field_col].values
        # Calculate spatial gradients
        gradients = []
        for i in range(len(lats)):
            distances = []
            diff_values = []
            for j in range(len(lats)):
                if i != j:
                    # Calculate distance
                    distance = self._haversine_distance(lats[i], lons[i], 
                                                       lats[j], lons[j])
                    value_diff = values[j] - values[i]
                    distances.append(distance)
                    diff_values.append(value_diff)
            if distances:
                # Weighted average gradient
                weights = 1.0 / np.array(distances)
                weights /= np.sum(weights)
                gradient = np.sum(np.array(diff_values) * weights)
                gradients.append(abs(gradient))
        if gradients:
            results['gradient_mean'] = float(np.mean(gradients))
            results['gradient_std'] = float(np.std(gradients))
            results['gradient_max'] = float(np.max(gradients))
            # Count of large gradients
            threshold = self.config['gradient_threshold']
            large_gradients = np.array(gradients) > threshold
            results['large_gradient_count'] = float(np.sum(large_gradients))
            results['large_gradient_ratio'] = float(np.mean(large_gradients))
        return results
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers."""
        R = 6371
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c
    def _calculate_magnetic_index(self, parameters: Dict[str, float]) -> float:
        """Calculate combined magnetic index."""
        weights = {
            'absolute_anomaly_ratio': 0.30,
            'relative_anomaly_ratio': 0.25,
            'gradient_max': 0.15,
            'spike_ratio': 0.10,
            'field_strength_std': 0.10,
            'large_gradient_ratio': 0.05,
            'bz_detrended_std': 0.05,
        }
        index = 0.5
        for param, weight in weights.items():
            if param in parameters:
                value = parameters[param]
                # Normalize parameter
                if param.endswith('_ratio'):
                    normalized = min(1.0, value * 5)  # 20% ratio = 1.0
                elif param == 'gradient_max':
                    normalized = min(1.0, value / 5.0)  # 5 nT/km = 1.0
                elif param == 'field_strength_std':
                    normalized = min(1.0, value / 10.0)  # 10 nT std = 1.0
                elif param == 'bz_detrended_std':
                    normalized = min(1.0, value / 5.0)  # 5 nT = 1.0
                else:
                    normalized = 0.5
                index += weight * (normalized - 0.5)
        index = max(0.0, min(1.0, index))
        return float(index)
    def _estimate_uncertainties(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Estimate uncertainties for magnetic parameters."""
        uncertainties = {}
        for param, value in parameters.items():
            if param.endswith('_std'):
                uncertainties[param.replace('_std', '')] = value
            elif param == 'magnetic_index':
                uncertainties[param] = 0.2
            elif param.endswith('_ratio'):
                # Binomial uncertainty
                n = 100
                p = value
                if 0 < p < 1:
                    uncertainty = np.sqrt(p * (1 - p) / n)
                    uncertainties[param] = float(uncertainty)
                else:
                    uncertainties[param] = 0.1
            else:
                if value != 0:
                    uncertainties[param] = abs(value) * 0.15
                else:
                    uncertainties[param] = 0.1
        return uncertainties
    def _get_default_parameters(self) -> Dict[str, float]:
        return {
            'field_strength_mean': 50000.0,
            'field_strength_std': 10.0,
            'absolute_anomaly_ratio': 0.0,
            'gradient_mean': 0.0,
            'magnetic_index': 0.5,
            'uncertainties': {
                'magnetic_index': 0.2,
                'field_strength_mean': 5.0,
                'absolute_anomaly_ratio': 0.05,
            }
        }
# Convenience functions
def detect_magnetic_anomalies(magnetic_data: pd.DataFrame,
                             threshold_nT: float = 10.0) -> Tuple[int, float]:
    analyzer = MagneticAnalyzer({'anomaly_threshold_nT': threshold_nT})
    params = analyzer.calculate_parameters(magnetic_data)
    return (int(params.get('absolute_anomaly_count', 0)),
            params.get('absolute_anomaly_ratio', 0.0))
def calculate_magnetic_index(magnetic_data: pd.DataFrame) -> float:
    analyzer = MagneticAnalyzer()
    params = analyzer.calculate_parameters(magnetic_data)
    return params.get('magnetic_index', 0.5)