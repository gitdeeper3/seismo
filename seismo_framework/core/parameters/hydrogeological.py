"""
Hydrogeological Indicators (W) parameter module.
This module handles groundwater and hydrological analysis including:
- Groundwater level changes
- Radon emission measurements
- Water chemistry variations
- Spring discharge monitoring
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import warnings
logger = logging.getLogger(__name__)
class HydrogeologicalAnalyzer:
    """
    Analyzer for hydrogeological indicators parameter (W).
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize hydrogeological analyzer.
        Parameters
        ----------
        config : dict, optional
            Configuration parameters
        """
        self.config = config or {}
        self._setup_defaults()
    def _setup_defaults(self):
        """Setup default configuration."""
        defaults = {
            'radon_threshold_bq': 100.0,  # Bq/m³
            'water_level_threshold': 0.1,  # meters
            'seasonal_period_days': 365,
            'tidal_analysis': True,
            'anomaly_detection_sigma': 3.0,
        }
        defaults.update(self.config)
        self.config = defaults
    def calculate_parameters(self, water_data: pd.DataFrame,
                            radon_data: Optional[pd.DataFrame] = None,
                            chemistry_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Calculate hydrogeological parameters.
        Parameters
        ----------
        water_data : pd.DataFrame
            Water level/well data
        radon_data : pd.DataFrame, optional
            Radon concentration data
        chemistry_data : pd.DataFrame, optional
            Water chemistry data
        Returns
        -------
        dict
            Dictionary of calculated parameters
        """
        results = {}
        try:
            # 1. Water level analysis
            if not water_data.empty:
                water_params = self._analyze_water_data(water_data)
                results.update(water_params)
            # 2. Radon analysis
            if radon_data is not None and not radon_data.empty:
                radon_params = self._analyze_radon_data(radon_data)
                results.update(radon_params)
            # 3. Chemistry analysis
            if chemistry_data is not None and not chemistry_data.empty:
                chem_params = self._analyze_chemistry_data(chemistry_data)
                results.update(chem_params)
            # 4. Combined hydrogeological index
            if results:
                results['hydrogeological_index'] = self._calculate_hydro_index(results)
            # 5. Uncertainty estimates
            results['uncertainties'] = self._estimate_uncertainties(results)
            logger.info(f"Hydrogeological parameters calculated: {len(results)} metrics")
        except Exception as e:
            logger.error(f"Error calculating hydrogeological parameters: {e}")
            return self._get_default_parameters()
        return results
    def _analyze_water_data(self, water_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze groundwater level data."""
        results = {}
        # Find water level column
        level_col = None
        for col in ['water_level', 'level', 'value', 'depth']:
            if col in water_data.columns:
                level_col = col
                break
        if level_col is None:
            return results
        values = water_data[level_col].dropna().values
        if len(values) < 10:
            return results
        # Basic statistics
        results['water_level_mean'] = float(np.mean(values))
        results['water_level_std'] = float(np.std(values))
        results['water_level_range'] = float(np.max(values) - np.min(values))
        # Trend analysis
        if 'time' in water_data.columns and len(values) > 1:
            try:
                times = pd.to_datetime(water_data['time']).values
                valid_idx = ~np.isnan(values)
                if np.sum(valid_idx) > 2:
                    # Linear trend
                    times_sec = times[valid_idx].astype('datetime64[s]').astype(float)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        times_sec, values[valid_idx]
                    )
                    results['water_trend_slope'] = float(slope)
                    results['water_trend_pvalue'] = float(p_value)
                    # Convert to mm/day
                    trend_mm_per_day = slope * 86400 * 1000
                    results['water_trend_mm_per_day'] = float(trend_mm_per_day)
                    # Remove trend for residual analysis
                    detrended = values[valid_idx] - (slope * times_sec + intercept)
                    results['water_detrended_std'] = float(np.std(detrended))
            except:
                pass
        # Seasonal/tidal analysis
        if 'time' in water_data.columns and len(values) >= 30:
            seasonal_params = self._analyze_seasonal_patterns(water_data, level_col)
            results.update(seasonal_params)
        # Anomaly detection
        anomalies = self._detect_anomalies(values)
        results['water_anomaly_count'] = float(len(anomalies))
        results['water_anomaly_ratio'] = float(len(anomalies) / len(values))
        return results
    def _analyze_radon_data(self, radon_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze radon concentration data."""
        results = {}
        # Find radon column
        radon_col = None
        for col in ['radon', 'rn222', 'concentration', 'value']:
            if col in radon_data.columns:
                radon_col = col
                break
        if radon_col is None:
            return results
        values = radon_data[radon_col].dropna().values
        if len(values) < 5:
            return results
        # Basic statistics
        results['radon_mean'] = float(np.mean(values))
        results['radon_std'] = float(np.std(values))
        results['radon_max'] = float(np.max(values))
        results['radon_min'] = float(np.min(values))
        # Threshold exceedance
        threshold = self.config['radon_threshold_bq']
        exceedances = values > threshold
        results['radon_exceedance_ratio'] = float(np.mean(exceedances))
        results['radon_exceedance_count'] = float(np.sum(exceedances))
        # Trend analysis
        if 'time' in radon_data.columns and len(values) > 2:
            try:
                times = pd.to_datetime(radon_data['time']).values
                valid_idx = ~np.isnan(values)
                if np.sum(valid_idx) > 2:
                    times_sec = times[valid_idx].astype('datetime64[s]').astype(float)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        times_sec, values[valid_idx]
                    )
                    results['radon_trend_slope'] = float(slope)
                    results['radon_trend_pvalue'] = float(p_value)
                    # Percent change per day
                    if intercept != 0:
                        percent_change_per_day = (slope / abs(intercept)) * 86400 * 100
                        results['radon_trend_percent_per_day'] = float(percent_change_per_day)
            except:
                pass
        # Anomaly detection
        anomalies = self._detect_anomalies(values, method='iqr')
        results['radon_anomaly_count'] = float(len(anomalies))
        # Spike detection
        spikes = self._detect_spikes(values)
        results['radon_spike_count'] = float(len(spikes))
        return results
    def _analyze_chemistry_data(self, chemistry_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze water chemistry data."""
        results = {}
        if chemistry_data.empty:
            return results
        # Common chemistry parameters
        chem_params = {
            'pH': {'range': (6.0, 8.5)},
            'conductivity': {'range': (100, 2000)},  # μS/cm
            'temperature': {'range': (5, 25)},  # °C
            'turbidity': {'range': (0, 10)},  # NTU
            'chloride': {'range': (0, 100)},  # mg/L
            'sulfate': {'range': (0, 100)},  # mg/L
            'bicarbonate': {'range': (50, 300)},  # mg/L
        }
        for param, info in chem_params.items():
            if param in chemistry_data.columns:
                values = chemistry_data[param].dropna().values
                if len(values) > 0:
                    results[f'{param}_mean'] = float(np.mean(values))
                    results[f'{param}_std'] = float(np.std(values))
                    # Check if values are within normal range
                    normal_range = info['range']
                    in_range = np.sum((values >= normal_range[0]) & (values <= normal_range[1]))
                    results[f'{param}_in_range_ratio'] = float(in_range / len(values))
        # Ionic ratios (if available)
        if 'Ca' in chemistry_data.columns and 'Mg' in chemistry_data.columns:
            ca_values = chemistry_data['Ca'].dropna().values
            mg_values = chemistry_data['Mg'].dropna().values
            if len(ca_values) > 0 and len(mg_values) > 0:
                # Simple Ca/Mg ratio
                min_len = min(len(ca_values), len(mg_values))
                ca_mg_ratio = ca_values[:min_len] / (mg_values[:min_len] + 1e-10)
                results['ca_mg_ratio_mean'] = float(np.mean(ca_mg_ratio))
                results['ca_mg_ratio_std'] = float(np.std(ca_mg_ratio))
        return results
    def _analyze_seasonal_patterns(self, data: pd.DataFrame, value_col: str) -> Dict[str, float]:
        """Analyze seasonal and tidal patterns in time series."""
        results = {}
        try:
            times = pd.to_datetime(data['time']).values
            values = data[value_col].values
            valid_idx = ~(np.isnan(times) | np.isnan(values))
            times = times[valid_idx]
            values = values[valid_idx]
            if len(values) < 100:
                return results
            # Convert to daily averages if needed
            times_dt = pd.to_datetime(times)
            df = pd.DataFrame({'time': times_dt, 'value': values})
            daily = df.set_index('time').resample('D').mean().dropna()
            if len(daily) < 60:
                return results
            daily_values = daily['value'].values
            # FFT for periodicity analysis
            n = len(daily_values)
            if n > 10:
                # Remove linear trend
                x = np.arange(n)
                slope, intercept = np.polyfit(x, daily_values, 1)
                detrended = daily_values - (slope * x + intercept)
                # FFT
                fft_values = np.fft.fft(detrended)
                freqs = np.fft.fftfreq(n, d=1)  # Daily sampling
                # Find dominant frequencies (excluding DC component)
                magnitude = np.abs(fft_values[1:n//2])
                frequencies = freqs[1:n//2]
                if len(magnitude) > 0:
                    # Find peaks
                    peaks, properties = signal.find_peaks(magnitude, 
                                                        height=np.mean(magnitude))
                    if len(peaks) > 0:
                        # Get strongest peak
                        strongest_idx = peaks[np.argmax(properties['peak_heights'])]
                        dominant_freq = frequencies[strongest_idx]
                        if dominant_freq > 0:
                            dominant_period = 1 / dominant_freq  # in days
                            results['dominant_period_days'] = float(dominant_period)
                            # Check if it's near annual (365 days) or semi-annual (182.5)
                            if abs(dominant_period - 365) < 30:
                                results['seasonal_strength'] = float(
                                    magnitude[strongest_idx] / np.sum(magnitude)
                                )
                            elif abs(dominant_period - 182.5) < 15:
                                results['semi_annual_strength'] = float(
                                    magnitude[strongest_idx] / np.sum(magnitude)
                                )
        except Exception as e:
            logger.warning(f"Error in seasonal analysis: {e}")
        return results
    def _detect_anomalies(self, values: np.ndarray, 
                         method: str = 'sigma') -> np.ndarray:
        """Detect anomalies in time series."""
        if len(values) < 10:
            return np.array([], dtype=bool)
        if method == 'sigma':
            # Sigma-based anomaly detection
            mean = np.mean(values)
            std = np.std(values)
            threshold = self.config['anomaly_detection_sigma']
            return np.abs(values - mean) > (threshold * std)
        elif method == 'iqr':
            # IQR-based anomaly detection
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return (values < lower_bound) | (values > upper_bound)
        else:
            return np.zeros(len(values), dtype=bool)
    def _detect_spikes(self, values: np.ndarray, 
                      threshold_std: float = 3.0) -> np.ndarray:
        """Detect spikes in time series."""
        if len(values) < 10:
            return np.array([], dtype=bool)
        # Calculate differences
        diffs = np.abs(np.diff(values))
        if len(diffs) == 0:
            return np.array([], dtype=bool)
        # Detect large changes
        spike_threshold = np.mean(diffs) + threshold_std * np.std(diffs)
        spike_indices = diffs > spike_threshold
        # Convert to same length as input
        spikes = np.zeros(len(values), dtype=bool)
        spikes[1:] = spike_indices
        return spikes
    def _calculate_hydro_index(self, parameters: Dict[str, float]) -> float:
        """Calculate combined hydrogeological index."""
        # Weights for different parameters
        weights = {
            'radon_exceedance_ratio': 0.25,
            'water_anomaly_ratio': 0.20,
            'water_trend_mm_per_day': 0.15,
            'radon_trend_percent_per_day': 0.15,
            'water_level_std': 0.10,
            'radon_spike_count': 0.05,
            'ca_mg_ratio_std': 0.05,
            'seasonal_strength': 0.05,
        }
        index = 0.5  # Baseline
        for param, weight in weights.items():
            if param in parameters:
                value = parameters[param]
                # Normalize parameter
                if param == 'radon_exceedance_ratio':
                    normalized = min(1.0, value * 5)  # 20% exceedance = 1.0
                elif param == 'water_anomaly_ratio':
                    normalized = min(1.0, value * 10)  # 10% anomalies = 1.0
                elif param == 'water_trend_mm_per_day':
                    normalized = min(1.0, abs(value) / 10.0)  # 10 mm/day = 1.0
                elif param == 'radon_trend_percent_per_day':
                    normalized = min(1.0, abs(value) / 50.0)  # 50%/day = 1.0
                elif param == 'water_level_std':
                    normalized = min(1.0, value / 0.5)  # 0.5m std = 1.0
                elif param == 'radon_spike_count':
                    normalized = min(1.0, value / 10.0)  # 10 spikes = 1.0
                elif param == 'ca_mg_ratio_std':
                    normalized = min(1.0, value / 2.0)  # std of 2 = 1.0
                elif param == 'seasonal_strength':
                    normalized = 1.0 - min(1.0, value)  # Less seasonal = more concerning
                else:
                    normalized = 0.5
                index += weight * (normalized - 0.5)
        # Ensure index is between 0 and 1
        index = max(0.0, min(1.0, index))
        return float(index)
    def _estimate_uncertainties(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Estimate uncertainties for hydrogeological parameters."""
        uncertainties = {}
        for param, value in parameters.items():
            if param.endswith('_std'):
                # Use standard deviation as uncertainty
                uncertainties[param.replace('_std', '')] = value
            elif param == 'hydrogeological_index':
                # Combined uncertainty
                uncertainties[param] = 0.2  # 20% uncertainty
            elif param.endswith('_ratio'):
                # Ratio parameters: binomial uncertainty
                n = 100  # Assumed sample size
                p = value
                if 0 < p < 1:
                    uncertainty = np.sqrt(p * (1 - p) / n)
                    uncertainties[param] = float(uncertainty)
                else:
                    uncertainties[param] = 0.1
            else:
                # Default uncertainty
                if value != 0:
                    uncertainties[param] = abs(value) * 0.15  # 15% relative
                else:
                    uncertainties[param] = 0.1
        return uncertainties
    def _get_default_parameters(self) -> Dict[str, float]:
        """Return default parameters when calculation fails."""
        return {
            'water_level_mean': 0.0,
            'water_level_std': 0.1,
            'radon_mean': 50.0,
            'radon_std': 10.0,
            'hydrogeological_index': 0.5,
            'uncertainties': {
                'hydrogeological_index': 0.2,
                'water_level_mean': 0.05,
                'radon_mean': 5.0,
            }
        }
    def analyze_time_series(self, water_data: pd.DataFrame,
                           window_days: int = 30,
                           step_days: int = 7) -> pd.DataFrame:
        """
        Analyze hydrogeological parameters over time.
        Parameters
        ----------
        water_data : pd.DataFrame
            Water level time series
        window_days : int
            Window size in days
        step_days : int
            Step size in days
        Returns
        -------
        pd.DataFrame
            Time series of parameters
        """
        if water_data.empty or 'time' not in water_data.columns:
            return pd.DataFrame()
        try:
            water_data = water_data.copy()
            water_data['time'] = pd.to_datetime(water_data['time'])
            water_data = water_data.sort_values('time')
            start_time = water_data['time'].min()
            end_time = water_data['time'].max()
            results = []
            current_start = start_time
            while current_start < end_time:
                current_end = current_start + timedelta(days=window_days)
                window_data = water_data[
                    (water_data['time'] >= current_start) & 
                    (water_data['time'] < current_end)
                ]
                if len(window_data) >= 10:
                    params = self.calculate_parameters(window_data)
                    params['window_start'] = current_start
                    params['window_end'] = current_end
                    params['data_points'] = len(window_data)
                    results.append(params)
                current_start += timedelta(days=step_days)
            if results:
                return pd.DataFrame(results)
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error in hydrogeological time series analysis: {e}")
            return pd.DataFrame()
# Convenience functions
def calculate_radon_anomalies(radon_data: pd.DataFrame, 
                             threshold_bq: float = 100.0) -> Tuple[float, int]:
    """
    Calculate radon anomaly metrics.
    Parameters
    ----------
    radon_data : pd.DataFrame
        Radon concentration data
    threshold_bq : float
        Threshold for exceedance (Bq/m³)
    Returns
    -------
    tuple
        (exceedance_ratio, exceedance_count)
    """
    analyzer = HydrogeologicalAnalyzer({'radon_threshold_bq': threshold_bq})
    params = analyzer.calculate_parameters(pd.DataFrame(), radon_data)
    return (params.get('radon_exceedance_ratio', 0.0), 
            int(params.get('radon_exceedance_count', 0)))
def calculate_hydrogeological_index(water_data: pd.DataFrame,
                                   radon_data: Optional[pd.DataFrame] = None) -> float:
    """
    Calculate hydrogeological index from water and radon data.
    Parameters
    ----------
    water_data : pd.DataFrame
        Water level data
    radon_data : pd.DataFrame, optional
        Radon concentration data
    Returns
    -------
    float
        Hydrogeological index (0-1)
    """
    analyzer = HydrogeologicalAnalyzer()
    params = analyzer.calculate_parameters(water_data, radon_data)
    return params.get('hydrogeological_index', 0.5)