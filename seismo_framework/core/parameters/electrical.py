"""
Electrical Signals (E) parameter module.
This module handles electrical signal analysis including:
- Electrical resistivity measurements
- Self-potential monitoring
- Telluric currents
- Electromagnetic emissions
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import warnings
logger = logging.getLogger(__name__)
class ElectricalAnalyzer:
    """
    Analyzer for electrical signals parameter (E).
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._setup_defaults()
    def _setup_defaults(self):
        defaults = {
            'resistivity_threshold': 0.1,  # 10% change
            'spike_threshold_std': 3.0,
            'frequency_bands': {
                'ulf': (0.001, 1.0),     # Ultra Low Frequency (mHz)
                'elf': (1.0, 100.0),     # Extremely Low Frequency (Hz)
                'lf': (100.0, 10000.0),  # Low Frequency (Hz)
            },
            'noise_floor_db': -120,
        }
        defaults.update(self.config)
        self.config = defaults
    def calculate_parameters(self, resistivity_data: pd.DataFrame,
                            sp_data: Optional[pd.DataFrame] = None,
                            em_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Calculate electrical parameters.
        Parameters
        ----------
        resistivity_data : pd.DataFrame
            Resistivity measurements
        sp_data : pd.DataFrame, optional
            Self-potential data
        em_data : pd.DataFrame, optional
            Electromagnetic data
        Returns
        -------
        dict
            Dictionary of calculated parameters
        """
        results = {}
        try:
            # 1. Resistivity analysis
            if not resistivity_data.empty:
                res_params = self._analyze_resistivity(resistivity_data)
                results.update(res_params)
            # 2. Self-potential analysis
            if sp_data is not None and not sp_data.empty:
                sp_params = self._analyze_self_potential(sp_data)
                results.update(sp_params)
            # 3. EM analysis
            if em_data is not None and not em_data.empty:
                em_params = self._analyze_electromagnetic(em_data)
                results.update(em_params)
            # 4. Combined electrical index
            if results:
                results['electrical_index'] = self._calculate_electrical_index(results)
            # 5. Uncertainty estimates
            results['uncertainties'] = self._estimate_uncertainties(results)
            logger.info(f"Electrical parameters calculated: {len(results)} metrics")
        except Exception as e:
            logger.error(f"Error calculating electrical parameters: {e}")
            return self._get_default_parameters()
        return results
    def _analyze_resistivity(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze resistivity data."""
        results = {}
        # Find resistivity column
        res_col = None
        for col in ['resistivity', 'rho', 'value', 'apparent_resistivity']:
            if col in data.columns:
                res_col = col
                break
        if res_col is None:
            return results
        values = data[res_col].dropna().values
        if len(values) < 5:
            return results
        # Basic statistics
        results['resistivity_mean'] = float(np.mean(values))
        results['resistivity_std'] = float(np.std(values))
        results['resistivity_cv'] = float(results['resistivity_std'] / 
                                         results['resistivity_mean'] if 
                                         results['resistivity_mean'] != 0 else 0)
        # Percent change from baseline
        if 'baseline' in self.config:
            baseline = self.config['baseline']
            percent_change = ((results['resistivity_mean'] - baseline) / baseline) * 100
            results['resistivity_percent_change'] = float(percent_change)
        # Trend analysis
        if 'time' in data.columns and len(values) > 2:
            try:
                times = pd.to_datetime(data['time']).values
                valid_idx = ~np.isnan(values)
                if np.sum(valid_idx) > 2:
                    times_sec = times[valid_idx].astype('datetime64[s]').astype(float)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        times_sec, values[valid_idx]
                    )
                    results['resistivity_trend_slope'] = float(slope)
                    results['resistivity_trend_pvalue'] = float(p_value)
                    # Percent change per day
                    if intercept != 0:
                        percent_change_per_day = (slope / abs(intercept)) * 86400 * 100
                        results['resistivity_trend_percent_per_day'] = float(
                            percent_change_per_day
                        )
            except:
                pass
        # Anomaly detection
        anomalies = self._detect_resistivity_anomalies(values)
        results['resistivity_anomaly_count'] = float(len(anomalies))
        results['resistivity_anomaly_ratio'] = float(len(anomalies) / len(values))
        # Spike detection
        spikes = self._detect_spikes(values)
        results['resistivity_spike_count'] = float(len(spikes))
        return results
    def _analyze_self_potential(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze self-potential data."""
        results = {}
        # Find SP column
        sp_col = None
        for col in ['self_potential', 'sp', 'potential', 'voltage']:
            if col in data.columns:
                sp_col = col
                break
        if sp_col is None:
            return results
        values = data[sp_col].dropna().values
        if len(values) < 10:
            return results
        # Basic statistics
        results['sp_mean'] = float(np.mean(values))
        results['sp_std'] = float(np.std(values))
        results['sp_range'] = float(np.max(values) - np.min(values))
        # Gradient analysis
        if len(values) > 1:
            gradients = np.abs(np.diff(values))
            results['sp_gradient_mean'] = float(np.mean(gradients))
            results['sp_gradient_max'] = float(np.max(gradients))
        # Spectral analysis for ULF signals
        if 'time' in data.columns and len(values) >= 100:
            try:
                times = pd.to_datetime(data['time']).values
                valid_idx = ~np.isnan(values)
                if np.sum(valid_idx) >= 100:
                    times_sec = times[valid_idx].astype('datetime64[s]').astype(float)
                    values_valid = values[valid_idx]
                    # Resample to regular intervals if needed
                    time_diff = np.diff(times_sec)
                    if np.std(time_diff) / np.mean(time_diff) < 0.1:  # Regular enough
                        spectral_params = self._analyze_sp_spectrum(values_valid, 
                                                                   np.mean(time_diff))
                        results.update(spectral_params)
            except:
                pass
        return results
    def _analyze_electromagnetic(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze electromagnetic data."""
        results = {}
        # Find EM field columns
        field_cols = {}
        for field in ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'magnitude']:
            if field in data.columns:
                field_cols[field] = data[field].dropna().values
        if not field_cols:
            return results
        # Analyze each field component
        for field, values in field_cols.items():
            if len(values) > 5:
                results[f'{field}_mean'] = float(np.mean(values))
                results[f'{field}_std'] = float(np.std(values))
                results[f'{field}_max'] = float(np.max(np.abs(values)))
        # Combined field strength
        if 'ex' in field_cols and 'ey' in field_cols:
            ex_mean = results.get('ex_mean', 0)
            ey_mean = results.get('ey_mean', 0)
            results['horizontal_em_strength'] = float(np.sqrt(ex_mean**2 + ey_mean**2))
        # Spectral analysis in different bands
        if 'time' in data.columns and len(data) >= 100:
            for field, values in field_cols.items():
                if len(values) >= 100:
                    try:
                        spectral_power = self._analyze_em_spectrum(values)
                        for band, power in spectral_power.items():
                            results[f'{field}_{band}_power'] = float(power)
                    except:
                        pass
        return results
    def _detect_resistivity_anomalies(self, values: np.ndarray) -> np.ndarray:
        """Detect anomalies in resistivity data."""
        if len(values) < 20:
            return np.array([], dtype=bool)
        # Use rolling statistics
        window = min(20, len(values) // 2)
        anomalies = np.zeros(len(values), dtype=bool)
        for i in range(len(values)):
            start = max(0, i - window)
            end = min(len(values), i + window + 1)
            window_values = values[start:end]
            window_mean = np.mean(window_values)
            window_std = np.std(window_values)
            if window_std > 0:
                z_score = abs(values[i] - window_mean) / window_std
                if z_score > self.config['spike_threshold_std']:
                    anomalies[i] = True
        return anomalies
    def _detect_spikes(self, values: np.ndarray) -> np.ndarray:
        """Detect spikes in electrical signals."""
        if len(values) < 10:
            return np.array([], dtype=bool)
        # Calculate median absolute deviation
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad == 0:
            return np.array([], dtype=bool)
        # Modified Z-score
        modified_z_scores = 0.6745 * (values - median) / mad
        return np.abs(modified_z_scores) > self.config['spike_threshold_std']
    def _analyze_sp_spectrum(self, values: np.ndarray, 
                            sampling_interval: float) -> Dict[str, float]:
        """Analyze spectrum of self-potential data."""
        results = {}
        # Remove DC component
        values_detrended = values - np.mean(values)
        # FFT
        n = len(values_detrended)
        fft_vals = fft.fft(values_detrended)
        freqs = fft.fftfreq(n, d=sampling_interval)
        # Calculate power spectral density
        psd = np.abs(fft_vals) ** 2
        # Analyze in different frequency bands
        for band_name, (f_low, f_high) in self.config['frequency_bands'].items():
            mask = (np.abs(freqs) >= f_low) & (np.abs(freqs) <= f_high)
            if np.any(mask):
                band_power = np.mean(psd[mask])
                results[f'sp_{band_name}_power'] = float(band_power)
        # Dominant frequency
        if n > 10:
            positive_freqs = freqs[:n//2]
            positive_psd = psd[:n//2]
            if len(positive_psd) > 0:
                dominant_idx = np.argmax(positive_psd)
                dominant_freq = positive_freqs[dominant_idx]
                results['sp_dominant_frequency'] = float(dominant_freq)
                results['sp_dominant_power'] = float(positive_psd[dominant_idx])
        return results
    def _analyze_em_spectrum(self, values: np.ndarray) -> Dict[str, float]:
        """Analyze spectrum of electromagnetic data."""
        results = {}
        # Simple spectral analysis
        n = len(values)
        if n < 100:
            return results
        # FFT
        fft_vals = fft.fft(values - np.mean(values))
        freqs = fft.fftfreq(n)
        # Power in different bands
        for band_name, (f_low, f_high) in self.config['frequency_bands'].items():
            # Convert to normalized frequency
            f_low_norm = f_low / (0.5 / freqs[1])  # Assuming sampling rate of 1 Hz
            f_high_norm = f_high / (0.5 / freqs[1])
            mask = (np.abs(freqs) >= f_low_norm) & (np.abs(freqs) <= f_high_norm)
            if np.any(mask):
                band_power = np.mean(np.abs(fft_vals[mask]) ** 2)
                results[band_name] = float(band_power)
        return results
    def _calculate_electrical_index(self, parameters: Dict[str, float]) -> float:
        """Calculate combined electrical index."""
        weights = {
            'resistivity_percent_change': 0.25,
            'resistivity_anomaly_ratio': 0.20,
            'sp_gradient_max': 0.15,
            'horizontal_em_strength': 0.15,
            'sp_ulf_power': 0.10,
            'resistivity_spike_count': 0.05,
            'sp_dominant_power': 0.05,
            'ex_elf_power': 0.05,
        }
        index = 0.5
        for param, weight in weights.items():
            if param in parameters:
                value = parameters[param]
                # Normalize parameter
                if param == 'resistivity_percent_change':
                    normalized = min(1.0, abs(value) / 20.0)  # 20% change = 1.0
                elif param == 'resistivity_anomaly_ratio':
                    normalized = min(1.0, value * 10)  # 10% anomalies = 1.0
                elif param == 'sp_gradient_max':
                    normalized = min(1.0, value / 10.0)  # 10 mV/m = 1.0
                elif param == 'horizontal_em_strength':
                    normalized = min(1.0, value / 100.0)  # 100 nT = 1.0
                elif param.endswith('_power'):
                    # Log scale for power
                    normalized = min(1.0, np.log10(max(1, value)) / 3.0)
                elif param == 'resistivity_spike_count':
                    normalized = min(1.0, value / 10.0)  # 10 spikes = 1.0
                else:
                    normalized = 0.5
                index += weight * (normalized - 0.5)
        index = max(0.0, min(1.0, index))
        return float(index)
    def _estimate_uncertainties(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Estimate uncertainties for electrical parameters."""
        uncertainties = {}
        for param, value in parameters.items():
            if param.endswith('_std'):
                uncertainties[param.replace('_std', '')] = value
            elif param == 'electrical_index':
                uncertainties[param] = 0.2
            elif param.endswith('_power'):
                # Power measurements: sqrt(N) statistics
                uncertainties[param] = value * 0.3  # 30% for spectral estimates
            else:
                if value != 0:
                    uncertainties[param] = abs(value) * 0.2
                else:
                    uncertainties[param] = 0.1
        return uncertainties
    def _get_default_parameters(self) -> Dict[str, float]:
        return {
            'resistivity_mean': 100.0,
            'resistivity_std': 10.0,
            'sp_mean': 0.0,
            'sp_std': 1.0,
            'electrical_index': 0.5,
            'uncertainties': {
                'electrical_index': 0.2,
                'resistivity_mean': 5.0,
                'sp_mean': 0.5,
            }
        }
# Convenience functions
def calculate_resistivity_change(resistivity_data: pd.DataFrame, 
                                baseline: float) -> float:
    analyzer = ElectricalAnalyzer({'baseline': baseline})
    params = analyzer.calculate_parameters(resistivity_data)
    return params.get('resistivity_percent_change', 0.0)
def calculate_electrical_index(resistivity_data: pd.DataFrame,
                              sp_data: Optional[pd.DataFrame] = None) -> float:
    analyzer = ElectricalAnalyzer()
    params = analyzer.calculate_parameters(resistivity_data, sp_data)
    return params.get('electrical_index', 0.5)