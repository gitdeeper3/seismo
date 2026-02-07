"""
Rock Properties (R) parameter module.
This module handles rock physics analysis including:
- Seismic velocity measurements (Vp, Vs)
- Velocity ratio (Vp/Vs) analysis
- Attenuation (Q) measurements
- Density and elasticity calculations
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import warnings
logger = logging.getLogger(__name__)
class RockPropertiesAnalyzer:
    """
    Analyzer for rock properties parameter (R).
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._setup_defaults()
    def _setup_defaults(self):
        defaults = {
            'vp_vs_normal': 1.73,  # Normal Vp/Vs ratio for crustal rocks
            'vp_vs_threshold': 0.05,  # 5% change threshold
            'q_normal': 100.0,  # Normal Q value
            'density_normal': 2.7,  # g/cm³
            'poissons_normal': 0.25,
        }
        defaults.update(self.config)
        self.config = defaults
    def calculate_parameters(self, velocity_data: pd.DataFrame,
                            attenuation_data: Optional[pd.DataFrame] = None,
                            density_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Calculate rock properties parameters.
        Parameters
        ----------
        velocity_data : pd.DataFrame
            Seismic velocity measurements
        attenuation_data : pd.DataFrame, optional
            Attenuation (Q) measurements
        density_data : pd.DataFrame, optional
            Density measurements
        Returns
        -------
        dict
            Dictionary of calculated parameters
        """
        results = {}
        try:
            # 1. Velocity analysis
            if not velocity_data.empty:
                velocity_params = self._analyze_velocities(velocity_data)
                results.update(velocity_params)
            # 2. Attenuation analysis
            if attenuation_data is not None and not attenuation_data.empty:
                attenuation_params = self._analyze_attenuation(attenuation_data)
                results.update(attenuation_params)
            # 3. Density analysis
            if density_data is not None and not density_data.empty:
                density_params = self._analyze_density(density_data)
                results.update(density_params)
            # 4. Elastic properties
            if results:
                elastic_params = self._calculate_elastic_properties(results)
                results.update(elastic_params)
            # 5. Combined rock properties index
            if results:
                results['rock_properties_index'] = self._calculate_rock_index(results)
            # 6. Uncertainty estimates
            results['uncertainties'] = self._estimate_uncertainties(results)
            logger.info(f"Rock properties parameters calculated: {len(results)} metrics")
        except Exception as e:
            logger.error(f"Error calculating rock properties parameters: {e}")
            return self._get_default_parameters()
        return results
    def _analyze_velocities(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze seismic velocity data."""
        results = {}
        # Check for velocity columns
        has_vp = 'vp' in data.columns or 'p_velocity' in data.columns
        has_vs = 'vs' in data.columns or 's_velocity' in data.columns
        if not (has_vp or has_vs):
            return results
        # P-wave velocity
        if has_vp:
            vp_col = 'vp' if 'vp' in data.columns else 'p_velocity'
            vp_values = data[vp_col].dropna().values
            if len(vp_values) > 0:
                results['vp_mean'] = float(np.mean(vp_values))
                results['vp_std'] = float(np.std(vp_values))
                results['vp_min'] = float(np.min(vp_values))
                results['vp_max'] = float(np.max(vp_values))
                results['vp_cv'] = float(results['vp_std'] / results['vp_mean'] 
                                       if results['vp_mean'] != 0 else 0)
        # S-wave velocity
        if has_vs:
            vs_col = 'vs' if 'vs' in data.columns else 's_velocity'
            vs_values = data[vs_col].dropna().values
            if len(vs_values) > 0:
                results['vs_mean'] = float(np.mean(vs_values))
                results['vs_std'] = float(np.std(vs_values))
                results['vs_min'] = float(np.min(vs_values))
                results['vs_max'] = float(np.max(vs_values))
                results['vs_cv'] = float(results['vs_std'] / results['vs_mean'] 
                                       if results['vs_mean'] != 0 else 0)
        # Vp/Vs ratio
        if has_vp and has_vs:
            # Ensure arrays have same length
            min_len = min(len(vp_values), len(vs_values))
            if min_len > 0:
                vp_subset = vp_values[:min_len]
                vs_subset = vs_values[:min_len]
                # Avoid division by zero
                valid_idx = vs_subset > 0
                if np.any(valid_idx):
                    vp_vs_ratios = vp_subset[valid_idx] / vs_subset[valid_idx]
                    results['vp_vs_mean'] = float(np.mean(vp_vs_ratios))
                    results['vp_vs_std'] = float(np.std(vp_vs_ratios))
                    results['vp_vs_min'] = float(np.min(vp_vs_ratios))
                    results['vp_vs_max'] = float(np.max(vp_vs_ratios))
                    # Percent change from normal
                    normal = self.config['vp_vs_normal']
                    percent_change = ((results['vp_vs_mean'] - normal) / normal) * 100
                    results['vp_vs_percent_change'] = float(percent_change)
                    # Anomaly detection
                    threshold = self.config['vp_vs_threshold']
                    anomalies = np.abs(vp_vs_ratios - normal) > (normal * threshold)
                    results['vp_vs_anomaly_ratio'] = float(np.mean(anomalies))
        # Time variation analysis if time data available
        if 'time' in data.columns and (has_vp or has_vs):
            time_variation = self._analyze_velocity_time_variation(data)
            results.update(time_variation)
        return results
    def _analyze_attenuation(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze attenuation (Q) data."""
        results = {}
        # Find Q column
        q_col = None
        for col in ['q', 'q_value', 'attenuation', 'quality_factor']:
            if col in data.columns:
                q_col = col
                break
        if q_col is None:
            return results
        q_values = data[q_col].dropna().values
        if len(q_values) > 0:
            results['q_mean'] = float(np.mean(q_values))
            results['q_std'] = float(np.std(q_values))
            results['q_min'] = float(np.min(q_values))
            results['q_max'] = float(np.max(q_values))
            # Percent change from normal
            normal = self.config['q_normal']
            percent_change = ((results['q_mean'] - normal) / normal) * 100
            results['q_percent_change'] = float(percent_change)
            # Low Q anomalies (high attenuation)
            low_q_threshold = normal * 0.7  # 30% reduction
            low_q_anomalies = q_values < low_q_threshold
            results['low_q_ratio'] = float(np.mean(low_q_anomalies))
        return results
    def _analyze_density(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze density data."""
        results = {}
        # Find density column
        density_col = None
        for col in ['density', 'rho', 'density_gcc']:
            if col in data.columns:
                density_col = col
                break
        if density_col is None:
            return results
        density_values = data[density_col].dropna().values
        if len(density_values) > 0:
            results['density_mean'] = float(np.mean(density_values))
            results['density_std'] = float(np.std(density_values))
            results['density_min'] = float(np.min(density_values))
            results['density_max'] = float(np.max(density_values))
            # Percent change from normal
            normal = self.config['density_normal']
            percent_change = ((results['density_mean'] - normal) / normal) * 100
            results['density_percent_change'] = float(percent_change)
            # Low density anomalies (fracturing/fluid)
            low_density_threshold = normal * 0.9  # 10% reduction
            low_density_anomalies = density_values < low_density_threshold
            results['low_density_ratio'] = float(np.mean(low_density_anomalies))
        return results
    def _analyze_velocity_time_variation(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze time variation of velocities."""
        results = {}
        if 'time' not in data.columns:
            return results
        # Check for velocity columns over time
        velocity_columns = []
        for col in ['vp', 'p_velocity', 'vs', 's_velocity']:
            if col in data.columns:
                velocity_columns.append(col)
        if not velocity_columns:
            return results
        try:
            times = pd.to_datetime(data['time']).values
            for vcol in velocity_columns:
                values = data[vcol].dropna().values
                time_idx = data[vcol].dropna().index
                if len(values) > 2 and len(time_idx) > 2:
                    times_valid = times[time_idx]
                    # Convert to seconds for regression
                    times_sec = times_valid.astype('datetime64[s]').astype(float)
                    # Linear trend
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        times_sec, values
                    )
                    results[f'{vcol}_trend_slope'] = float(slope)
                    results[f'{vcol}_trend_pvalue'] = float(p_value)
                    # Percent change per year
                    if intercept != 0:
                        percent_change_per_year = (slope / abs(intercept)) * 365.25 * 86400 * 100
                        results[f'{vcol}_trend_percent_per_year'] = float(
                            percent_change_per_year
                        )
                    # Detrended variability
                    detrended = values - (slope * times_sec + intercept)
                    results[f'{vcol}_detrended_std'] = float(np.std(detrended))
        except Exception as e:
            logger.warning(f"Error in velocity time variation analysis: {e}")
        return results
    def _calculate_elastic_properties(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Calculate elastic properties from velocities and density."""
        results = {}
        try:
            # Check if we have required parameters
            has_vp = 'vp_mean' in parameters
            has_vs = 'vs_mean' in parameters
            has_density = 'density_mean' in parameters
            if not (has_vp and has_vs):
                return results
            vp = parameters['vp_mean'] * 1000  # Convert to m/s
            vs = parameters['vs_mean'] * 1000  # Convert to m/s
            # Use average crustal density if not available
            if has_density:
                density = parameters['density_mean'] * 1000  # Convert to kg/m³
            else:
                density = self.config['density_normal'] * 1000
            # Shear modulus (μ = ρ * Vs²)
            shear_modulus = density * vs**2
            results['shear_modulus_gpa'] = float(shear_modulus / 1e9)
            # Bulk modulus (K = ρ * (Vp² - 4/3 * Vs²))
            bulk_modulus = density * (vp**2 - (4/3) * vs**2)
            results['bulk_modulus_gpa'] = float(bulk_modulus / 1e9)
            # Young's modulus (E = 9Kμ / (3K + μ))
            if bulk_modulus > 0 and shear_modulus > 0:
                youngs_modulus = (9 * bulk_modulus * shear_modulus) / \
                                (3 * bulk_modulus + shear_modulus)
                results['youngs_modulus_gpa'] = float(youngs_modulus / 1e9)
            # Poisson's ratio (ν = (Vp² - 2Vs²) / (2(Vp² - Vs²)))
            if vp**2 != vs**2:
                poissons_ratio = (vp**2 - 2*vs**2) / (2 * (vp**2 - vs**2))
                results['poissons_ratio'] = float(poissons_ratio)
                # Percent change from normal
                normal_poisson = self.config['poissons_normal']
                percent_change = ((poissons_ratio - normal_poisson) / normal_poisson) * 100
                results['poisson_percent_change'] = float(percent_change)
            # Lame parameters
            results['lame_first_gpa'] = float(bulk_modulus / 1e9)
            results['lame_second_gpa'] = float(shear_modulus / 1e9)
        except Exception as e:
            logger.warning(f"Error calculating elastic properties: {e}")
        return results
    def _calculate_rock_index(self, parameters: Dict[str, float]) -> float:
        """Calculate combined rock properties index."""
        weights = {
            'vp_vs_percent_change': 0.25,
            'vp_vs_anomaly_ratio': 0.20,
            'q_percent_change': 0.15,
            'low_density_ratio': 0.15,
            'vp_trend_percent_per_year': 0.10,
            'poisson_percent_change': 0.05,
            'low_q_ratio': 0.05,
            'vp_detrended_std': 0.05,
        }
        index = 0.5
        for param, weight in weights.items():
            if param in parameters:
                value = parameters[param]
                # Normalize parameter
                if param.endswith('_percent_change'):
                    normalized = min(1.0, abs(value) / 20.0)  # 20% change = 1.0
                elif param.endswith('_ratio'):
                    normalized = min(1.0, value * 5)  # 20% ratio = 1.0
                elif param == 'vp_trend_percent_per_year':
                    normalized = min(1.0, abs(value) / 10.0)  # 10%/year = 1.0
                elif param == 'vp_detrended_std':
                    normalized = min(1.0, value / 0.1)  # 0.1 km/s std = 1.0
                else:
                    normalized = 0.5
                index += weight * (normalized - 0.5)
        index = max(0.0, min(1.0, index))
        return float(index)
    def _estimate_uncertainties(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Estimate uncertainties for rock properties."""
        uncertainties = {}
        for param, value in parameters.items():
            if param.endswith('_std'):
                uncertainties[param.replace('_std', '')] = value
            elif param == 'rock_properties_index':
                uncertainties[param] = 0.2
            elif param.endswith('_cv'):  # Coefficient of variation
                base_param = param.replace('_cv', '')
                if base_param in parameters:
                    uncertainties[base_param] = parameters[base_param] * value
            elif param.endswith('_ratio'):
                # Binomial uncertainty
                n = 100
                p = value
                if 0 < p < 1:
                    uncertainty = np.sqrt(p * (1 - p) / n)
                    uncertainties[param] = float(uncertainty)
                else:
                    uncertainties[param] = 0.1
            elif param.endswith('_percent_change'):
                # Percent change: use absolute value for uncertainty
                if value != 0:
                    uncertainties[param] = abs(value) * 0.15  # 15% of change
                else:
                    uncertainties[param] = 5.0  # 5% absolute
            else:
                if value != 0:
                    uncertainties[param] = abs(value) * 0.1
                else:
                    uncertainties[param] = 0.1
        return uncertainties
    def _get_default_parameters(self) -> Dict[str, float]:
        return {
            'vp_mean': 6.0,
            'vs_mean': 3.46,
            'vp_vs_mean': 1.73,
            'q_mean': 100.0,
            'density_mean': 2.7,
            'poissons_ratio': 0.25,
            'rock_properties_index': 0.5,
            'uncertainties': {
                'rock_properties_index': 0.2,
                'vp_mean': 0.1,
                'vp_vs_mean': 0.05,
                'q_mean': 10.0,
            }
        }
# Convenience functions
def calculate_vp_vs_ratio(velocity_data: pd.DataFrame) -> float:
    analyzer = RockPropertiesAnalyzer()
    params = analyzer.calculate_parameters(velocity_data)
    return params.get('vp_vs_mean', 1.73)
def calculate_rock_properties_index(velocity_data: pd.DataFrame) -> float:
    analyzer = RockPropertiesAnalyzer()
    params = analyzer.calculate_parameters(velocity_data)
    return params.get('rock_properties_index', 0.5)