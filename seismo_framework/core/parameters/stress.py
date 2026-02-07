"""
Tectonic Stress State (T) parameter module.
This module handles stress analysis including:
- Coulomb stress calculations
- Focal mechanism solutions
- Stress tensor inversion
- Stress transfer modeling
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import warnings
logger = logging.getLogger(__name__)
class StressAnalyzer:
    """
    Analyzer for tectonic stress state parameter (T).
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._setup_defaults()
    def _setup_defaults(self):
        defaults = {
            'friction_coefficient': 0.6,
            'skempton_coefficient': 0.5,
            'shear_modulus_gpa': 30.0,
            'poissons_ratio': 0.25,
            'critical_coulomb_stress': 0.1,  # MPa
        }
        defaults.update(self.config)
        self.config = defaults
    def calculate_parameters(self, focal_mechanisms: pd.DataFrame,
                            seismicity_data: Optional[pd.DataFrame] = None,
                            fault_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Calculate stress parameters.
        Parameters
        ----------
        focal_mechanisms : pd.DataFrame
            Focal mechanism solutions
        seismicity_data : pd.DataFrame, optional
            Seismicity catalog for stress transfer
        fault_data : pd.DataFrame, optional
            Fault geometry data
        Returns
        -------
        dict
            Dictionary of calculated parameters
        """
        results = {}
        try:
            # 1. Stress tensor analysis from focal mechanisms
            if not focal_mechanisms.empty:
                tensor_params = self._analyze_stress_tensor(focal_mechanisms)
                results.update(tensor_params)
            # 2. Coulomb stress analysis
            if (seismicity_data is not None and not seismicity_data.empty and
                fault_data is not None and not fault_data.empty):
                coulomb_params = self._analyze_coulomb_stress(seismicity_data, fault_data)
                results.update(coulomb_params)
            # 3. Stress ratio and regime
            if results:
                regime_params = self._determine_stress_regime(results)
                results.update(regime_params)
            # 4. Combined stress index
            if results:
                results['stress_index'] = self._calculate_stress_index(results)
            # 5. Uncertainty estimates
            results['uncertainties'] = self._estimate_uncertainties(results)
            logger.info(f"Stress parameters calculated: {len(results)} metrics")
        except Exception as e:
            logger.error(f"Error calculating stress parameters: {e}")
            return self._get_default_parameters()
        return results
    def _analyze_stress_tensor(self, focal_mechanisms: pd.DataFrame) -> Dict[str, float]:
        """Analyze stress tensor from focal mechanisms."""
        results = {}
        # Check for required columns
        required_cols = ['strike', 'dip', 'rake']
        available_cols = [col for col in required_cols if col in focal_mechanisms.columns]
        if len(available_cols) < 3:
            return results
        strikes = focal_mechanisms['strike'].dropna().values
        dips = focal_mechanisms['dip'].dropna().values
        rakes = focal_mechanisms['rake'].dropna().values
        # Ensure arrays have same length
        min_len = min(len(strikes), len(dips), len(rakes))
        if min_len < 3:
            return results
        strikes = strikes[:min_len]
        dips = dips[:min_len]
        rakes = rakes[:min_len]
        # Convert to principal stress orientations
        try:
            # Calculate P and T axes for each mechanism
            p_axes = []
            t_axes = []
            for strike, dip, rake in zip(strikes, dips, rakes):
                p_axis, t_axis = self._calculate_p_t_axes(strike, dip, rake)
                p_axes.append(p_axis)
                t_axes.append(t_axis)
            p_axes = np.array(p_axes)
            t_axes = np.array(t_axes)
            # Calculate mean orientations
            if len(p_axes) > 0:
                mean_p_axis = self._spherical_mean(p_axes)
                mean_t_axis = self._spherical_mean(t_axes)
                results['p_axis_trend'] = float(mean_p_axis[0])
                results['p_axis_plunge'] = float(mean_p_axis[1])
                results['t_axis_trend'] = float(mean_t_axis[0])
                results['t_axis_plunge'] = float(mean_t_axis[1])
            # Calculate stress ratio R = (σ₂ - σ₁) / (σ₃ - σ₁)
            # Using simplified method from focal mechanism inversion
            if len(p_axes) >= 5:
                try:
                    stress_ratio = self._estimate_stress_ratio(p_axes, t_axes)
                    results['stress_ratio_r'] = float(stress_ratio)
                except:
                    pass
            # Calculate orientation consistency
            if len(p_axes) > 2:
                p_consistency = self._calculate_orientation_consistency(p_axes)
                t_consistency = self._calculate_orientation_consistency(t_axes)
                results['p_axis_consistency'] = float(p_consistency)
                results['t_axis_consistency'] = float(t_consistency)
        except Exception as e:
            logger.warning(f"Error in stress tensor analysis: {e}")
        return results
    def _calculate_p_t_axes(self, strike: float, dip: float, rake: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate P and T axes from focal mechanism."""
        # Convert to radians
        strike_rad = np.radians(strike)
        dip_rad = np.radians(dip)
        rake_rad = np.radians(rake)
        # Calculate P axis orientation
        p_trend = strike_rad + np.arctan2(
            np.cos(rake_rad),
            np.sin(rake_rad) * np.cos(dip_rad)
        )
        p_plunge = np.arcsin(np.sin(rake_rad) * np.sin(dip_rad))
        # Calculate T axis orientation (perpendicular to P in focal sphere)
        t_trend = p_trend + np.pi/2
        t_plunge = -p_plunge
        # Convert back to degrees and ensure 0-360 range
        p_trend = np.degrees(p_trend) % 360
        p_plunge = np.degrees(p_plunge)
        t_trend = np.degrees(t_trend) % 360
        t_plunge = np.degrees(t_plunge)
        return np.array([p_trend, p_plunge]), np.array([t_trend, t_plunge])
    def _spherical_mean(self, vectors: np.ndarray) -> np.ndarray:
        """Calculate mean orientation on sphere."""
        # Convert to Cartesian coordinates
        trends = np.radians(vectors[:, 0])
        plunges = np.radians(vectors[:, 1])
        x = np.cos(plunges) * np.cos(trends)
        y = np.cos(plunges) * np.sin(trends)
        z = np.sin(plunges)
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        mean_z = np.mean(z)
        # Convert back to spherical
        mean_trend = np.arctan2(mean_y, mean_x)
        mean_plunge = np.arcsin(mean_z / np.sqrt(mean_x**2 + mean_y**2 + mean_z**2))
        return np.array([np.degrees(mean_trend) % 360, np.degrees(mean_plunge)])
    def _estimate_stress_ratio(self, p_axes: np.ndarray, t_axes: np.ndarray) -> float:
        """Estimate stress ratio R from focal mechanisms."""
        # Simplified method: Use variance of P and T axes
        p_var = np.var(p_axes[:, 1])  # Variance in plunge
        t_var = np.var(t_axes[:, 1])
        # Stress ratio R ranges from 0 (σ₂ = σ₁) to 1 (σ₂ = σ₃)
        # More variance in P axes suggests intermediate stress closer to σ₁
        total_var = p_var + t_var
        if total_var > 0:
            r_value = p_var / total_var
        else:
            r_value = 0.5
        return float(r_value)
    def _calculate_orientation_consistency(self, vectors: np.ndarray) -> float:
        """Calculate consistency of orientations (0-1)."""
        if len(vectors) < 2:
            return 0.0
        # Convert to Cartesian and calculate resultant length
        trends = np.radians(vectors[:, 0])
        plunges = np.radians(vectors[:, 1])
        x = np.cos(plunges) * np.cos(trends)
        y = np.cos(plunges) * np.sin(trends)
        z = np.sin(plunges)
        resultant = np.sqrt(np.sum(x)**2 + np.sum(y)**2 + np.sum(z)**2)
        consistency = resultant / len(vectors)
        return float(consistency)
    def _analyze_coulomb_stress(self, seismicity_data: pd.DataFrame,
                                fault_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze Coulomb stress changes."""
        results = {}
        # Simplified Coulomb stress calculation
        try:
            # Extract fault parameters
            if not all(col in fault_data.columns for col in ['strike', 'dip', 'rake']):
                return results
            fault_strike = fault_data['strike'].iloc[0]
            fault_dip = fault_data['dip'].iloc[0]
            fault_rake = fault_data['rake'].iloc[0]
            # Calculate stress changes from seismicity
            if len(seismicity_data) > 0:
                stress_changes = []
                for _, event in seismicity_data.iterrows():
                    if all(col in event for col in ['magnitude', 'depth', 'distance']):
                        # Simplified stress change calculation
                        stress_change = self._calculate_simplified_coulomb(
                            event['magnitude'],
                            event['depth'],
                            event['distance'],
                            fault_strike, fault_dip, fault_rake
                        )
                        stress_changes.append(stress_change)
                if stress_changes:
                    stress_array = np.array(stress_changes)
                    results['coulomb_stress_mean'] = float(np.mean(stress_array))
                    results['coulomb_stress_std'] = float(np.std(stress_array))
                    results['coulomb_stress_max'] = float(np.max(np.abs(stress_array)))
                    # Count of positive stress changes (loading)
                    positive_stress = stress_array > 0
                    results['positive_coulomb_ratio'] = float(np.mean(positive_stress))
                    # Count above critical threshold
                    critical = self.config['critical_coulomb_stress']
                    above_critical = np.abs(stress_array) > critical
                    results['above_critical_ratio'] = float(np.mean(above_critical))
        except Exception as e:
            logger.warning(f"Error in Coulomb stress analysis: {e}")
        return results
    def _calculate_simplified_coulomb(self, magnitude: float, depth: float,
                                     distance: float, strike: float, 
                                     dip: float, rake: float) -> float:
        """Calculate simplified Coulomb stress change."""
        # Simplified formula: ΔCFS = Δτ + μ'Δσₙ
        # where Δτ is shear stress change, Δσₙ is normal stress change
        # Stress drop from magnitude (simplified)
        stress_drop = 10**(1.5 * magnitude - 9.0)  # in MPa
        # Geometric attenuation with distance
        attenuation = 1.0 / (1.0 + (distance / 10.0)**2)
        # Depth correction
        depth_factor = np.exp(-depth / 15.0)
        # Fault orientation factor (simplified)
        orientation_factor = np.abs(np.sin(np.radians(strike)) * 
                                   np.cos(np.radians(dip)) * 
                                   np.sin(np.radians(rake)))
        # Coulomb stress change
        delta_cfs = stress_drop * attenuation * depth_factor * orientation_factor
        # Apply Skempton's coefficient for pore pressure
        b = self.config['skempton_coefficient']
        delta_cfs_effective = delta_cfs * (1 - b)
        return float(delta_cfs_effective)
    def _determine_stress_regime(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Determine stress regime from parameters."""
        results = {}
        try:
            # Use plunge of P and T axes to determine regime
            p_plunge = parameters.get('p_axis_plunge', 45.0)
            t_plunge = parameters.get('t_axis_plunge', 45.0)
            # Stress regime classification
            if p_plunge < 30 and t_plunge > 60:
                regime = 'normal'
            elif p_plunge > 60 and t_plunge < 30:
                regime = 'reverse'
            elif 30 <= p_plunge <= 60 and 30 <= t_plunge <= 60:
                regime = 'strike_slip'
            else:
                regime = 'oblique'
            results['stress_regime'] = regime
            # Regime confidence based on consistency
            p_consistency = parameters.get('p_axis_consistency', 0.5)
            t_consistency = parameters.get('t_axis_consistency', 0.5)
            results['regime_confidence'] = float((p_consistency + t_consistency) / 2)
        except:
            results['stress_regime'] = 'unknown'
            results['regime_confidence'] = 0.0
        return results
    def _calculate_stress_index(self, parameters: Dict[str, float]) -> float:
        """Calculate combined stress index."""
        weights = {
            'positive_coulomb_ratio': 0.30,
            'above_critical_ratio': 0.25,
            'coulomb_stress_max': 0.15,
            'stress_ratio_r': 0.10,
            'regime_confidence': 0.10,
            'p_axis_consistency': 0.05,
            't_axis_consistency': 0.05,
        }
        index = 0.5
        for param, weight in weights.items():
            if param in parameters:
                value = parameters[param]
                # Normalize parameter
                if param in ['positive_coulomb_ratio', 'above_critical_ratio']:
                    normalized = min(1.0, value * 2)  # 50% = 1.0
                elif param == 'coulomb_stress_max':
                    normalized = min(1.0, value / 0.5)  # 0.5 MPa = 1.0
                elif param == 'stress_ratio_r':
                    # Intermediate R values (0.3-0.7) indicate more complex stress
                    normalized = 2.0 * abs(value - 0.5)  # 0 or 1 = 1.0, 0.5 = 0.0
                elif param in ['regime_confidence', 'p_axis_consistency', 't_axis_consistency']:
                    normalized = value  # Already 0-1
                else:
                    normalized = 0.5
                index += weight * (normalized - 0.5)
        # Adjust based on stress regime
        regime = parameters.get('stress_regime', 'unknown')
        if regime in ['reverse', 'thrust']:
            index += 0.1  # Reverse faulting often associated with higher stress
        elif regime == 'normal':
            index -= 0.05  # Normal faulting often associated with lower stress
        index = max(0.0, min(1.0, index))
        return float(index)
    def _estimate_uncertainties(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Estimate uncertainties for stress parameters."""
        uncertainties = {}
        for param, value in parameters.items():
            if param.endswith('_std'):
                uncertainties[param.replace('_std', '')] = value
            elif param == 'stress_index':
                uncertainties[param] = 0.25  # Higher uncertainty for complex calculations
            elif param.endswith('_ratio'):
                # Binomial uncertainty
                n = 50  # Assumed sample size
                p = value
                if 0 < p < 1:
                    uncertainty = np.sqrt(p * (1 - p) / n)
                    uncertainties[param] = float(uncertainty)
                else:
                    uncertainties[param] = 0.1
            elif param.endswith('_consistency') or param == 'regime_confidence':
                uncertainties[param] = 0.15  # Fixed for orientation measures
            else:
                if value != 0:
                    uncertainties[param] = abs(value) * 0.2
                else:
                    uncertainties[param] = 0.1
        return uncertainties
    def _get_default_parameters(self) -> Dict[str, float]:
        return {
            'p_axis_trend': 0.0,
            'p_axis_plunge': 45.0,
            't_axis_trend': 90.0,
            't_axis_plunge': 45.0,
            'stress_ratio_r': 0.5,
            'stress_regime': 'unknown',
            'stress_index': 0.5,
            'uncertainties': {
                'stress_index': 0.25,
                'p_axis_trend': 10.0,
                'stress_ratio_r': 0.2,
            }
        }
# Convenience functions
def calculate_coulomb_stress(seismicity_data: pd.DataFrame,
                            fault_data: pd.DataFrame) -> float:
    analyzer = StressAnalyzer()
    params = analyzer.calculate_parameters(pd.DataFrame(), seismicity_data, fault_data)
    return params.get('coulomb_stress_mean', 0.0)
def calculate_stress_index(focal_mechanisms: pd.DataFrame) -> float:
    analyzer = StressAnalyzer()
    params = analyzer.calculate_parameters(focal_mechanisms)
    return params.get('stress_index', 0.5)