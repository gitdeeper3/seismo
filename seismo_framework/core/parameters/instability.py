"""
Instability Parameter Analyzer.
Analyzes slope instability and landslide potential indicators.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
logger = logging.getLogger(__name__)
class InstabilityAnalyzer:
    """
    Analyzer for slope instability and landslide potential.
    This module analyzes:
    - Ground movement rates
    - Crack propagation
    - Slope geometry changes
    - Rainfall-induced instability
    - Seismic-triggered landslides
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._setup_defaults()
    def _setup_defaults(self):
        """Setup default configuration."""
        defaults = {
            'critical_displacement_rate': 10.0,  # mm/day
            'crack_growth_threshold': 5.0,  # mm/day
            'rainfall_intensity_threshold': 50.0,  # mm/hour
            'slope_angle_threshold': 30.0,  # degrees
            'seismic_trigger_threshold': 0.2,  # g
            'monitoring_period': 30,  # days
        }
        defaults.update(self.config)
        self.config = defaults
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze instability indicators.
        Parameters:
        -----------
        data : dict
            Dictionary containing instability data with keys:
            - displacement_rates: Ground movement rates (mm/day)
            - crack_measurements: Crack width/length measurements
            - slope_geometry: Slope angle, height measurements
            - rainfall_data: Rainfall intensity/duration
            - seismic_data: Ground acceleration data
            - timestamps: Measurement timestamps
        Returns:
        --------
        dict
            Analysis results including instability index
        """
        results = {
            'instability_index': 0.0,
            'uncertainty': 0.0,
            'components': {},
            'warnings': [],
            'recommendations': []
        }
        try:
            # Extract data components
            components = self._extract_components(data)
            # Calculate individual metrics
            metrics = self._calculate_metrics(components)
            # Combine metrics into instability index
            instability_index = self._combine_metrics(metrics)
            # Determine warnings
            warnings = self._generate_warnings(metrics, instability_index)
            # Generate recommendations
            recommendations = self._generate_recommendations(instability_index, warnings)
            # Update results
            results.update({
                'instability_index': float(instability_index),
                'uncertainty': self._calculate_uncertainty(metrics),
                'components': metrics,
                'warnings': warnings,
                'recommendations': recommendations,
                'metadata': {
                    'analysis_time': pd.Timestamp.now().isoformat(),
                    'parameters_used': list(components.keys()),
                    'config': self.config
                }
            })
            logger.info(f"Instability analysis complete: index={instability_index:.3f}")
        except Exception as e:
            logger.error(f"Error in instability analysis: {e}")
            results['error'] = str(e)
        return results
    def _extract_components(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate data components."""
        components = {}
        # Displacement rates
        if 'displacement_rates' in data:
            rates = np.array(data['displacement_rates'])
            if rates.size > 0:
                components['displacement'] = {
                    'rates': rates,
                    'mean_rate': np.mean(rates),
                    'max_rate': np.max(rates),
                    'trend': self._calculate_trend(rates)
                }
        # Crack measurements
        if 'crack_measurements' in data:
            cracks = np.array(data['crack_measurements'])
            if cracks.size > 0:
                components['cracks'] = {
                    'measurements': cracks,
                    'growth_rate': self._calculate_growth_rate(cracks),
                    'total_length': np.sum(cracks[:, 0]) if cracks.ndim > 1 else np.sum(cracks)
                }
        # Slope geometry
        if 'slope_geometry' in data:
            geometry = data['slope_geometry']
            if isinstance(geometry, dict):
                components['slope'] = geometry
        # Rainfall data
        if 'rainfall_data' in data:
            rainfall = np.array(data['rainfall_data'])
            if rainfall.size > 0:
                components['rainfall'] = {
                    'intensity': np.max(rainfall),
                    'duration': len(rainfall),
                    'cumulative': np.sum(rainfall)
                }
        # Seismic data
        if 'seismic_data' in data:
            seismic = np.array(data['seismic_data'])
            if seismic.size > 0:
                components['seismic'] = {
                    'max_acceleration': np.max(np.abs(seismic)),
                    'frequency_content': self._analyze_frequency(seismic)
                }
        return components
    def _calculate_metrics(self, components: Dict[str, Any]) -> Dict[str, float]:
        """Calculate instability metrics from components."""
        metrics = {}
        # Displacement metric (0-1)
        if 'displacement' in components:
            disp = components['displacement']
            rate_ratio = disp['max_rate'] / self.config['critical_displacement_rate']
            metrics['displacement_risk'] = min(1.0, rate_ratio)
        # Crack growth metric
        if 'cracks' in components:
            cracks = components['cracks']
            if 'growth_rate' in cracks:
                growth_ratio = cracks['growth_rate'] / self.config['crack_growth_threshold']
                metrics['crack_risk'] = min(1.0, growth_ratio)
        # Slope stability metric
        if 'slope' in components:
            slope = components['slope']
            if 'angle' in slope:
                angle_ratio = slope['angle'] / self.config['slope_angle_threshold']
                metrics['slope_risk'] = min(1.0, angle_ratio)
        # Rainfall-induced risk
        if 'rainfall' in components:
            rainfall = components['rainfall']
            if 'intensity' in rainfall:
                rain_ratio = rainfall['intensity'] / self.config['rainfall_intensity_threshold']
                metrics['rainfall_risk'] = min(1.0, rain_ratio)
        # Seismic-triggered risk
        if 'seismic' in components:
            seismic = components['seismic']
            if 'max_acceleration' in seismic:
                seismic_ratio = seismic['max_acceleration'] / self.config['seismic_trigger_threshold']
                metrics['seismic_risk'] = min(1.0, seismic_ratio)
        # Ensure all metrics exist
        for key in ['displacement_risk', 'crack_risk', 'slope_risk', 'rainfall_risk', 'seismic_risk']:
            if key not in metrics:
                metrics[key] = 0.0
        return metrics
    def _combine_metrics(self, metrics: Dict[str, float]) -> float:
        """Combine individual metrics into overall instability index."""
        # Weighted combination based on importance
        weights = {
            'displacement_risk': 0.30,
            'crack_risk': 0.25,
            'slope_risk': 0.20,
            'rainfall_risk': 0.15,
            'seismic_risk': 0.10
        }
        total_weight = 0.0
        weighted_sum = 0.0
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                weighted_sum += metrics[metric_name] * weight
                total_weight += weight
        if total_weight > 0:
            instability_index = weighted_sum / total_weight
        else:
            instability_index = 0.5  # Default moderate risk
        # Apply nonlinear scaling (higher risks amplify each other)
        if instability_index > 0.7:
            # Amplify high risks
            instability_index = 0.7 + (instability_index - 0.7) * 1.5
        elif instability_index < 0.3:
            # Reduce low risks
            instability_index = instability_index * 0.8
        return np.clip(instability_index, 0.0, 1.0)
    def _calculate_uncertainty(self, metrics: Dict[str, float]) -> float:
        """Calculate uncertainty in instability assessment."""
        # Uncertainty based on:
        # 1. Number of available metrics
        # 2. Consistency between metrics
        # 3. Extreme values
        n_metrics = len(metrics)
        if n_metrics == 0:
            return 0.5  # High uncertainty
        # Base uncertainty decreases with more metrics
        base_uncertainty = 0.3 / (1 + np.log1p(n_metrics))
        # Add uncertainty from metric inconsistency
        values = list(metrics.values())
        if len(values) > 1:
            std_dev = np.std(values)
            inconsistency_uncertainty = std_dev * 0.2
        else:
            inconsistency_uncertainty = 0.1
        # Add uncertainty from extreme values
        if any(v > 0.8 for v in values):
            extreme_uncertainty = 0.15
        else:
            extreme_uncertainty = 0.0
        total_uncertainty = base_uncertainty + inconsistency_uncertainty + extreme_uncertainty
        return np.clip(total_uncertainty, 0.05, 0.5)
    def _generate_warnings(self, metrics: Dict[str, float], 
                          instability_index: float) -> List[str]:
        """Generate warnings based on metrics."""
        warnings = []
        # Check individual thresholds
        if metrics.get('displacement_risk', 0) > 0.8:
            warnings.append("CRITICAL: Rapid ground displacement detected")
        elif metrics.get('displacement_risk', 0) > 0.6:
            warnings.append("WARNING: Accelerated ground movement")
        if metrics.get('crack_risk', 0) > 0.7:
            warnings.append("WARNING: Significant crack propagation")
        if metrics.get('rainfall_risk', 0) > 0.8:
            warnings.append("CRITICAL: Extreme rainfall intensity")
        # Overall instability warning
        if instability_index > 0.8:
            warnings.append("IMMEDIATE ACTION REQUIRED: High landslide probability")
        elif instability_index > 0.6:
            warnings.append("HIGH RISK: Significant instability indicators")
        elif instability_index > 0.4:
            warnings.append("MODERATE RISK: Monitor closely")
        return warnings
    def _generate_recommendations(self, instability_index: float,
                                 warnings: List[str]) -> List[str]:
        """Generate recommendations based on instability level."""
        recommendations = []
        if instability_index > 0.8:
            recommendations = [
                "Immediate evacuation of high-risk areas",
                "Close access to unstable slopes",
                "Continuous monitoring with high frequency",
                "Alert emergency services",
                "Prepare for possible landslide within hours"
            ]
        elif instability_index > 0.6:
            recommendations = [
                "Restrict access to unstable areas",
                "Increase monitoring frequency",
                "Install additional sensors",
                "Review evacuation plans",
                "Notify local authorities"
            ]
        elif instability_index > 0.4:
            recommendations = [
                "Close monitoring of critical areas",
                "Regular visual inspections",
                "Check drainage systems",
                "Update risk assessments",
                "Prepare monitoring equipment"
            ]
        else:
            recommendations = [
                "Continue routine monitoring",
                "Maintain drainage systems",
                "Document any changes",
                "Regular equipment checks",
                "Update baseline measurements"
            ]
        return recommendations
    def _calculate_trend(self, data: np.ndarray) -> float:
        """Calculate trend in time series data."""
        if len(data) < 2:
            return 0.0
        x = np.arange(len(data))
        slope, _ = np.polyfit(x, data, 1)
        # Normalize trend to -1 to 1 range
        max_val = np.max(np.abs(data))
        if max_val > 0:
            normalized_trend = slope / max_val
        else:
            normalized_trend = 0.0
        return float(normalized_trend)
    def _calculate_growth_rate(self, crack_data: np.ndarray) -> float:
        """Calculate crack growth rate."""
        if len(crack_data) < 2:
            return 0.0
        if crack_data.ndim == 1:
            # Simple 1D array
            growth = crack_data[-1] - crack_data[0]
            time_interval = len(crack_data)
        else:
            # Multi-dimensional array (width, length, depth)
            growth = crack_data[-1, 0] - crack_data[0, 0]  # Width change
            time_interval = len(crack_data)
        if time_interval > 0:
            return growth / time_interval
        else:
            return 0.0
    def _analyze_frequency(self, seismic_data: np.ndarray) -> Dict[str, float]:
        """Analyze frequency content of seismic data."""
        if len(seismic_data) < 10:
            return {'dominant_frequency': 0.0}
        # Simple frequency analysis using FFT
        fft_result = np.fft.fft(seismic_data)
        frequencies = np.fft.fftfreq(len(seismic_data))
        # Find dominant frequency
        magnitude = np.abs(fft_result)
        dominant_idx = np.argmax(magnitude[1:]) + 1  # Skip DC component
        dominant_freq = np.abs(frequencies[dominant_idx])
        return {
            'dominant_frequency': float(dominant_freq),
            'energy_ratio': float(magnitude[dominant_idx] / np.sum(magnitude))
        }
    def predict_failure_probability(self, historical_data: pd.DataFrame,
                                   time_horizon: int = 7) -> float:
        """
        Predict landslide probability over time horizon.
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical instability measurements
        time_horizon : int
            Prediction horizon in days
        Returns:
        --------
        float
            Probability of major instability within time horizon
        """
        try:
            if historical_data.empty:
                return 0.05  # Base rate
            # Extract displacement rates
            if 'displacement_rate' in historical_data.columns:
                rates = historical_data['displacement_rate'].dropna().values
                if len(rates) >= 3:
                    # Calculate acceleration
                    acceleration = np.diff(rates, n=2)
                    if len(acceleration) > 0:
                        mean_accel = np.mean(acceleration)
                        # Simple predictive model
                        if mean_accel > 0:
                            # Accelerating movement
                            base_prob = 0.3 + min(0.6, mean_accel * 10)
                        else:
                            # Decelerating or stable
                            base_prob = 0.1
                        # Adjust for time horizon
                        horizon_factor = min(1.0, time_horizon / 30)
                        probability = base_prob * horizon_factor
                        return np.clip(probability, 0.01, 0.95)
            return 0.1  # Default probability
        except Exception as e:
            logger.error(f"Error in failure prediction: {e}")
            return 0.05
# Convenience function
def analyze_instability_simple(displacement_rates: List[float],
                              rainfall_intensity: Optional[float] = None) -> float:
    """
    Simple instability analysis.
    Parameters:
    -----------
    displacement_rates : list
        Ground displacement rates (mm/day)
    rainfall_intensity : float, optional
        Rainfall intensity (mm/hour)
    Returns:
    --------
    float
        Instability index (0-1)
    """
    analyzer = InstabilityAnalyzer()
    data = {
        'displacement_rates': displacement_rates
    }
    if rainfall_intensity is not None:
        data['rainfall_data'] = [rainfall_intensity]
    result = analyzer.analyze(data)
    return result['instability_index']