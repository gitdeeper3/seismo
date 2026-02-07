"""
Deformation Parameter Analyzer.
Analyzes ground deformation patterns from GPS, InSAR, and tiltmeter data.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import math
logger = logging.getLogger(__name__)
class DeformationAnalyzer:
    """
    Analyzer for ground deformation parameters.
    This module analyzes:
    - GPS displacement time series
    - InSAR deformation maps
    - Tiltmeter measurements
    - Strain accumulation rates
    - Inflation/deflation patterns
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._setup_defaults()
    def _setup_defaults(self):
        """Setup default configuration."""
        defaults = {
            'critical_displacement_rate': 10.0,  # mm/year
            'strain_threshold': 1e-6,  # microstrain
            'inflation_threshold': 5.0,  # mm/year
            'spatial_correlation_distance': 10.0,  # km
            'temporal_window': 30,  # days
            'uncertainty_scaling': 0.1,
        }
        defaults.update(self.config)
        self.config = defaults
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze deformation data.
        Parameters:
        -----------
        data : dict
            Dictionary containing deformation data with keys:
            - gps_displacements: GPS time series data
            - insar_data: InSAR deformation maps
            - tilt_measurements: Tiltmeter data
            - station_locations: Station coordinates
            - timestamps: Measurement times
        Returns:
        --------
        dict
            Analysis results including deformation index
        """
        results = {
            'deformation_index': 0.0,
            'uncertainty': 0.0,
            'displacement_rates': {},
            'strain_rates': {},
            'inflation_volume': 0.0,
            'spatial_patterns': {},
            'temporal_trends': {},
            'warnings': [],
            'recommendations': []
        }
        try:
            # Extract data components
            components = self._extract_components(data)
            if not components:
                logger.warning("No deformation data components found")
                return self._get_default_result()
            # Calculate deformation metrics
            metrics = self._calculate_metrics(components)
            # Calculate deformation index
            deformation_index = self._calculate_deformation_index(metrics)
            results['deformation_index'] = deformation_index
            # Calculate uncertainty
            uncertainty = self._calculate_uncertainty(metrics, components)
            results['uncertainty'] = uncertainty
            # Update results with metrics
            results.update({
                'displacement_rates': metrics.get('displacement', {}),
                'strain_rates': metrics.get('strain', {}),
                'inflation_volume': metrics.get('inflation', 0.0),
                'spatial_patterns': self._analyze_spatial_patterns(components),
                'temporal_trends': self._analyze_temporal_trends(components),
                'metadata': {
                    'analysis_time': pd.Timestamp.now().isoformat(),
                    'components_analyzed': list(components.keys()),
                    'config': self.config
                }
            })
            # Generate warnings
            warnings = self._generate_warnings(metrics, deformation_index)
            results['warnings'] = warnings
            # Generate recommendations
            recommendations = self._generate_recommendations(deformation_index, warnings)
            results['recommendations'] = recommendations
            logger.info(f"Deformation analysis complete: index={deformation_index:.3f}, "
                       f"uncertainty={uncertainty:.3f}")
        except Exception as e:
            logger.error(f"Error in deformation analysis: {e}")
            results['error'] = str(e)
        return results
    def _extract_components(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate deformation data components."""
        components = {}
        # GPS displacements
        if 'gps_displacements' in data:
            gps_data = data['gps_displacements']
            if isinstance(gps_data, dict) or isinstance(gps_data, pd.DataFrame):
                components['gps'] = self._process_gps_data(gps_data)
        # InSAR data
        if 'insar_data' in data:
            insar_data = data['insar_data']
            if isinstance(insar_data, (np.ndarray, list, dict)):
                components['insar'] = self._process_insar_data(insar_data)
        # Tilt measurements
        if 'tilt_measurements' in data:
            tilt_data = data['tilt_measurements']
            if isinstance(tilt_data, (np.ndarray, list, dict)):
                components['tilt'] = self._process_tilt_data(tilt_data)
        return components
    def _process_gps_data(self, gps_data: Any) -> Dict[str, Any]:
        """Process GPS displacement data."""
        processed = {
            'displacements': [],
            'rates': [],
            'stations': [],
            'uncertainties': []
        }
        try:
            if isinstance(gps_data, pd.DataFrame):
                # Process DataFrame
                if 'easting' in gps_data.columns and 'northing' in gps_data.columns:
                    east = gps_data['easting'].values
                    north = gps_data['northing'].values
                    processed['displacements'] = np.sqrt(east**2 + north**2).tolist()
                if 'rate' in gps_data.columns:
                    processed['rates'] = gps_data['rate'].values.tolist()
                if 'station' in gps_data.columns:
                    processed['stations'] = gps_data['station'].values.tolist()
            elif isinstance(gps_data, dict):
                # Process dictionary
                for key in ['displacements', 'rates', 'stations']:
                    if key in gps_data:
                        processed[key] = gps_data[key]
        except Exception as e:
            logger.warning(f"Error processing GPS data: {e}")
        return processed
    def _process_insar_data(self, insar_data: Any) -> Dict[str, Any]:
        """Process InSAR deformation data."""
        processed = {
            'deformation_map': None,
            'max_deformation': 0.0,
            'mean_deformation': 0.0,
            'area': 0.0
        }
        try:
            if isinstance(insar_data, np.ndarray):
                # Process numpy array
                data_array = np.array(insar_data)
                processed['deformation_map'] = data_array
                processed['max_deformation'] = float(np.max(np.abs(data_array)))
                processed['mean_deformation'] = float(np.mean(np.abs(data_array)))
                processed['area'] = float(data_array.size)
            elif isinstance(insar_data, dict):
                # Process dictionary
                for key in ['deformation_map', 'max_deformation', 'mean_deformation', 'area']:
                    if key in insar_data:
                        processed[key] = insar_data[key]
        except Exception as e:
            logger.warning(f"Error processing InSAR data: {e}")
        return processed
    def _process_tilt_data(self, tilt_data: Any) -> Dict[str, Any]:
        """Process tiltmeter data."""
        processed = {
            'tilt_angles': [],
            'tilt_rates': [],
            'max_tilt': 0.0,
            'mean_tilt': 0.0
        }
        try:
            if isinstance(tilt_data, (list, np.ndarray)):
                # Process array
                tilt_array = np.array(tilt_data)
                processed['tilt_angles'] = tilt_array.tolist()
                processed['max_tilt'] = float(np.max(np.abs(tilt_array)))
                processed['mean_tilt'] = float(np.mean(np.abs(tilt_array)))
            elif isinstance(tilt_data, dict):
                # Process dictionary
                for key in ['tilt_angles', 'tilt_rates', 'max_tilt', 'mean_tilt']:
                    if key in tilt_data:
                        processed[key] = tilt_data[key]
        except Exception as e:
            logger.warning(f"Error processing tilt data: {e}")
        return processed
    def _calculate_metrics(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate deformation metrics from components."""
        metrics = {}
        # Displacement metrics
        if 'gps' in components:
            gps = components['gps']
            if gps.get('rates'):
                rates = gps['rates']
                max_rate = max(rates) if rates else 0
                mean_rate = np.mean(rates) if rates else 0
                metrics['displacement'] = {
                    'max_rate': float(max_rate),
                    'mean_rate': float(mean_rate),
                    'rate_ratio': float(max_rate / self.config['critical_displacement_rate'])
                }
        # Strain metrics (simplified)
        if 'gps' in components and 'stations' in components['gps']:
            # Simplified strain calculation
            n_stations = len(components['gps'].get('stations', []))
            if n_stations >= 3:
                # Very simplified strain estimation
                metrics['strain'] = {
                    'estimated': 1e-7 * n_stations,  # Placeholder
                    'stations': n_stations
                }
        # Inflation metrics from InSAR
        if 'insar' in components:
            insar = components['insar']
            max_def = insar.get('max_deformation', 0)
            if max_def > 0:
                inflation = max_def / self.config['inflation_threshold']
                metrics['inflation'] = float(min(inflation, 1.0))
        # Tilt metrics
        if 'tilt' in components:
            tilt = components['tilt']
            max_tilt = tilt.get('max_tilt', 0)
            if max_tilt > 0:
                tilt_ratio = max_tilt / 1.0  # 1 degree threshold
                metrics['tilt'] = float(min(tilt_ratio, 1.0))
        return metrics
    def _calculate_deformation_index(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall deformation index."""
        if not metrics:
            return 0.1  # Very low deformation
        # Collect individual risk factors
        factors = []
        weights = []
        # Displacement factor
        if 'displacement' in metrics:
            disp = metrics['displacement']
            rate_ratio = disp.get('rate_ratio', 0)
            factors.append(min(rate_ratio, 1.0))
            weights.append(0.4)
        # Inflation factor
        if 'inflation' in metrics:
            factors.append(metrics['inflation'])
            weights.append(0.3)
        # Tilt factor
        if 'tilt' in metrics:
            factors.append(metrics['tilt'])
            weights.append(0.2)
        # Strain factor
        if 'strain' in metrics:
            # Simplified strain factor
            strain_val = metrics['strain'].get('estimated', 0)
            strain_factor = min(strain_val / self.config['strain_threshold'], 1.0)
            factors.append(strain_factor)
            weights.append(0.1)
        if not factors:
            return 0.5  # Default moderate
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
            # Weighted average
            deformation_index = sum(f * w for f, w in zip(factors, normalized_weights))
        else:
            deformation_index = np.mean(factors) if factors else 0.5
        # Apply nonlinear scaling
        if deformation_index > 0.7:
            deformation_index = 0.7 + (deformation_index - 0.7) * 1.3
        elif deformation_index < 0.3:
            deformation_index = deformation_index * 0.8
        return np.clip(deformation_index, 0.0, 1.0)
    def _calculate_uncertainty(self, metrics: Dict[str, Any], 
                              components: Dict[str, Any]) -> float:
        """Calculate uncertainty in deformation assessment."""
        # Base uncertainty
        base_uncertainty = 0.2
        # Reduce uncertainty with more data components
        n_components = len(components)
        if n_components >= 3:
            base_uncertainty *= 0.5
        elif n_components >= 2:
            base_uncertainty *= 0.7
        # Add uncertainty based on data quality
        data_quality_uncertainty = 0.0
        for name, component in components.items():
            # Check if component has data
            if name == 'gps' and not component.get('rates'):
                data_quality_uncertainty += 0.1
            elif name == 'insar' and not component.get('deformation_map'):
                data_quality_uncertainty += 0.1
            elif name == 'tilt' and not component.get('tilt_angles'):
                data_quality_uncertainty += 0.1
        total_uncertainty = base_uncertainty + data_quality_uncertainty
        return np.clip(total_uncertainty, 0.05, 0.5)
    def _analyze_spatial_patterns(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze spatial patterns of deformation."""
        patterns = {
            'has_spatial_data': False,
            'pattern_type': 'unknown',
            'spatial_extent': 0.0,
            'correlation': 0.0
        }
        # Check for spatial data
        if 'gps' in components and components['gps'].get('stations'):
            patterns['has_spatial_data'] = True
            patterns['pattern_type'] = 'point_measurements'
            patterns['n_stations'] = len(components['gps']['stations'])
        if 'insar' in components and components['insar'].get('area'):
            patterns['has_spatial_data'] = True
            patterns['pattern_type'] = 'areal_coverage'
            patterns['spatial_extent'] = components['insar']['area']
        return patterns
    def _analyze_temporal_trends(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal trends in deformation."""
        trends = {
            'has_temporal_data': False,
            'trend_direction': 'stable',
            'acceleration': 0.0,
            'consistency': 0.0
        }
        # Check GPS time series
        if 'gps' in components and components['gps'].get('rates'):
            rates = components['gps']['rates']
            if len(rates) >= 3:
                trends['has_temporal_data'] = True
                # Simple trend detection
                x = np.arange(len(rates))
                if len(set(rates)) > 1:
                    # Simple linear trend
                    coeffs = np.polyfit(x, rates, 1)
                    slope = coeffs[0]
                    if slope > 0.1:
                        trends['trend_direction'] = 'increasing'
                    elif slope < -0.1:
                        trends['trend_direction'] = 'decreasing'
                    trends['acceleration'] = float(abs(slope))
                    # Consistency (inverse of variance)
                    variance = np.var(rates)
                    if variance > 0:
                        trends['consistency'] = float(1 / (1 + variance))
        return trends
    def _generate_warnings(self, metrics: Dict[str, Any], 
                          deformation_index: float) -> List[str]:
        """Generate warnings based on deformation analysis."""
        warnings = []
        # Check displacement rates
        if 'displacement' in metrics:
            disp = metrics['displacement']
            rate_ratio = disp.get('rate_ratio', 0)
            if rate_ratio > 1.0:
                warnings.append("CRITICAL: Displacement rate exceeds critical threshold")
            elif rate_ratio > 0.7:
                warnings.append("WARNING: High displacement rate detected")
            elif rate_ratio > 0.5:
                warnings.append("ALERT: Elevated displacement rate")
        # Check inflation
        if 'inflation' in metrics and metrics['inflation'] > 0.8:
            warnings.append("WARNING: Significant inflation detected")
        # Check overall deformation index
        if deformation_index > 0.8:
            warnings.append("CRITICAL: High deformation risk detected")
        elif deformation_index > 0.6:
            warnings.append("WARNING: Elevated deformation risk")
        elif deformation_index > 0.4:
            warnings.append("ALERT: Moderate deformation activity")
        return warnings
    def _generate_recommendations(self, deformation_index: float,
                                 warnings: List[str]) -> List[str]:
        """Generate recommendations based on deformation index."""
        recommendations = []
        if deformation_index > 0.8 or any("CRITICAL" in w for w in warnings):
            recommendations = [
                "Immediate deployment of field monitoring team",
                "Increase GPS measurement frequency to hourly",
                "Schedule urgent InSAR acquisition",
                "Alert civil protection authorities",
                "Prepare for potential volcanic unrest"
            ]
        elif deformation_index > 0.6 or any("WARNING" in w for w in warnings):
            recommendations = [
                "Increase monitoring frequency",
                "Deploy additional GPS stations",
                "Schedule InSAR acquisition",
                "Review historical deformation patterns",
                "Update risk assessments"
            ]
        elif deformation_index > 0.4 or any("ALERT" in w for w in warnings):
            recommendations = [
                "Close monitoring of deformation signals",
                "Check equipment calibration",
                "Review data quality",
                "Prepare for increased monitoring",
                "Update baseline measurements"
            ]
        else:
            recommendations = [
                "Continue routine monitoring",
                "Regular equipment maintenance",
                "Data quality assessment",
                "Historical data review",
                "Baseline updates as needed"
            ]
        return recommendations
    def _get_default_result(self) -> Dict[str, Any]:
        """Return default deformation analysis result."""
        return {
            'deformation_index': 0.1,
            'uncertainty': 0.3,
            'displacement_rates': {},
            'strain_rates': {},
            'inflation_volume': 0.0,
            'spatial_patterns': {'has_spatial_data': False},
            'temporal_trends': {'has_temporal_data': False},
            'warnings': [],
            'recommendations': [
                "No deformation data available",
                "Check monitoring equipment",
                "Verify data transmission"
            ],
            'metadata': {
                'analysis_time': pd.Timestamp.now().isoformat(),
                'components_analyzed': [],
                'note': 'Default result - no data available'
            }
        }
    def estimate_volume_change(self, deformation_data: np.ndarray,
                              pixel_size: float = 30.0) -> float:
        """
        Estimate volume change from deformation data.
        Parameters:
        -----------
        deformation_data : np.ndarray
            Deformation map (meters)
        pixel_size : float
            Pixel size in meters
        Returns:
        --------
        float
            Estimated volume change in cubic meters
        """
        try:
            if deformation_data.size == 0:
                return 0.0
            # Simple volume estimation: sum of deformation * pixel area
            pixel_area = pixel_size ** 2
            volume = np.sum(deformation_data) * pixel_area
            return float(volume)
        except Exception as e:
            logger.error(f"Error estimating volume change: {e}")
            return 0.0
# Simple utility function
def analyze_deformation_simple(displacement_rates: List[float],
                              tilt_angles: Optional[List[float]] = None) -> float:
    """
    Simple deformation analysis.
    Parameters:
    -----------
    displacement_rates : list
        GPS displacement rates (mm/year)
    tilt_angles : list, optional
        Tilt angles (degrees)
    Returns:
    --------
    float
        Deformation index (0-1)
    """
    analyzer = DeformationAnalyzer()
    data = {
        'gps_displacements': {
            'rates': displacement_rates,
            'stations': [f'station_{i}' for i in range(len(displacement_rates))]
        }
    }
    if tilt_angles:
        data['tilt_measurements'] = {'tilt_angles': tilt_angles}
    result = analyzer.analyze(data)
    return result['deformation_index']