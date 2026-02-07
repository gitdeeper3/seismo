"""
Seismic Parameter Analyzer.
Analyzes seismic activity including earthquake detection,
magnitude-frequency distribution, and temporal patterns.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import math
from datetime import datetime
logger = logging.getLogger(__name__)
class SeismicAnalyzer:
    """
    Analyzer for seismic activity parameters.
    This module analyzes:
    - Earthquake frequency-magnitude distribution (b-value)
    - Temporal clustering and foreshock/aftershock sequences
    - Depth distribution of hypocenters
    - Seismic energy release rates
    - Spatial distribution patterns
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._setup_defaults()
    def _setup_defaults(self):
        """Setup default configuration."""
        defaults = {
            'min_magnitude': 2.0,
            'max_magnitude': 8.0,
            'b_value_default': 1.0,
            'mc_completeness': 2.5,
            'time_window_days': 30,
            'spatial_window_km': 50,
            'alert_threshold_probability': 0.7,
        }
        defaults.update(self.config)
        self.config = defaults
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze seismic data.
        Parameters:
        -----------
        data : dict
            Dictionary containing seismic data with keys:
            - events: List of earthquake events (dicts with time, lat, lon, depth, mag)
            - waveforms: Optional waveform data
            - station_info: Optional station metadata
            - sampling_rate: Optional sampling rate for waveforms
        Returns:
        --------
        dict
            Analysis results including seismic index
        """
        results = {
            'seismic_index': 0.0,
            'b_value': 1.0,
            'uncertainty': 0.0,
            'event_count': 0,
            'max_magnitude': 0.0,
            'depth_distribution': {},
            'temporal_patterns': {},
            'spatial_clusters': {},
            'warnings': [],
            'recommendations': []
        }
        try:
            # Extract and validate events
            events = self._extract_events(data)
            if len(events) == 0:
                logger.warning("No seismic events found in data")
                return self._get_default_result()
            # Update basic statistics
            results['event_count'] = len(events)
            magnitudes = [e['magnitude'] for e in events if 'magnitude' in e]
            if magnitudes:
                results['max_magnitude'] = max(magnitudes)
            # Calculate b-value using simplified method
            b_value, b_uncertainty = self._calculate_b_value_simple(events)
            results['b_value'] = b_value
            results['b_value_uncertainty'] = b_uncertainty
            # Calculate seismic index
            seismic_index = self._calculate_seismic_index(events, b_value)
            results['seismic_index'] = seismic_index
            # Calculate uncertainty
            uncertainty = self._calculate_uncertainty(events, b_value, b_uncertainty)
            results['uncertainty'] = uncertainty
            # Analyze depth distribution
            depth_stats = self._analyze_depth_distribution(events)
            results['depth_distribution'] = depth_stats
            # Analyze temporal patterns
            temporal_stats = self._analyze_temporal_patterns(events)
            results['temporal_patterns'] = temporal_stats
            # Analyze spatial clusters
            spatial_stats = self._analyze_spatial_clusters(events)
            results['spatial_clusters'] = spatial_stats
            # Generate warnings
            warnings = self._generate_warnings(events, seismic_index, b_value)
            results['warnings'] = warnings
            # Generate recommendations
            recommendations = self._generate_recommendations(seismic_index, warnings)
            results['recommendations'] = recommendations
            # Add metadata
            results['metadata'] = {
                'analysis_time': datetime.now().isoformat(),
                'events_analyzed': len(events),
                'time_range': self._get_time_range(events),
                'magnitude_range': self._get_magnitude_range(events),
                'config': self.config
            }
            logger.info(f"Seismic analysis complete: index={seismic_index:.3f}, "
                       f"b-value={b_value:.2f}±{b_uncertainty:.2f}, "
                       f"events={len(events)}")
        except Exception as e:
            logger.error(f"Error in seismic analysis: {e}")
            results['error'] = str(e)
            results['seismic_index'] = 0.5  # Default moderate value
        return results
    def _extract_events(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and validate earthquake events."""
        events = []
        if 'events' in data:
            event_list = data['events']
            if isinstance(event_list, list):
                for event in event_list:
                    if isinstance(event, dict):
                        # Validate required fields
                        valid_event = {}
                        # Check for magnitude
                        if 'magnitude' in event:
                            mag = float(event['magnitude'])
                            if self.config['min_magnitude'] <= mag <= self.config['max_magnitude']:
                                valid_event['magnitude'] = mag
                        # Check for time
                        if 'time' in event:
                            valid_event['time'] = event['time']
                        # Check for location
                        if all(k in event for k in ['latitude', 'longitude']):
                            valid_event['latitude'] = float(event['latitude'])
                            valid_event['longitude'] = float(event['longitude'])
                        # Check for depth
                        if 'depth' in event:
                            valid_event['depth'] = float(event['depth'])
                        if valid_event:  # Only add if we have at least some data
                            events.append(valid_event)
        return events
    def _calculate_b_value_simple(self, events: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculate b-value using simple maximum likelihood method."""
        magnitudes = [e['magnitude'] for e in events if 'magnitude' in e]
        if len(magnitudes) < 5:
            return self.config['b_value_default'], 0.2  # Default with high uncertainty
        # Remove magnitudes below completeness threshold
        mc = self.config['mc_completeness']
        complete_mags = [m for m in magnitudes if m >= mc]
        if len(complete_mags) < 3:
            return self.config['b_value_default'], 0.3
        # Simple b-value calculation: b = log10(e) / (mean_magnitude - mc)
        mean_mag = np.mean(complete_mags)
        b_value = math.log10(math.e) / (mean_mag - mc + 0.05)  # Add small offset
        # Simple uncertainty estimation
        std_mag = np.std(complete_mags)
        uncertainty = std_mag / math.sqrt(len(complete_mags))
        # Constrain to reasonable range
        b_value = max(0.5, min(2.0, b_value))
        uncertainty = min(0.5, uncertainty)
        return float(b_value), float(uncertainty)
    def _calculate_seismic_index(self, events: List[Dict[str, Any]], b_value: float) -> float:
        """Calculate seismic activity index."""
        if len(events) == 0:
            return 0.1  # Very low activity
        # Factors for index calculation
        factors = []
        # 1. Event count factor
        n_events = len(events)
        if n_events > 100:
            count_factor = 0.9
        elif n_events > 50:
            count_factor = 0.7
        elif n_events > 20:
            count_factor = 0.5
        elif n_events > 10:
            count_factor = 0.3
        else:
            count_factor = 0.1
        factors.append(count_factor)
        # 2. Maximum magnitude factor
        magnitudes = [e['magnitude'] for e in events if 'magnitude' in e]
        if magnitudes:
            max_mag = max(magnitudes)
            if max_mag > 6.0:
                mag_factor = 0.9
            elif max_mag > 5.0:
                mag_factor = 0.7
            elif max_mag > 4.0:
                mag_factor = 0.5
            elif max_mag > 3.0:
                mag_factor = 0.3
            else:
                mag_factor = 0.1
            factors.append(mag_factor)
        # 3. b-value factor (low b-value = higher risk)
        if b_value < 0.8:
            b_factor = 0.8  # High risk
        elif b_value < 1.0:
            b_factor = 0.6  # Moderate risk
        elif b_value < 1.2:
            b_factor = 0.4  # Low risk
        else:
            b_factor = 0.2  # Very low risk
        factors.append(b_factor)
        # 4. Temporal clustering factor
        clustering = self._assess_temporal_clustering(events)
        factors.append(clustering)
        # 5. Spatial clustering factor
        spatial_clustering = self._assess_spatial_clustering(events)
        factors.append(spatial_clustering)
        # Combine factors with weights
        weights = [0.25, 0.25, 0.20, 0.15, 0.15]  # Sum to 1.0
        # Ensure we have enough factors
        n_factors = min(len(factors), len(weights))
        if n_factors == 0:
            return 0.5
        # Weighted average
        weighted_sum = sum(f * w for f, w in zip(factors[:n_factors], weights[:n_factors]))
        weight_sum = sum(weights[:n_factors])
        seismic_index = weighted_sum / weight_sum
        # Apply nonlinear scaling
        if seismic_index > 0.7:
            # Amplify high values
            seismic_index = 0.7 + (seismic_index - 0.7) * 1.5
        elif seismic_index < 0.3:
            # Reduce low values
            seismic_index = seismic_index * 0.8
        return np.clip(seismic_index, 0.0, 1.0)
    def _calculate_uncertainty(self, events: List[Dict[str, Any]], 
                              b_value: float, b_uncertainty: float) -> float:
        """Calculate overall uncertainty in seismic assessment."""
        if len(events) < 3:
            return 0.5  # High uncertainty with few events
        # Base uncertainty decreases with more events
        base_uncertainty = 0.3 / (1 + math.log1p(len(events)))
        # Add b-value uncertainty
        b_uncertainty_contrib = b_uncertainty * 0.3
        # Add magnitude range uncertainty
        magnitudes = [e['magnitude'] for e in events if 'magnitude' in e]
        if magnitudes:
            mag_range = max(magnitudes) - min(magnitudes)
            mag_uncertainty = min(0.2, mag_range * 0.1)
        else:
            mag_uncertainty = 0.15
        total_uncertainty = base_uncertainty + b_uncertainty_contrib + mag_uncertainty
        return np.clip(total_uncertainty, 0.05, 0.5)
    def _analyze_depth_distribution(self, events: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze depth distribution of earthquakes."""
        depths = [e['depth'] for e in events if 'depth' in e]
        stats = {
            'count': len(depths),
            'mean_depth': 0.0,
            'min_depth': 0.0,
            'max_depth': 0.0,
            'shallow_ratio': 0.0  # Ratio of shallow (<10km) events
        }
        if depths:
            stats['mean_depth'] = float(np.mean(depths))
            stats['min_depth'] = float(min(depths))
            stats['max_depth'] = float(max(depths))
            # Calculate shallow event ratio
            shallow_count = sum(1 for d in depths if d < 10.0)
            stats['shallow_ratio'] = shallow_count / len(depths)
        return stats
    def _analyze_temporal_patterns(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns of earthquakes."""
        if len(events) < 2:
            return {'event_count': len(events), 'rate': 0.0}
        # Extract times
        times = []
        for event in events:
            if 'time' in event:
                try:
                    if isinstance(event['time'], (int, float)):
                        times.append(event['time'])
                    elif isinstance(event['time'], str):
                        # Try to parse ISO format
                        dt = datetime.fromisoformat(event['time'].replace('Z', '+00:00'))
                        times.append(dt.timestamp())
                except:
                    continue
        if len(times) < 2:
            return {'event_count': len(events), 'rate': 0.0}
        times.sort()
        # Calculate rates
        time_range = times[-1] - times[0]
        if time_range > 0:
            rate = len(times) / (time_range / 86400)  # Events per day
        else:
            rate = 0.0
        # Check for temporal clustering
        intervals = np.diff(times)
        if len(intervals) > 0:
            mean_interval = np.mean(intervals)
            cv = np.std(intervals) / mean_interval if mean_interval > 0 else 0
            clustered = cv > 1.5  # High coefficient of variation indicates clustering
        else:
            clustered = False
        return {
            'event_count': len(times),
            'time_range_days': time_range / 86400 if time_range > 0 else 0,
            'rate_per_day': float(rate),
            'is_clustered': clustered,
            'mean_interval_hours': float(np.mean(intervals) / 3600) if len(intervals) > 0 else 0
        }
    def _analyze_spatial_clusters(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze spatial clustering of earthquakes."""
        locations = []
        for event in events:
            if 'latitude' in event and 'longitude' in event:
                locations.append((event['latitude'], event['longitude']))
        if len(locations) < 3:
            return {'location_count': len(locations), 'clustered': False}
        # Simple spatial clustering detection
        lats = [loc[0] for loc in locations]
        lons = [loc[1] for loc in locations]
        # Calculate spatial extent
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        # Simple clustering heuristic
        area = lat_range * lon_range
        density = len(locations) / max(area, 0.01)  # Avoid division by zero
        # Consider clustered if density > threshold
        clustered = density > 10.0  # More than 10 events per degree squared
        return {
            'location_count': len(locations),
            'spatial_extent_deg': {'lat': float(lat_range), 'lon': float(lon_range)},
            'density': float(density),
            'clustered': clustered,
            'centroid': {'lat': float(np.mean(lats)), 'lon': float(np.mean(lons))}
        }
    def _assess_temporal_clustering(self, events: List[Dict[str, Any]]) -> float:
        """Assess temporal clustering level."""
        temporal_stats = self._analyze_temporal_patterns(events)
        if temporal_stats['is_clustered']:
            return 0.8  # High clustering = higher risk
        elif temporal_stats['rate_per_day'] > 10:
            return 0.6  # High rate = moderate risk
        elif temporal_stats['rate_per_day'] > 5:
            return 0.4  # Moderate rate = low risk
        else:
            return 0.2  # Low rate = very low risk
    def _assess_spatial_clustering(self, events: List[Dict[str, Any]]) -> float:
        """Assess spatial clustering level."""
        spatial_stats = self._analyze_spatial_clusters(events)
        if spatial_stats['clustered']:
            return 0.8  # High clustering = higher risk
        elif spatial_stats['density'] > 5:
            return 0.5  # Moderate density = moderate risk
        else:
            return 0.2  # Low density = low risk
    def _generate_warnings(self, events: List[Dict[str, Any]], 
                          seismic_index: float, b_value: float) -> List[str]:
        """Generate warnings based on seismic analysis."""
        warnings = []
        # Check magnitude thresholds
        magnitudes = [e['magnitude'] for e in events if 'magnitude' in e]
        if magnitudes:
            max_mag = max(magnitudes)
            if max_mag > 6.0:
                warnings.append("CRITICAL: Major earthquake detected (M>6.0)")
            elif max_mag > 5.0:
                warnings.append("WARNING: Strong earthquake detected (M>5.0)")
            elif max_mag > 4.0:
                warnings.append("ALERT: Moderate earthquake detected (M>4.0)")
        # Check b-value (low b-value indicates higher stress)
        if b_value < 0.7:
            warnings.append("HIGH STRESS: Low b-value indicates high stress accumulation")
        elif b_value < 0.9:
            warnings.append("ELEVATED STRESS: Below-average b-value detected")
        # Check event rate
        temporal_stats = self._analyze_temporal_patterns(events)
        if temporal_stats['rate_per_day'] > 20:
            warnings.append("HIGH ACTIVITY: Elevated earthquake rate detected")
        elif temporal_stats['rate_per_day'] > 10:
            warnings.append("INCREASED ACTIVITY: Above-average earthquake rate")
        # Check overall seismic index
        if seismic_index > 0.8:
            warnings.append("CRITICAL: High seismic risk detected")
        elif seismic_index > 0.6:
            warnings.append("WARNING: Elevated seismic risk")
        elif seismic_index > 0.4:
            warnings.append("ALERT: Moderate seismic activity")
        return warnings
    def _generate_recommendations(self, seismic_index: float,
                                 warnings: List[str]) -> List[str]:
        """Generate recommendations based on seismic index and warnings."""
        recommendations = []
        if seismic_index > 0.8 or any("CRITICAL" in w for w in warnings):
            recommendations = [
                "Immediate deployment of field response team",
                "Activate emergency response protocols",
                "Issue public safety warnings",
                "Increase monitoring to highest frequency",
                "Prepare for potential aftershocks"
            ]
        elif seismic_index > 0.6 or any("WARNING" in w for w in warnings):
            recommendations = [
                "Increase monitoring frequency",
                "Review emergency response plans",
                "Alert relevant authorities",
                "Prepare field equipment",
                "Update risk assessments"
            ]
        elif seismic_index > 0.4 or any("ALERT" in w for w in warnings):
            recommendations = [
                "Close monitoring of seismic activity",
                "Check equipment status",
                "Review historical patterns",
                "Update hazard maps",
                "Prepare for increased monitoring"
            ]
        else:
            recommendations = [
                "Continue routine monitoring",
                "Regular data quality checks",
                "Maintain equipment calibration",
                "Review historical data",
                "Update baseline measurements"
            ]
        return recommendations
    def _get_time_range(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get time range of events."""
        times = []
        for event in events:
            if 'time' in event:
                try:
                    if isinstance(event['time'], (int, float)):
                        times.append(event['time'])
                except:
                    continue
        if times:
            return {
                'start': min(times),
                'end': max(times),
                'duration_days': (max(times) - min(times)) / 86400 if len(times) > 1 else 0
            }
        else:
            return {'start': None, 'end': None, 'duration_days': 0}
    def _get_magnitude_range(self, events: List[Dict[str, Any]]) -> Dict[str, float]:
        """Get magnitude range of events."""
        magnitudes = [e['magnitude'] for e in events if 'magnitude' in e]
        if magnitudes:
            return {
                'min': min(magnitudes),
                'max': max(magnitudes),
                'mean': np.mean(magnitudes),
                'std': np.std(magnitudes)
            }
        else:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0}
    def _get_default_result(self) -> Dict[str, Any]:
        """Return default seismic analysis result."""
        return {
            'seismic_index': 0.1,
            'b_value': self.config['b_value_default'],
            'uncertainty': 0.3,
            'event_count': 0,
            'max_magnitude': 0.0,
            'depth_distribution': {},
            'temporal_patterns': {'event_count': 0, 'rate': 0.0},
            'spatial_clusters': {'location_count': 0, 'clustered': False},
            'warnings': [],
            'recommendations': [
                "No seismic activity detected",
                "Continue routine monitoring",
                "Check equipment connectivity"
            ],
            'metadata': {
                'analysis_time': datetime.now().isoformat(),
                'events_analyzed': 0,
                'note': 'Default result - no data available'
            }
        }
    def calculate_earthquake_probability(self, historical_data: pd.DataFrame,
                                        time_window: int = 7) -> float:
        """
        Calculate earthquake probability for given time window.
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical earthquake data
        time_window : int
            Prediction window in days
        Returns:
        --------
        float
            Probability of significant earthquake (M≥5.0) within time window
        """
        try:
            if historical_data.empty:
                return 0.05  # Base rate for no data
            # Simple probability calculation based on historical rate
            if 'magnitude' in historical_data.columns and 'time' in historical_data.columns:
                # Count significant events
                significant_events = historical_data[
                    historical_data['magnitude'] >= 5.0
                ]
                if len(significant_events) >= 3:
                    # Calculate historical rate
                    times = pd.to_datetime(historical_data['time'])
                    time_range_days = (times.max() - times.min()).days
                    if time_range_days > 0:
                        historical_rate = len(significant_events) / time_range_days
                        # Project to requested time window
                        probability = 1 - math.exp(-historical_rate * time_window)
                        # Constrain to reasonable range
                        return np.clip(probability, 0.01, 0.95)
            return 0.1  # Default moderate probability
        except Exception as e:
            logger.error(f"Error in probability calculation: {e}")
            return 0.05
# Simple utility function
def analyze_seismic_simple(magnitudes: List[float], 
                          depths: Optional[List[float]] = None) -> float:
    """
    Simple seismic analysis.
    Parameters:
    -----------
    magnitudes : list
        Earthquake magnitudes
    depths : list, optional
        Earthquake depths in km
    Returns:
    --------
    float
        Seismic index (0-1)
    """
    analyzer = SeismicAnalyzer()
    events = []
    for i, mag in enumerate(magnitudes):
        event = {'magnitude': mag}
        if depths and i < len(depths):
            event['depth'] = depths[i]
        event['time'] = datetime.now().timestamp() - i * 3600  # Fake times
        events.append(event)
    data = {'events': events}
    result = analyzer.analyze(data)
    return result['seismic_index']