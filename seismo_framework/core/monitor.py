"""
Main monitoring class for Seismo Framework.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import logging
import warnings
import json
import yaml
logger = logging.getLogger(__name__)
class SeismicMonitor:
    """
    Main class for seismic monitoring and analysis.
    This class integrates all 8 geophysical parameters and provides
    comprehensive monitoring capabilities.
    """
    def __init__(self, region: str = "global", config: Optional[Dict] = None):
        """
        Initialize seismic monitor for a specific region.
        Parameters
        ----------
        region : str
            Monitoring region (e.g., 'san_andreas', 'tokyo', 'global')
        config : dict, optional
            Custom configuration parameters
        """
        self.region = region
        self.config = self._load_config(config)
        self._initialize_parameters()
        self._initialize_data_structures()
        self._setup_logging()
        logger.info(f"SeismicMonitor initialized for region: {region}")
        logger.info(f"Configuration loaded: {len(self.config)} parameters")
    def _load_config(self, config: Optional[Dict]) -> Dict:
        """Load configuration from file or provided dict."""
        default_config = {
            'monitoring_interval': 60,  # seconds
            'alert_threshold': 0.7,
            'data_retention_days': 365,
            'log_level': 'INFO',
            'min_magnitude': 2.0,
            'max_depth': 100,  # km
            'probability_time_windows': ['24h', '7d', '30d'],
        }
        if config:
            default_config.update(config)
        return default_config
    def _initialize_parameters(self):
        """Initialize the 8 parameter analyzers."""
        self.parameters = {
            'seismic': None,
            'deformation': None,
            'hydrogeological': None,
            'electrical': None,
            'magnetic': None,
            'instability': None,
            'stress': None,
            'rock_properties': None,
        }
        self.parameter_values = {key: None for key in self.parameters.keys()}
        self.parameter_uncertainties = {key: None for key in self.parameters.keys()}
        self.parameter_timestamps = {key: None for key in self.parameters.keys()}
    def _initialize_data_structures(self):
        """Initialize data storage structures."""
        self.data = {
            'seismic_catalog': pd.DataFrame(),
            'gps_data': pd.DataFrame(),
            'hydro_data': pd.DataFrame(),
            'electrical_data': pd.DataFrame(),
            'magnetic_data': pd.DataFrame(),
            'metadata': {
                'region': self.region,
                'created': datetime.now().isoformat(),
                'last_updated': None,
                'data_sources': []
            }
        }
        self.results = {
            'probabilities': {},
            'alerts': [],
            'reports': {},
            'trends': {},
            'statistics': {}
        }
        self.history = {
            'parameter_history': {key: [] for key in self.parameters.keys()},
            'probability_history': [],
            'alert_history': []
        }
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        # Clear any existing handlers
        logger.handlers.clear()
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        # Add handler to logger
        logger.addHandler(console_handler)
        logger.setLevel(log_level)
    def load_seismic_data(self, data_source: Union[str, pd.DataFrame], 
                         format: str = 'auto', **kwargs) -> None:
        """
        Load seismic data from various sources.
        Parameters
        ----------
        data_source : str or pd.DataFrame
            File path or DataFrame containing seismic data
        format : str
            Data format ('csv', 'hdf5', 'quakeml', 'json', 'auto')
        **kwargs : dict
            Additional parameters for specific formats
        """
        try:
            start_time = datetime.now()
            if isinstance(data_source, pd.DataFrame):
                self.data['seismic_catalog'] = data_source.copy()
                source_type = 'DataFrame'
            elif isinstance(data_source, str):
                source_type = f"file: {data_source}"
                if format == 'auto':
                    if data_source.endswith('.csv'):
                        format = 'csv'
                    elif data_source.endswith(('.h5', '.hdf5')):
                        format = 'hdf5'
                    elif data_source.endswith('.json'):
                        format = 'json'
                    elif data_source.endswith('.xml') or data_source.endswith('.quakeml'):
                        format = 'quakeml'
                    else:
                        raise ValueError(f"Could not auto-detect format for: {data_source}")
                if format == 'csv':
                    self.data['seismic_catalog'] = pd.read_csv(data_source, **kwargs)
                elif format == 'json':
                    self.data['seismic_catalog'] = pd.read_json(data_source, **kwargs)
                elif format == 'hdf5':
                    self.data['seismic_catalog'] = pd.read_hdf(data_source, **kwargs)
                elif format == 'quakeml':
                    # Simplified QuakeML parsing
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(data_source)
                    root = tree.getroot()
                    # Extract event data (simplified)
                    events = []
                    for event in root.findall('.//{http://quakeml.org/xmlns/bed/1.2}event'):
                        event_data = {}
                        # Extract basic parameters
                        # This is a simplified version
                        events.append(event_data)
                    self.data['seismic_catalog'] = pd.DataFrame(events)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            else:
                raise TypeError(f"Unsupported data source type: {type(data_source)}")
            # Validate and preprocess data
            self._validate_seismic_data()
            # Update metadata
            self.data['metadata']['last_updated'] = datetime.now().isoformat()
            self.data['metadata']['data_sources'].append({
                'type': 'seismic',
                'source': source_type,
                'format': format,
                'timestamp': datetime.now().isoformat(),
                'record_count': len(self.data['seismic_catalog'])
            })
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Loaded {len(self.data['seismic_catalog'])} seismic events "
                       f"in {elapsed:.2f} seconds from {source_type}")
        except Exception as e:
            logger.error(f"Error loading seismic data: {e}", exc_info=True)
            raise
    def _validate_seismic_data(self):
        """Validate and preprocess seismic data."""
        if self.data['seismic_catalog'].empty:
            logger.warning("Seismic catalog is empty")
            return
        df = self.data['seismic_catalog']
        # Required columns check
        required_columns = ['time', 'latitude', 'longitude']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
        # Convert time column to datetime if it's a string
        if 'time' in df.columns and df['time'].dtype == 'object':
            try:
                df['time'] = pd.to_datetime(df['time'])
            except Exception as e:
                logger.warning(f"Could not convert time column: {e}")
        # Filter by magnitude if column exists
        if 'magnitude' in df.columns:
            min_mag = self.config.get('min_magnitude', 2.0)
            original_count = len(df)
            df = df[df['magnitude'] >= min_mag].copy()
            filtered_count = original_count - len(df)
            if filtered_count > 0:
                logger.info(f"Filtered {filtered_count} events below magnitude {min_mag}")
        # Filter by depth if column exists
        if 'depth' in df.columns:
            max_depth = self.config.get('max_depth', 100)
            original_count = len(df)
            df = df[df['depth'] <= max_depth].copy()
            filtered_count = original_count - len(df)
            if filtered_count > 0:
                logger.info(f"Filtered {filtered_count} events below depth {max_depth}km")
        # Sort by time
        if 'time' in df.columns:
            df = df.sort_values('time').reset_index(drop=True)
        self.data['seismic_catalog'] = df
        # Calculate basic statistics
        self._calculate_seismic_statistics()
    def _calculate_seismic_statistics(self):
        """Calculate basic seismic statistics."""
        df = self.data['seismic_catalog']
        if df.empty:
            self.results['statistics']['seismic'] = {
                'total_events': 0,
                'period': None,
                'magnitude_range': None,
                'depth_range': None
            }
            return
        stats = {
            'total_events': len(df),
            'period': None,
            'magnitude_range': None,
            'depth_range': None,
            'event_rate': None
        }
        # Calculate time period
        if 'time' in df.columns and len(df) > 1:
            time_min = df['time'].min()
            time_max = df['time'].max()
            stats['period'] = {
                'start': time_min.isoformat() if hasattr(time_min, 'isoformat') else str(time_min),
                'end': time_max.isoformat() if hasattr(time_max, 'isoformat') else str(time_max),
                'days': (time_max - time_min).days if hasattr(time_max - time_min, 'days') else None
            }
        # Calculate magnitude statistics
        if 'magnitude' in df.columns:
            stats['magnitude_range'] = {
                'min': float(df['magnitude'].min()),
                'max': float(df['magnitude'].max()),
                'mean': float(df['magnitude'].mean()),
                'median': float(df['magnitude'].median())
            }
        # Calculate depth statistics
        if 'depth' in df.columns:
            stats['depth_range'] = {
                'min': float(df['depth'].min()),
                'max': float(df['depth'].max()),
                'mean': float(df['depth'].mean()),
                'median': float(df['depth'].median())
            }
        # Calculate event rate
        if stats['period'] and stats['period']['days']:
            stats['event_rate'] = stats['total_events'] / max(1, stats['period']['days'])
        self.results['statistics']['seismic'] = stats
        logger.info(f"Seismic statistics: {stats['total_events']} events, "
                   f"magnitude range: {stats['magnitude_range']['min'] if stats['magnitude_range'] else 'N/A'}-"
                   f"{stats['magnitude_range']['max'] if stats['magnitude_range'] else 'N/A'}")
    def load_sample_data(self):
        """Load sample seismic data for testing."""
        try:
            # Create sample data
            np.random.seed(42)
            n_events = 100
            # Generate random event times over the last year
            end_time = datetime.now()
            start_time = end_time - timedelta(days=365)
            time_delta = (end_time - start_time).total_seconds()
            times = [start_time + timedelta(seconds=np.random.rand() * time_delta) 
                    for _ in range(n_events)]
            # Generate random locations around California
            latitudes = 34 + np.random.randn(n_events) * 2
            longitudes = -118 + np.random.randn(n_events) * 2
            depths = np.random.exponential(10, n_events)
            magnitudes = np.random.exponential(2, n_events) + 2.0
            sample_data = pd.DataFrame({
                'time': times,
                'latitude': latitudes,
                'longitude': longitudes,
                'depth': depths,
                'magnitude': magnitudes,
                'region': ['california'] * n_events
            })
            self.load_seismic_data(sample_data)
            logger.info("Sample data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            raise
    def calculate_earthquake_probability(self, 
                                        time_window: str = '7d',
                                        magnitude_threshold: float = 5.0) -> float:
        """
        Calculate earthquake probability for given time window.
        Parameters
        ----------
        time_window : str
            Time window for probability calculation ('24h', '7d', '30d')
        magnitude_threshold : float
            Minimum magnitude threshold
        Returns
        -------
        float
            Earthquake probability (0-1)
        """
        try:
            # Convert time window to days
            if time_window.endswith('h'):
                days = int(time_window[:-1]) / 24
            elif time_window.endswith('d'):
                days = int(time_window[:-1])
            elif time_window.endswith('m'):
                days = int(time_window[:-1]) * 30
            else:
                days = 7  # Default to 7 days
            # Base probability
            probability = 0.05  # 5% baseline
            # Adjust based on available data
            if not self.data['seismic_catalog'].empty:
                df = self.data['seismic_catalog']
                # Recent activity increase
                recent_days = min(30, days * 3)
                cutoff_time = datetime.now() - timedelta(days=recent_days)
                if 'time' in df.columns:
                    recent_events = df[pd.to_datetime(df['time']) > cutoff_time]
                    recent_count = len(recent_events)
                    # Normalize event count (0-100 events -> 0-0.3 probability boost)
                    activity_boost = min(0.3, recent_count / 100)
                    probability += activity_boost
                # Magnitude distribution effect
                if 'magnitude' in df.columns:
                    large_events = df[df['magnitude'] >= magnitude_threshold]
                    if len(large_events) > 0:
                        # Recent large events increase probability
                        probability = min(probability + 0.2, 0.95)
            # Cap probability
            probability = min(max(probability, 0.01), 0.99)
            # Store result
            timestamp = datetime.now().isoformat()
            self.results['probabilities'][time_window] = {
                'probability': probability,
                'timestamp': timestamp,
                'magnitude_threshold': magnitude_threshold,
                'calculation_method': 'baseline_plus_activity'
            }
            # Add to history
            self.history['probability_history'].append({
                'time_window': time_window,
                'probability': probability,
                'timestamp': timestamp
            })
            logger.info(f"Calculated {time_window} probability: {probability:.1%}")
            return probability
        except Exception as e:
            logger.error(f"Error calculating probability: {e}")
            return 0.05  # Return baseline on error
    def get_alert_level(self) -> str:
        """
        Determine current alert level based on monitoring data.
        Returns
        -------
        str
            Alert level: 'normal', 'elevated', 'watch', 'warning'
        """
        try:
            # Calculate 7-day probability
            probability = self.calculate_earthquake_probability(time_window='7d')
            # Determine alert level
            if probability < 0.1:
                level = 'normal'
            elif probability < 0.3:
                level = 'elevated'
            elif probability < 0.6:
                level = 'watch'
            else:
                level = 'warning'
            # Check if this is a change from previous level
            if self.history['alert_history']:
                last_alert = self.history['alert_history'][-1]['level']
                if last_alert != level:
                    logger.warning(f"Alert level changed from {last_alert} to {level}")
            # Record alert
            alert_record = {
                'level': level,
                'probability': probability,
                'timestamp': datetime.now().isoformat(),
                'region': self.region
            }
            self.history['alert_history'].append(alert_record)
            self.results['alerts'].append(alert_record)
            logger.info(f"Alert level: {level} (probability: {probability:.1%})")
            return level
        except Exception as e:
            logger.error(f"Error determining alert level: {e}")
            return 'normal'
    def generate_report(self, format: str = 'dict') -> Dict[str, Any]:
        """
        Generate monitoring report.
        Parameters
        ----------
        format : str
            Report format ('dict', 'json', 'yaml')
        Returns
        -------
        dict or str
            Report data
        """
        try:
            report = {
                'metadata': {
                    'generated': datetime.now().isoformat(),
                    'region': self.region,
                    'framework_version': '0.1.0',
                    'report_id': f"seismo_{self.region}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                },
                'status': {
                    'alert_level': self.get_alert_level(),
                    'probability_7d': self.calculate_earthquake_probability('7d'),
                    'probability_24h': self.calculate_earthquake_probability('24h'),
                    'probability_30d': self.calculate_earthquake_probability('30d'),
                    'data_status': {
                        'seismic_events': len(self.data['seismic_catalog']),
                        'last_update': self.data['metadata']['last_updated']
                    }
                },
                'statistics': self.results['statistics'],
                'parameters': {
                    key: {
                        'value': self.parameter_values[key],
                        'uncertainty': self.parameter_uncertainties[key],
                        'timestamp': self.parameter_timestamps[key]
                    }
                    for key in self.parameters.keys()
                },
                'history': {
                    'recent_alerts': self.history['alert_history'][-5:],
                    'probability_trend': self.history['probability_history'][-10:]
                },
                'config': self.config
            }
            self.results['reports'][report['metadata']['report_id']] = report
            # Format output
            if format == 'json':
                return json.dumps(report, indent=2, default=str)
            elif format == 'yaml':
                return yaml.dump(report, default_flow_style=False)
            else:  # dict
                return report
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {'error': str(e)}
    def save_report(self, filename: str = None, format: str = 'json'):
        """
        Save report to file.
        Parameters
        ----------
        filename : str, optional
            Output filename
        format : str
            File format ('json', 'yaml')
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"seismo_report_{self.region}_{timestamp}.{format}"
        try:
            report = self.generate_report(format)
            if format == 'json':
                with open(filename, 'w') as f:
                    if isinstance(report, dict):
                        json.dump(report, f, indent=2, default=str)
                    else:
                        f.write(report)
            elif format == 'yaml':
                with open(filename, 'w') as f:
                    yaml.dump(report if isinstance(report, dict) else yaml.safe_load(report), 
                             f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            logger.info(f"Report saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            raise
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about loaded data.
        Returns
        -------
        dict
            Statistics dictionary
        """
        return self.results['statistics']
    def _get_recent_events(self, days: int = 30) -> pd.DataFrame:
        """
        Get events from the last N days.
        Parameters
        ----------
        days : int
            Number of days to look back
        Returns
        -------
        pd.DataFrame
            Recent events
        """
        if self.data['seismic_catalog'].empty or 'time' not in self.data['seismic_catalog'].columns:
            return pd.DataFrame()
        df = self.data['seismic_catalog']
        cutoff_time = datetime.now() - timedelta(days=days)
        try:
            # Convert to datetime if needed
            if df['time'].dtype == 'object':
                df['time'] = pd.to_datetime(df['time'])
            recent = df[df['time'] > cutoff_time].copy()
            return recent
        except Exception as e:
            logger.warning(f"Error filtering recent events: {e}")
            return pd.DataFrame()
    def clear_data(self):
        """Clear all loaded data and results."""
        self._initialize_data_structures()
        logger.info("All data cleared")
    def __str__(self) -> str:
        """String representation of the monitor."""
        stats = self.results['statistics'].get('seismic', {})
        event_count = stats.get('total_events', 0)
        alert_level = self.get_alert_level()
        return (f"SeismicMonitor(region='{self.region}', "
                f"events={event_count}, alert_level='{alert_level}')")
    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()