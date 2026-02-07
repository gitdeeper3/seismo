"""
Real-time monitoring system for volcano parameters.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import asyncio
import time
import threading
from queue import Queue
logger = logging.getLogger(__name__)
class RealTimeMonitor:
    """
    Real-time monitoring system for continuous parameter tracking.
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._setup_defaults()
        # Monitoring state
        self.is_running = False
        self.last_update = None
        self.monitoring_thread = None
        self.data_queue = Queue()
        # Data storage
        self.historical_data = pd.DataFrame()
        self.current_state = {}
        self.alerts = []
        # Integration components
        from ..integration.algorithms import ParameterIntegrator
        from ..integration.weighting import WeightOptimizer
        self.integrator = ParameterIntegrator()
        self.weight_optimizer = WeightOptimizer()
        logger.info("Real-time monitor initialized")
    def _setup_defaults(self):
        defaults = {
            'monitoring_interval': 60,  # seconds
            'data_retention_days': 30,
            'alert_threshold': 0.7,
            'parameter_sources': {
                'seismic': 'auto',
                'deformation': 'auto',
                'hydrogeological': 'auto',
                'electrical': 'auto',
                'magnetic': 'auto',
                'instability': 'auto',
                'stress': 'auto',
                'rock_properties': 'auto'
            },
            'data_sampling': {
                'seismic': 10,  # Hz
                'deformation': 1,  # Hz
                'hydrogeological': 0.1,  # Hz
                'electrical': 5,  # Hz
                'magnetic': 2,  # Hz
                'instability': 0.5,  # Hz
                'stress': 1,  # Hz
                'rock_properties': 0.2  # Hz
            },
            'alert_levels': {
                'normal': {'min': 0.0, 'max': 0.3},
                'elevated': {'min': 0.3, 'max': 0.5},
                'watch': {'min': 0.5, 'max': 0.7},
                'warning': {'min': 0.7, 'max': 1.0}
            }
        }
        defaults.update(self.config)
        self.config = defaults
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        self.is_running = True
        self.last_update = datetime.now()
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Real-time monitoring started (interval: {self.config['monitoring_interval']}s)")
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Real-time monitoring stopped")
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Collect data from all parameters
                current_data = self._collect_parameter_data()
                # Process and integrate
                processed_data = self._process_parameters(current_data)
                # Update current state
                self.current_state = processed_data
                # Check for alerts
                self._check_alerts(processed_data)
                # Store historical data
                self._store_historical_data(processed_data)
                # Update timestamp
                self.last_update = datetime.now()
                # Notify listeners
                self._notify_listeners(processed_data)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            # Sleep until next interval
            time.sleep(self.config['monitoring_interval'])
    def _collect_parameter_data(self) -> Dict[str, Dict[str, Any]]:
        """Collect data from all parameter sources."""
        data = {}
        # Import parameter modules
        from ..parameters.seismic import SeismicAnalyzer
        from ..parameters.deformation import DeformationAnalyzer
        from ..parameters.hydrogeological import HydrogeologicalAnalyzer
        from ..parameters.electrical import ElectricalAnalyzer
        from ..parameters.magnetic import MagneticAnalyzer
        from ..parameters.instability import InstabilityAnalyzer
        from ..parameters.stress import StressAnalyzer
        from ..parameters.rock_properties import RockPropertiesAnalyzer
        # Define analyzers
        analyzers = {
            'seismic': SeismicAnalyzer(),
            'deformation': DeformationAnalyzer(),
            'hydrogeological': HydrogeologicalAnalyzer(),
            'electrical': ElectricalAnalyzer(),
            'magnetic': MagneticAnalyzer(),
            'instability': InstabilityAnalyzer(),
            'stress': StressAnalyzer(),
            'rock_properties': RockPropertiesAnalyzer()
        }
        # Collect data from each analyzer
        for param_name, analyzer in analyzers.items():
            try:
                # For real-time monitoring, we would normally get live data
                # For now, generate synthetic data or use last known values
                param_data = self._get_parameter_simulation(param_name, analyzer)
                data[param_name] = param_data
            except Exception as e:
                logger.warning(f"Failed to collect data for {param_name}: {e}")
                # Provide default data
                data[param_name] = {
                    f'{param_name}_index': 0.5,
                    'uncertainty': 0.2,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'default'
                }
        return data
    def _get_parameter_simulation(self, param_name: str, analyzer) -> Dict[str, Any]:
        """Generate simulated parameter data for testing."""
        # In a real system, this would connect to actual sensors
        # For simulation, generate realistic values
        # Base value with some randomness
        np.random.seed(int(time.time()) + hash(param_name))
        if param_name == 'seismic':
            # Seismic activity simulation
            value = np.random.beta(2, 5)  # Usually low, but can spike
            if np.random.random() < 0.05:  # 5% chance of spike
                value = min(1.0, value + np.random.exponential(0.3))
        elif param_name == 'deformation':
            # Deformation simulation
            value = np.random.beta(3, 4)
            # Slow trends
            trend = np.sin(time.time() / 86400) * 0.1 + 0.5
            value = (value + trend) / 2
        elif param_name == 'hydrogeological':
            # Hydrogeological simulation
            value = np.random.beta(4, 3)
            # Diurnal cycle
            diurnal = np.sin(time.time() / 43200) * 0.15 + 0.5
            value = (value + diurnal) / 2
        elif param_name == 'electrical':
            # Electrical signals simulation
            value = np.random.beta(2, 6)  # Usually very low
            if np.random.random() < 0.02:  # 2% chance of anomaly
                value = min(1.0, value + np.random.exponential(0.4))
        elif param_name == 'magnetic':
            # Magnetic anomalies simulation
            value = np.random.beta(3, 5)
            # Random spikes
            if np.random.random() < 0.03:
                value = min(1.0, value + 0.2)
        elif param_name == 'instability':
            # Instability simulation
            value = np.random.beta(5, 3)  # Usually moderate
            # Gradual increase possibility
            if np.random.random() < 0.1:
                value = min(1.0, value * 1.1)
        elif param_name == 'stress':
            # Stress simulation
            value = np.random.beta(4, 4)  # Balanced
            # Tectonic cycles
            tectonic = np.sin(time.time() / 604800) * 0.2 + 0.5
            value = (value + tectonic) / 2
        elif param_name == 'rock_properties':
            # Rock properties simulation
            value = np.random.beta(6, 2)  # Usually stable
            # Very slow changes
            if np.random.random() < 0.01:
                value = max(0.0, value - 0.05)
        else:
            value = 0.5
        # Add some measurement noise
        value += np.random.normal(0, 0.05)
        value = np.clip(value, 0.0, 1.0)
        # Calculate uncertainty
        uncertainty = 0.1 + np.random.exponential(0.05)
        uncertainty = min(0.3, uncertainty)
        return {
            f'{param_name}_index': float(value),
            'uncertainty': float(uncertainty),
            'timestamp': datetime.now().isoformat(),
            'source': 'simulation',
            'raw_value': float(value),
            'quality': 'good' if uncertainty < 0.15 else 'moderate'
        }
    def _process_parameters(self, raw_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Process and integrate parameter data."""
        # Extract parameter values
        parameter_values = {}
        for param_name, param_data in raw_data.items():
            key = f'{param_name}_index'
            if key in param_data:
                parameter_values[param_name] = param_data[key]
        # Integrate parameters
        integration_result = self.integrator.integrate(parameter_values)
        # Prepare comprehensive result
        result = {
            'timestamp': datetime.now().isoformat(),
            'raw_parameters': raw_data,
            'parameter_values': parameter_values,
            'integration': integration_result,
            'metadata': {
                'parameters_available': list(parameter_values.keys()),
                'total_parameters': len(parameter_values),
                'processing_time': time.time()
            }
        }
        # Calculate trends if we have historical data
        if not self.historical_data.empty:
            trends = self._calculate_trends(parameter_values)
            result['trends'] = trends
        return result
    def _calculate_trends(self, current_values: Dict[str, float]) -> Dict[str, Any]:
        """Calculate trends from historical data."""
        trends = {}
        if self.historical_data.empty:
            return trends
        try:
            # Get recent data (last 24 hours)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_mask = pd.to_datetime(self.historical_data['timestamp']) > recent_cutoff
            if recent_mask.any():
                recent_data = self.historical_data[recent_mask]
                # Calculate trends for each parameter
                for param_name in current_values.keys():
                    param_col = f'param_{param_name}'
                    if param_col in recent_data.columns:
                        param_series = recent_data[param_col].dropna()
                        if len(param_series) >= 2:
                            # Linear trend
                            x = np.arange(len(param_series))
                            y = param_series.values
                            if len(np.unique(y)) > 1:
                                slope, intercept = np.polyfit(x, y, 1)
                                trends[param_name] = {
                                    'slope': float(slope),
                                    'intercept': float(intercept),
                                    'current': float(current_values.get(param_name, 0.5)),
                                    'trend_direction': 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable',
                                    'trend_magnitude': abs(float(slope)),
                                    'data_points': len(param_series)
                                }
                # Calculate integrated score trend
                if 'integrated_score' in recent_data.columns:
                    integrated_series = recent_data['integrated_score'].dropna()
                    if len(integrated_series) >= 2:
                        x = np.arange(len(integrated_series))
                        y = integrated_series.values
                        if len(np.unique(y)) > 1:
                            slope, intercept = np.polyfit(x, y, 1)
                            trends['integrated'] = {
                                'slope': float(slope),
                                'intercept': float(intercept),
                                'current': float(self.current_state.get('integration', {}).get('integrated_score', 0.5)),
                                'trend_direction': 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable',
                                'trend_magnitude': abs(float(slope))
                            }
        except Exception as e:
            logger.warning(f"Error calculating trends: {e}")
        return trends
    def _check_alerts(self, processed_data: Dict[str, Any]):
        """Check for alert conditions."""
        integration_result = processed_data.get('integration', {})
        score = integration_result.get('integrated_score', 0.5)
        alert_level = integration_result.get('alert_level', 'normal')
        # Check if alert level has changed
        current_alert = None
        if self.alerts:
            current_alert = self.alerts[-1]
        if not current_alert or current_alert.get('alert_level') != alert_level:
            # New or changed alert
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'alert_level': alert_level,
                'integrated_score': score,
                'parameters': processed_data.get('parameter_values', {}),
                'confidence': integration_result.get('confidence', 0.5),
                'message': self._generate_alert_message(alert_level, score, processed_data)
            }
            self.alerts.append(alert_data)
            # Log alert
            if alert_level != 'normal':
                logger.warning(f"Alert triggered: {alert_level} (score: {score:.3f})")
            else:
                logger.info(f"Alert level normal (score: {score:.3f})")
            # Store alert in historical data
            self._store_alert(alert_data)
    def _generate_alert_message(self, alert_level: str, score: float,
                               processed_data: Dict[str, Any]) -> str:
        """Generate alert message."""
        messages = {
            'normal': "All parameters within normal ranges.",
            'elevated': "Slight increase in some parameters. Monitor closely.",
            'watch': "Multiple parameters showing significant increases. Increased monitoring required.",
            'warning': "Critical levels detected in multiple parameters. Immediate action recommended."
        }
        base_message = messages.get(alert_level, "Unknown alert level.")
        # Add parameter-specific information
        param_values = processed_data.get('parameter_values', {})
        if param_values:
            # Find top 3 parameters with highest values
            sorted_params = sorted(param_values.items(), key=lambda x: x[1], reverse=True)[:3]
            if sorted_params:
                param_info = " Top parameters: "
                for param, value in sorted_params:
                    param_info += f"{param}={value:.2f}, "
                base_message += param_info.rstrip(', ')
        return base_message
    def _store_historical_data(self, processed_data: Dict[str, Any]):
        """Store processed data in historical database."""
        # Create data row
        row = {
            'timestamp': processed_data['timestamp'],
            'integrated_score': processed_data['integration'].get('integrated_score', 0.5),
            'confidence': processed_data['integration'].get('confidence', 0.5),
            'alert_level': processed_data['integration'].get('alert_level', 'normal')
        }
        # Add individual parameter values
        param_values = processed_data.get('parameter_values', {})
        for param_name, value in param_values.items():
            row[f'param_{param_name}'] = value
        # Add raw parameter values if available
        raw_params = processed_data.get('raw_parameters', {})
        for param_name, param_data in raw_params.items():
            if 'raw_value' in param_data:
                row[f'raw_{param_name}'] = param_data['raw_value']
            if 'uncertainty' in param_data:
                row[f'unc_{param_name}'] = param_data['uncertainty']
        # Convert to DataFrame and append
        new_row_df = pd.DataFrame([row])
        if self.historical_data.empty:
            self.historical_data = new_row_df
        else:
            self.historical_data = pd.concat(
                [self.historical_data, new_row_df],
                ignore_index=True
            )
        # Clean old data
        self._clean_old_data()
    def _clean_old_data(self):
        """Remove old data based on retention policy."""
        retention_days = self.config.get('data_retention_days', 30)
        if not self.historical_data.empty and 'timestamp' in self.historical_data.columns:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            # Convert timestamp column to datetime
            self.historical_data['timestamp'] = pd.to_datetime(self.historical_data['timestamp'])
            # Keep only recent data
            mask = self.historical_data['timestamp'] > cutoff_date
            self.historical_data = self.historical_data[mask].reset_index(drop=True)
            logger.debug(f"Historical data cleaned: {len(self.historical_data)} rows retained")
    def _store_alert(self, alert_data: Dict[str, Any]):
        """Store alert in separate alert log."""
        # In a real system, this would store to a database
        # For now, just keep in memory and log
        logger.info(f"Alert stored: {alert_data['alert_level']} at {alert_data['timestamp']}")
    def _notify_listeners(self, processed_data: Dict[str, Any]):
        """Notify registered listeners of new data."""
        # This would notify web sockets, callback functions, etc.
        # For now, just put in queue for external consumers
        self.data_queue.put(processed_data)
    def get_current_state(self) -> Dict[str, Any]:
        """Get current monitoring state."""
        state = {
            'is_monitoring': self.is_running,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'current_alert': self.alerts[-1] if self.alerts else None,
            'recent_data': self.current_state if self.current_state else {},
            'statistics': self._get_statistics()
        }
        return state
    def _get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        stats = {
            'total_data_points': len(self.historical_data) if not self.historical_data.empty else 0,
            'alerts_count': len(self.alerts),
            'parameters_monitored': 8,  # All 8 parameters
            'uptime_seconds': (datetime.now() - (self.last_update or datetime.now())).total_seconds() if self.last_update else 0,
            'data_quality': self._assess_data_quality()
        }
        if not self.historical_data.empty:
            stats.update({
                'data_start': self.historical_data['timestamp'].min().isoformat(),
                'data_end': self.historical_data['timestamp'].max().isoformat(),
                'avg_integrated_score': float(self.historical_data['integrated_score'].mean()),
                'max_integrated_score': float(self.historical_data['integrated_score'].max()),
                'min_integrated_score': float(self.historical_data['integrated_score'].min())
            })
        return stats
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess quality of monitoring data."""
        quality = {
            'overall': 'good',
            'parameter_coverage': 8,
            'missing_data': 0,
            'update_frequency': self.config['monitoring_interval']
        }
        if self.current_state:
            raw_params = self.current_state.get('raw_parameters', {})
            for param_name, param_data in raw_params.items():
                if param_data.get('source') == 'default':
                    quality['missing_data'] += 1
        if quality['missing_data'] > 2:
            quality['overall'] = 'poor'
        elif quality['missing_data'] > 0:
            quality['overall'] = 'moderate'
        return quality
    def get_historical_data(self, hours: int = 24) -> pd.DataFrame:
        """Get historical data for specified time window."""
        if self.historical_data.empty:
            return pd.DataFrame()
        # Filter by time window
        cutoff = datetime.now() - timedelta(hours=hours)
        mask = pd.to_datetime(self.historical_data['timestamp']) > cutoff
        return self.historical_data[mask].copy()
    def get_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alerts from specified time window."""
        if not self.alerts:
            return []
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_alerts = []
        for alert in reversed(self.alerts):  # Start from most recent
            alert_time = datetime.fromisoformat(alert['timestamp'].replace('Z', '+00:00'))
            if alert_time > cutoff:
                recent_alerts.append(alert)
            else:
                break
        return list(reversed(recent_alerts))  # Return in chronological order
    def reset_monitoring(self):
        """Reset monitoring state (for testing)."""
        self.historical_data = pd.DataFrame()
        self.alerts = []
        self.current_state = {}
        logger.info("Monitoring state reset")
class MonitoringDashboard:
    """
    Dashboard for visualizing monitoring data.
    """
    def __init__(self, monitor: RealTimeMonitor):
        self.monitor = monitor
        self.plots = {}
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report."""
        current_state = self.monitor.get_current_state()
        report = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_status': 'active' if self.monitor.is_running else 'inactive',
            'current_alert_level': current_state.get('current_alert', {}).get('alert_level', 'normal'),
            'integrated_score': current_state.get('recent_data', {}).get('integration', {}).get('integrated_score', 0.5),
            'confidence': current_state.get('recent_data', {}).get('integration', {}).get('confidence', 0.5),
            'parameter_status': self._get_parameter_status(current_state),
            'trends': current_state.get('recent_data', {}).get('trends', {}),
            'statistics': current_state.get('statistics', {}),
            'recommendations': self._generate_recommendations(current_state)
        }
        return report
    def _get_parameter_status(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get status of individual parameters."""
        status = {}
        recent_data = current_state.get('recent_data', {})
        raw_params = recent_data.get('raw_parameters', {})
        for param_name, param_data in raw_params.items():
            status[param_name] = {
                'value': param_data.get(f'{param_name}_index', 0.5),
                'uncertainty': param_data.get('uncertainty', 0.2),
                'source': param_data.get('source', 'unknown'),
                'quality': param_data.get('quality', 'unknown'),
                'status': 'normal' if param_data.get(f'{param_name}_index', 0.5) < 0.5 else 'elevated'
            }
        return status
    def _generate_recommendations(self, current_state: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on current state."""
        recommendations = []
        alert_level = current_state.get('current_alert', {}).get('alert_level', 'normal')
        integrated_score = current_state.get('recent_data', {}).get('integration', {}).get('integrated_score', 0.5)
        if alert_level == 'warning' or integrated_score > 0.7:
            recommendations = [
                "Immediate evacuation of high-risk zones",
                "Activate emergency response plan",
                "Continuous monitoring of all parameters",
                "Notify civil protection authorities",
                "Prepare for possible eruption within hours"
            ]
        elif alert_level == 'watch' or integrated_score > 0.5:
            recommendations = [
                "Increase monitoring frequency",
                "Restrict access to high-risk areas",
                "Review evacuation plans",
                "Alert emergency services",
                "Prepare monitoring equipment"
            ]
        elif alert_level == 'elevated' or integrated_score > 0.3:
            recommendations = [
                "Close monitoring of critical parameters",
                "Review alert protocols",
                "Check equipment status",
                "Update risk assessments",
                "Inform relevant authorities"
            ]
        else:  # normal
            recommendations = [
                "Continue routine monitoring",
                "Regular equipment maintenance",
                "Data quality assessment",
                "Review historical trends",
                "Update calibration if needed"
            ]
        return recommendations
    def export_report(self, format: str = 'json', filepath: Optional[str] = None) -> Any:
        """Export report in specified format."""
        report = self.generate_summary_report()
        if format == 'json':
            output = json.dumps(report, indent=2, default=str)
        elif format == 'csv':
            # Flatten report for CSV
            flat_data = self._flatten_report(report)
            df = pd.DataFrame([flat_data])
            output = df.to_csv(index=False)
        elif format == 'html':
            output = self._generate_html_report(report)
        else:
            output = str(report)
        if filepath:
            with open(filepath, 'w') as f:
                f.write(output)
            logger.info(f"Report exported to {filepath}")
        return output
    def _flatten_report(self, report: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """Flatten nested report dictionary."""
        flat = {}
        for key, value in report.items():
            full_key = f"{prefix}{key}"
            if isinstance(value, dict):
                flat.update(self._flatten_report(value, f"{full_key}_"))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        flat.update(self._flatten_report(item, f"{full_key}_{i}_"))
                    else:
                        flat[f"{full_key}_{i}"] = item
            else:
                flat[full_key] = value
        return flat
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Volcano Monitoring Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .alert-warning { background-color: #fff3cd; border-color: #ffeaa7; }
                .alert-danger { background-color: #f8d7da; border-color: #f5c6cb; }
                .parameter { display: inline-block; margin: 5px; padding: 5px 10px; background-color: #e9ecef; border-radius: 3px; }
                .value-high { color: #dc3545; font-weight: bold; }
                .value-medium { color: #fd7e14; }
                .value-low { color: #28a745; }
            </style>
        </head>
        <body>
        """
        # Header
        html += f"""
        <div class="header">
            <h1>🌋 Volcano Monitoring Report</h1>
            <p>Generated: {report.get('timestamp', 'Unknown')}</p>
            <p>Status: <strong>{report.get('monitoring_status', 'unknown')}</strong></p>
        </div>
        """
        # Alert Level
        alert_level = report.get('current_alert_level', 'normal')
        alert_class = 'alert-danger' if alert_level == 'warning' else 'alert-warning' if alert_level in ['watch', 'elevated'] else ''
        html += f"""
        <div class="section {alert_class}">
            <h2>Current Alert Level: {alert_level.upper()}</h2>
            <p>Integrated Score: <span class="value-high">{report.get('integrated_score', 0):.3f}</span></p>
            <p>Confidence: {report.get('confidence', 0):.1%}</p>
        </div>
        """
        # Parameters
        html += """
        <div class="section">
            <h2>Parameter Status</h2>
        """
        param_status = report.get('parameter_status', {})
        for param_name, status in param_status.items():
            value = status.get('value', 0.5)
            value_class = 'value-high' if value > 0.7 else 'value-medium' if value > 0.5 else 'value-low'
            html += f"""
            <div class="parameter">
                <strong>{param_name}</strong><br>
                Value: <span class="{value_class}">{value:.3f}</span><br>
                Uncertainty: {status.get('uncertainty', 0):.3f}<br>
                Status: {status.get('status', 'unknown')}
            </div>
            """
        html += "</div>"
        # Recommendations
        html += """
        <div class="section">
            <h2>Recommendations</h2>
            <ul>
        """
        recommendations = report.get('recommendations', [])
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        html += """
            </ul>
        </div>
        """
        # Statistics
        html += """
        <div class="section">
            <h2>Statistics</h2>
        """
        stats = report.get('statistics', {})
        for key, value in stats.items():
            if isinstance(value, dict):
                html += f"<p><strong>{key}:</strong></p><ul>"
                for subkey, subvalue in value.items():
                    html += f"<li>{subkey}: {subvalue}</li>"
                html += "</ul>"
            else:
                html += f"<p><strong>{key}:</strong> {value}</p>"
        html += """
        </div>
        </body>
        </html>
        """
        return html
# Utility function for quick monitoring setup
def start_monitoring(config_file: Optional[str] = None) -> RealTimeMonitor:
    """
    Quick function to start monitoring.
    Args:
        config_file: Optional configuration file
    Returns:
        RealTimeMonitor instance
    """
    # Load configuration if provided
    config = {}
    if config_file:
        try:
            with open(config_file, 'r') as f:
                import json
                config = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
    # Create and start monitor
    monitor = RealTimeMonitor(config)
    monitor.start_monitoring()
    return monitor