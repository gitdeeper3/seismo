"""
Visualization tools for monitoring data.
Matplotlib is optional - system works without it.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Check if matplotlib is available
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.info("Matplotlib not available. Visualization limited to text reports.")

class MonitoringVisualizer:
    """
    Visualization tools for monitoring data.
    Works with or without matplotlib.
    """
    
    def __init__(self, style: str = 'default'):
        self.style = style
        if MATPLOTLIB_AVAILABLE:
            self._setup_plotting_style()
    
    def _setup_plotting_style(self):
        """Setup plotting style if matplotlib is available."""
        try:
            if self.style == 'seaborn':
                try:
                    import seaborn as sns
                    sns.set_style("whitegrid")
                    sns.set_palette("husl")
                except ImportError:
                    plt.style.use('default')
            else:
                plt.style.use('default')
                
            plt.rcParams['figure.figsize'] = [12, 8]
            plt.rcParams['font.size'] = 10
            
        except Exception as e:
            logger.warning(f"Error setting up plotting style: {e}")
    
    def plot_parameter_timeseries(self, historical_data: pd.DataFrame,
                                 parameter_names: List[str],
                                 hours: int = 24,
                                 save_path: Optional[str] = None) -> bool:
        """
        Plot time series of parameters.
        
        Returns:
            bool: True if plot was created successfully
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return False
        
        try:
            if historical_data.empty:
                logger.warning("No historical data to plot")
                return False
            
            # Filter by time window
            cutoff = datetime.now() - timedelta(hours=hours)
            mask = pd.to_datetime(historical_data['timestamp']) > cutoff
            plot_data = historical_data[mask].copy()
            
            if plot_data.empty:
                logger.warning("No data in specified time window")
                return False
            
            # Create figure
            n_params = len(parameter_names)
            fig, axes = plt.subplots(n_params, 1, figsize=(14, 3*n_params), sharex=True)
            
            if n_params == 1:
                axes = [axes]
            
            # Plot each parameter
            for idx, param_name in enumerate(parameter_names):
                ax = axes[idx]
                
                # Get parameter column
                param_col = f'param_{param_name}'
                if param_col not in plot_data.columns:
                    logger.warning(f"Parameter {param_name} not found in data")
                    continue
                
                # Convert timestamps
                timestamps = pd.to_datetime(plot_data['timestamp'])
                
                # Plot parameter values
                ax.plot(timestamps, plot_data[param_col], 
                       label=param_name, linewidth=2, alpha=0.8)
                
                # Formatting
                ax.set_ylabel(param_name.replace('_', ' ').title(), fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right')
                
                # Add threshold lines
                ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Warning')
                ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Watch')
                ax.axhline(y=0.3, color='yellow', linestyle='--', alpha=0.5, label='Elevated')
            
            # Format x-axis
            axes[-1].set_xlabel('Time', fontsize=11)
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=max(1, hours//12)))
            
            plt.suptitle(f'Parameter Time Series (Last {hours} hours)', fontsize=14, y=0.98)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Figure saved to {save_path}")
                plt.close()
                return True
            else:
                plt.show()
                return True
            
        except Exception as e:
            logger.error(f"Error creating plot: {e}")
            return False
    
    def plot_integrated_score(self, historical_data: pd.DataFrame,
                             hours: int = 24,
                             save_path: Optional[str] = None) -> bool:
        """
        Plot integrated score over time.
        
        Returns:
            bool: True if plot was created successfully
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return False
        
        try:
            if historical_data.empty:
                logger.warning("No historical data to plot")
                return False
            
            # Filter by time window
            cutoff = datetime.now() - timedelta(hours=hours)
            mask = pd.to_datetime(historical_data['timestamp']) > cutoff
            plot_data = historical_data[mask].copy()
            
            if plot_data.empty:
                logger.warning("No data in specified time window")
                return False
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # Convert timestamps
            timestamps = pd.to_datetime(plot_data['timestamp'])
            
            # Top plot: Integrated score
            ax1 = axes[0]
            
            # Plot integrated score
            ax1.plot(timestamps, plot_data['integrated_score'], 
                    label='Integrated Score', linewidth=3, color='darkblue', alpha=0.8)
            
            # Add alert level regions
            ax1.axhspan(0.7, 1.0, alpha=0.1, color='red', label='Warning Zone')
            ax1.axhspan(0.5, 0.7, alpha=0.1, color='orange', label='Watch Zone')
            ax1.axhspan(0.3, 0.5, alpha=0.1, color='yellow', label='Elevated Zone')
            ax1.axhspan(0.0, 0.3, alpha=0.1, color='green', label='Normal Zone')
            
            # Formatting
            ax1.set_ylabel('Integrated Score', fontsize=12)
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper right')
            ax1.set_title(f'Integrated Volcanic Activity Score (Last {hours} hours)', fontsize=14)
            
            # Bottom plot: Alert level
            ax2 = axes[1]
            
            # Convert alert levels to numeric for plotting
            alert_levels = plot_data['alert_level'].map({
                'normal': 0,
                'elevated': 1,
                'watch': 2,
                'warning': 3
            })
            
            # Create stepped plot for alert levels
            ax2.step(timestamps, alert_levels, where='post', 
                    linewidth=2, color='darkred', alpha=0.8)
            
            # Formatting
            ax2.set_yticks([0, 1, 2, 3])
            ax2.set_yticklabels(['Normal', 'Elevated', 'Watch', 'Warning'])
            ax2.set_ylabel('Alert Level', fontsize=12)
            ax2.set_ylim(-0.5, 3.5)
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, hours//6)))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Figure saved to {save_path}")
                plt.close()
                return True
            else:
                plt.show()
                return True
            
        except Exception as e:
            logger.error(f"Error creating plot: {e}")
            return False
    
    def generate_text_report(self, historical_data: pd.DataFrame,
                            current_state: Dict[str, Any]) -> str:
        """
        Generate text report of monitoring data.
        Works without matplotlib.
        """
        report = []
        report.append("=" * 60)
        report.append("SEISMO FRAMEWORK - MONITORING REPORT")
        report.append("=" * 60)
        
        # Add timestamp
        report.append(f"Report generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Add current status
        if current_state:
            report.append("CURRENT STATUS:")
            report.append(f"  Monitoring: {'ACTIVE' if current_state.get('is_monitoring') else 'INACTIVE'}")
            report.append(f"  Last update: {current_state.get('last_update', 'N/A')}")
            
            current_alert = current_state.get('current_alert', {})
            if current_alert:
                report.append(f"  Current alert: {current_alert.get('alert_level', 'normal').upper()}")
                report.append(f"  Alert message: {current_alert.get('message', '')}")
            report.append("")
        
        # Add data summary
        if not historical_data.empty:
            report.append("DATA SUMMARY:")
            report.append(f"  Data points: {len(historical_data)}")
            
            # Convert timestamps
            if 'timestamp' in historical_data.columns:
                try:
                    timestamps = pd.to_datetime(historical_data['timestamp'])
                    report.append(f"  Time range: {timestamps.min()} to {timestamps.max()}")
                except:
                    report.append("  Time range: Could not parse timestamps")
            
            if 'integrated_score' in historical_data.columns:
                avg_score = historical_data['integrated_score'].mean()
                max_score = historical_data['integrated_score'].max()
                report.append(f"  Average integrated score: {avg_score:.3f}")
                report.append(f"  Maximum integrated score: {max_score:.3f}")
            report.append("")
        
        # Add parameter summary
        param_cols = [col for col in historical_data.columns 
                     if col.startswith('param_')]
        if param_cols:
            report.append("PARAMETER SUMMARY:")
            for param_col in param_cols:
                param_name = param_col.replace('param_', '')
                if param_col in historical_data.columns:
                    values = historical_data[param_col].dropna()
                    if len(values) > 0:
                        mean_val = values.mean()
                        max_val = values.max()
                        status = "⚠️ HIGH" if max_val > 0.7 else "⚠️ MEDIUM" if max_val > 0.5 else "✅ NORMAL"
                        report.append(f"  {param_name}: {status} (mean={mean_val:.3f}, max={max_val:.3f})")
            report.append("")
        
        # Add alert history
        if 'alert_level' in historical_data.columns:
            alert_counts = historical_data['alert_level'].value_counts()
            report.append("ALERT HISTORY:")
            for level, count in alert_counts.items():
                report.append(f"  {level.upper()}: {count} occurrences")
        
        # Add recommendations
        report.append("")
        report.append("RECOMMENDATIONS:")
        if not historical_data.empty and 'integrated_score' in historical_data.columns:
            latest_score = historical_data['integrated_score'].iloc[-1] if len(historical_data) > 0 else 0.5
            
            if latest_score > 0.7:
                report.append("  1. ⚠️ WARNING: High risk detected")
                report.append("  2. Increase monitoring frequency")
                report.append("  3. Alert emergency services")
                report.append("  4. Prepare evacuation plans")
            elif latest_score > 0.5:
                report.append("  1. ⚠️ Elevated activity detected")
                report.append("  2. Close monitoring recommended")
                report.append("  3. Review emergency protocols")
                report.append("  4. Update risk assessments")
            else:
                report.append("  1. ✅ Normal activity levels")
                report.append("  2. Continue routine monitoring")
                report.append("  3. Regular equipment checks")
                report.append("  4. Data quality verification")
        
        report.append("=" * 60)
        
        return '\n'.join(report)
    
    def save_report(self, historical_data: pd.DataFrame,
                   current_state: Dict[str, Any],
                   filepath: str) -> bool:
        """
        Save monitoring report to file.
        Works without matplotlib.
        """
        try:
            report = self.generate_text_report(historical_data, current_state)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"Report saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return False
    
    def generate_csv_report(self, historical_data: pd.DataFrame,
                           filepath: str) -> bool:
        """
        Generate CSV report of monitoring data.
        Works without matplotlib.
        """
        try:
            if historical_data.empty:
                logger.warning("No data to export")
                return False
            
            # Save to CSV
            historical_data.to_csv(filepath, index=False)
            logger.info(f"CSV report saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving CSV report: {e}")
            return False
    
    def plot_ascii_chart(self, data: List[float], width: int = 50, height: int = 20) -> str:
        """
        Create ASCII art chart for terminal display.
        Works without matplotlib.
        """
        if not data:
            return "No data for chart"
        
        # Normalize data to chart height
        min_val = min(data)
        max_val = max(data)
        
        if max_val == min_val:
            # All values are the same
            normalized = [height // 2] * len(data)
        else:
            normalized = [int((val - min_val) / (max_val - min_val) * (height - 1)) 
                         for val in data]
        
        # Create chart
        chart = []
        
        # Add top border
        chart.append("┌" + "─" * (width + 2) + "┐")
        
        # Add chart rows
        for row in range(height - 1, -1, -1):
            line = "│ "
            for col in range(width):
                if col < len(normalized):
                    if normalized[col] == row:
                        line += "●"
                    elif normalized[col] > row:
                        line += "│"
                    else:
                        line += " "
                else:
                    line += " "
            line += " │"
            chart.append(line)
        
        # Add bottom border
        chart.append("└" + "─" * (width + 2) + "┘")
        
        # Add labels
        if len(data) > 1:
            labels = f"Min: {min_val:.2f} | Max: {max_val:.2f} | Points: {len(data)}"
            chart.append(labels.center(width + 4))
        
        return '\n'.join(chart)

def create_monitoring_report(historical_data: pd.DataFrame,
                            current_state: Dict[str, Any],
                            output_dir: str = './reports'):
    """
    Create comprehensive monitoring report.
    Works with or without matplotlib.
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Initialize visualizer
    visualizer = MonitoringVisualizer()
    
    # Generate text report
    text_report_path = os.path.join(output_dir, f'report_{timestamp}.txt')
    text_success = visualizer.save_report(historical_data, current_state, text_report_path)
    
    # Generate CSV report
    csv_report_path = os.path.join(output_dir, f'data_{timestamp}.csv')
    csv_success = visualizer.generate_csv_report(historical_data, csv_report_path)
    
    # Try to create plots if matplotlib is available
    plots_created = []
    
    if MATPLOTLIB_AVAILABLE:
        # Try to create integrated score plot
        try:
            plot_path = os.path.join(output_dir, f'integrated_score_{timestamp}.png')
            if visualizer.plot_integrated_score(historical_data, hours=24, save_path=plot_path):
                plots_created.append('integrated_score')
        except Exception as e:
            logger.warning(f"Could not create integrated score plot: {e}")
    
    logger.info(f"Monitoring report created in {output_dir}")
    logger.info(f"Text report: {text_report_path}")
    logger.info(f"CSV data: {csv_report_path}")
    if plots_created:
        logger.info(f"Plots created: {', '.join(plots_created)}")
    else:
        logger.info("No plots created (matplotlib not available or error)")
    
    return text_success or csv_success
