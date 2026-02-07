
```
# Seismo Framework - Quick Start Guide 🌋

## 🚀 Installation

### Basic Installation (Termux Compatible)
```bash
# Install core dependencies
pip install numpy pandas python-dateutil pytz tzlocal pyyaml

# Install Seismo Framework from current directory
pip install -e .

# Or install from Git
pip install git+https://gitlab.com/gitdeeper3/seismo.git
```

Complete Installation (with Visualization)

```bash
# For systems with full Python support
pip install seismo-framework[full]

# Or install all dependencies manually
pip install numpy pandas matplotlib seaborn plotly scipy scikit-learn
```

📋 Basic Usage

1. Import and Initialize

```python
from seismo_framework import SeismicMonitor, about, citation

# Display framework information
about()

# Show citation information
citation()
```

2. Create a Monitoring Instance

```python
# Simple monitor
monitor = SeismicMonitor(region='san_andreas')

# Monitor with custom configuration
monitor = SeismicMonitor(
    region='your_volcano',
    config_file='config/observatory.yaml'  # Optional
)
```

3. Analyze Individual Parameters

```python
from seismo_framework.core.parameters import SeismicAnalyzer

# Initialize analyzer
seismic = SeismicAnalyzer()

# Analyze seismic data
seismic_data = {
    'events': [
        {'magnitude': 3.5, 'time': '2024-01-01T00:00:00', 'latitude': 40.5, 'longitude': 15.5},
        {'magnitude': 4.0, 'time': '2024-01-01T01:00:00', 'latitude': 40.6, 'longitude': 15.6},
    ]
}

result = seismic.analyze(seismic_data)
print(f"Seismic Index: {result['seismic_index']:.3f}")
print(f"Event Count: {result['event_count']}")
print(f"Maximum Magnitude: {result['max_magnitude']:.1f}")
```

4. Integrate Multiple Parameters

```python
from seismo_framework.core.integration import ParameterIntegrator

# Initialize integrator
integrator = ParameterIntegrator()

# Provide parameter values (0-1 scale)
parameters = {
    'seismic': 0.75,          # High seismic activity
    'deformation': 0.60,      # Moderate deformation
    'hydrogeological': 0.45,  # Normal hydro conditions
    'electrical': 0.80,       # High electrical signals
    'magnetic': 0.55,         # Moderate magnetic anomalies
    'instability': 0.65,      # Elevated instability
    'stress': 0.70,           # High stress accumulation
    'rock_properties': 0.50   # Normal rock properties
}

# Integrate parameters
result = integrator.integrate(parameters)

print(f"Integrated Score: {result['integrated_score']:.3f}")
print(f"Alert Level: {result['alert_level'].upper()}")
print(f"Confidence: {result['confidence']:.1%}")

# Display parameter contributions
for param, contrib in result['contributions'].items():
    print(f"{param}: weight={contrib['weight']:.3f}, value={contrib['value']:.3f}")
```

🔧 Practical Examples

Example 1: Real-time Monitoring Simulation

```bash
# Run the monitoring simulation
python start_monitoring.py
```

Example 2: Complete Data Analysis Pipeline

```bash
# Run the practical example
python practical_example.py
```

Example 3: Generate Data Report

```bash
# Run the data analysis and reporting
python run_seismo.py
```

Example 4: Test Basic Functionality

```bash
# Test framework components
python test_basic_functionality.py
```

📊 Command Line Interface

Start Monitoring Service

```bash
seismo-monitor --region san_andreas --config config/monitoring.yaml
```

Analyze Historical Data

```bash
seismo-analyze --input data/earthquakes.csv --output reports/analysis.pdf
```

Generate Dashboard

```bash
seismo-dashboard --port 8050 --live-update
```

🎯 Core Features

8-Parameter Monitoring System

```
1. Seismic Activity (S)        - Earthquake detection and analysis
2. Crustal Deformation (D)     - GPS and InSAR measurements
3. Hydrogeological (W)         - Water chemistry and gas emissions
4. Electrical Signals (E)      - Resistivity and SP measurements
5. Magnetic Anomalies (M)      - Magnetic field variations
6. Instability Indicators (L)  - Slope stability and landslides
7. Tectonic Stress (T)         - Stress accumulation analysis
8. Rock Properties (R)         - Rock strength and fracturing
```

Alert Levels

· NORMAL (0.0-0.3): No significant activity
· ELEVATED (0.3-0.5): Minor changes detected
· WATCH (0.5-0.7): Significant activity increase
· WARNING (0.7-1.0): Critical activity, immediate action required

📁 Project Structure

```
Seismo/
├── seismo_framework/          # Main package
│   ├── core/                 # Core scientific modules
│   │   ├── parameters/       # 8 parameter analyzers
│   │   ├── integration/      # Multi-parameter fusion
│   │   ├── monitoring/       # Real-time monitoring
│   │   └── utils/           # Utilities and helpers
│   └── __init__.py          # Package initialization
├── examples/                 # Usage examples
├── tests/                   # Test suite
├── config/                  # Configuration files
├── requirements.txt         # Python dependencies
└── pyproject.toml          # Build configuration
```

🔬 Scientific Configuration

Default Parameter Weights

```python
weights = {
    'seismic': 0.20,          # 20% weight
    'deformation': 0.15,      # 15% weight
    'hydrogeological': 0.12,  # 12% weight
    'electrical': 0.10,       # 10% weight
    'magnetic': 0.10,         # 10% weight
    'instability': 0.15,      # 15% weight
    'stress': 0.10,          # 10% weight
    'rock_properties': 0.08,  # 8% weight
}
```

Custom Configuration

Create config/monitoring.yaml:

```yaml
monitoring:
  interval: 60  # seconds
  retention_days: 30
  alert_threshold: 0.7

parameters:
  seismic:
    min_magnitude: 2.0
    max_magnitude: 8.0
  deformation:
    critical_rate: 10.0  # mm/year
```

📊 Output and Reporting

Generated Files

· CSV Files: Timestamped parameter data
· Text Reports: Analysis summaries
· Alert Logs: Notification records
· Configuration Files: System settings

Sample Report Structure

```
SEISMO FRAMEWORK - MONITORING REPORT
====================================

TIMESTAMP: 2024-01-15T14:30:00
LOCATION: Mount Example (45.0°N, 10.0°E)

CURRENT STATUS:
  Integrated Score: 0.436
  Alert Level: ELEVATED
  Confidence: 75.2%

PARAMETER STATUS:
  seismic:          0.500 [NORMAL]
  deformation:      0.726 [HIGH]
  hydrogeological:  0.400 [NORMAL]
  electrical:       0.300 [NORMAL]
  magnetic:         0.600 [MEDIUM]
  instability:      0.500 [MEDIUM]
  stress:           0.400 [NORMAL]
  rock_properties:  0.300 [NORMAL]

RECOMMENDATIONS:
  • Enhanced vigilance recommended
  • Close monitoring of deformation parameter
  • Regular equipment checks
  • Update risk assessments
```

🔧 Troubleshooting

Common Issues and Solutions

1. Import Errors

```bash
# If you see "No module named 'numpy'"
pip install numpy pandas

# If you see "No module named 'scipy'"
# On Termux:
pkg install python-scipy
# Or skip visualization features
```

2. Memory Issues

```python
# Reduce data retention
monitor = SeismicMonitor(config={'data_retention_days': 7})

# Process data in chunks
from seismo_framework.core.utils import TimeSeriesProcessor
processor = TimeSeriesProcessor()
resampled_data = processor.resample_time_series(data, freq='1H')
```

3. Performance Optimization

```python
# Use minimal configuration for mobile devices
config = {
    'monitoring_interval': 300,  # 5 minutes
    'data_retention_days': 7,
    'enable_visualization': False
}
monitor = SeismicMonitor(config=config)
```

Debug Mode

```bash
# Enable debug logging
export SEISMO_LOG_LEVEL=DEBUG

# Or in Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

🚀 Deployment Options

1. Local Development

```bash
# Clone and install
git clone https://gitlab.com/gitdeeper3/seismo.git
cd seismo
pip install -e .
python examples/practical_example.py
```

2. Docker Deployment

```bash
# Build and run with Docker
docker build -t seismo-framework .
docker run -p 8050:8050 seismo-framework
```

3. Server Deployment

```bash
# Install as system service
sudo cp systemd/seismo.service /etc/systemd/system/
sudo systemctl enable seismo
sudo systemctl start seismo
```

📞 Support and Resources

Documentation

· Online Docs: https://seismo.netlify.app/documentation
· API Reference: https://seismo.netlify.app/api
· Examples: examples/ directory

Community and Support

· GitLab Issues: https://gitlab.com/gitdeeper3/seismo/-/issues
· Email: gitdeeper@gmail.com
· Research Contact: +1 (714) 264-2074

Training and Workshops

· Basic Usage: Run through examples in this guide
· Advanced Topics: Custom parameter development
· Integration: Connecting to seismic networks

🔬 Scientific Validation

Testing Methodology

```python
# Test with historical data
from seismo_framework.core.utils import PerformanceMetrics

# Load test data
historical_data = pd.read_csv('data/historical_earthquakes.csv')

# Calculate performance metrics
metrics = PerformanceMetrics.calculate_metrics(
    predictions=model_predictions,
    actuals=actual_occurrences,
    threshold=0.5
)

print(f"Accuracy: {metrics['accuracy']:.1%}")
print(f"Sensitivity: {metrics['sensitivity']:.1%}")
print(f"Specificity: {metrics['specificity']:.1%}")
```

Citation

```python
# Display citation information
from seismo_framework import citation
citation()
```

📚 Additional Resources

Sample Datasets

· Example Data: data/sample/ directory
· Test Scenarios: tests/test_data/
· Configuration Templates: config/templates/

Integration Guides

· Seismic Networks: SEED, QuakeML formats
· GPS Data: RINEX format support
· Environmental Sensors: Custom adapter development

Advanced Topics

· Custom parameter weight optimization
· Machine learning integration
· Real-time dashboard development
· Multi-station network support

🎯 Getting Help

Quick Checklist

· Dependencies installed (numpy, pandas)
· Framework imported successfully
· Sample data loaded correctly
· Parameters configured appropriately
· Output files generated

Common Questions

1. Q: How do I add a new parameter?
   A: Create a new analyzer in core/parameters/ and update weights
2. Q: Can I use my own data format?
   A: Yes, implement a custom data loader in data/loaders/
3. Q: How do I deploy in production?
   A: Use Docker or systemd service as shown above

📄 License and Citation

This software is released under the MIT License. For scientific use, please cite:

```bibtex
@software{baladi2026seismo,
  author = {Baladi, Samir},
  title = {Seismo: Real-Time Earthquake Monitoring Through Multi-Parameter Geophysical Integration},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.14063164},
  url = {https://doi.org/10.5281/zenodo.14063164}
}
```

---

Need more help? Contact: gitdeeper@gmail.com
Documentation: https://seismo.netlify.app
Source Code: https://gitlab.com/gitdeeper3/seismo

Last Updated: 2026-02-07 | Version: 1.0.0

``` 

## 📁 Organized Project Structure

After running the organization script, your project will have this structure:

```

seismo_framework/
├── data/                    # Data management
│   ├── raw/                # Raw sensor data
│   ├── processed/          # Processed/cleaned data
│   ├── exports/            # Data exports (CSV, JSON)
│   └── samples/            # Sample datasets for testing
├── reports/                # Report management
│   ├── daily/              # Daily monitoring reports
│   ├── weekly/             # Weekly summary reports
│   ├── monthly/            # Monthly analysis reports
│   ├── alerts/             # Alert/notification reports
│   └── archived/           # Archived reports (>90 days)
├── logs/                   # System and application logs
├── config/                 # Configuration files
│   ├── regions/            # Region-specific configurations
│   ├── templates/          # Report templates
│   └── calibrations/       # Sensor calibration files
└── core/                   # Core framework code
├── parameters/         # 8 parameter analyzers
├── integration/        # Multi-parameter fusion
├── monitoring/         # Real-time monitoring
└── utils/             # Utilities and helpers

```

### Organize Your Project
```bash
# Run the organization script
python organize_project.py

# Or use the report manager
python -c "from seismo_framework.reports.manager import organize_project_reports; organize_project_reports()"
```

Access Organized Files

```python
# Load configuration
import yaml
with open('seismo_framework/config/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Save report to organized location
report_path = 'seismo_framework/reports/daily/report_20240101.txt'
with open(report_path, 'w') as f:
    f.write("Your report content")

# Load sample data
import pandas as pd
sample_data = pd.read_csv('seismo_framework/data/samples/sample_earthquakes.csv')
```

Report Management

```python
from seismo_framework.reports.manager import ReportManager

# Initialize manager
manager = ReportManager(base_dir="seismo_framework/reports")

# Organize existing reports
manager.organize_existing_reports(".")

# Clean up old reports
manager.cleanup_old_reports(days_to_keep=90)

# Generate HTML index
index_path = manager.generate_index()
print(f"Index generated: {index_path}")

# Get report summary
summary = manager.get_report_summary()
print(f"Total reports: {summary['total_reports']}")
print(f"Total size: {summary['total_size_mb']} MB")
```

