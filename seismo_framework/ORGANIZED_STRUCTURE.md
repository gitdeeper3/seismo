# Seismo Framework - Organized Structure

## 📁 Directory Structure

```

seismo_framework/
├── data/                    # Data management
│   ├── raw/                # Raw sensor data
│   ├── processed/          # Processed data
│   ├── exports/            # Data exports
│   └── samples/            # Sample datasets
├── reports/                # Report management
│   ├── daily/              # Daily reports
│   ├── weekly/             # Weekly reports
│   ├── monthly/            # Monthly reports
│   ├── alerts/             # Alert reports
│   └── archived/           # Archived reports
├── logs/                   # Log files
├── config/                 # Configuration
│   ├── regions/            # Region-specific configs
│   ├── templates/          # Report templates
│   └── calibrations/       # Calibration files
└── core/                   # Core framework (unchanged)

```

## 🔧 Usage

### Organize Reports
```python
from seismo_framework.reports.manager import organize_project_reports
organize_project_reports()
```

Use Organized Structure

```python
# Access organized data
import pandas as pd

# Load sample data
data = pd.read_csv('seismo_framework/data/samples/sample_earthquakes.csv')

# Save new report
report_path = 'seismo_framework/reports/daily/report_20240101.txt'
```

Configuration

Configuration files are in seismo_framework/config/:

· default.yaml: Default settings
· termux.yaml: Termux-optimized settings
· operational.yaml: Production settings

🚀 Quick Start

1. Run organization script:
   ```bash
   python organize_project.py
   ```
2. Use the organized structure:
   ```python
   from seismo_framework import SeismicMonitor
   monitor = SeismicMonitor()
   ```

📞 Support

For issues with the organized structure, check:

1. File permissions
2. Disk space
3. Path configurations

Last Organized: 2026-02-07T10:43:16.506326
