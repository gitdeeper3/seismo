"""
Organize Seismo Framework project structure.
"""

import os
import shutil
from datetime import datetime

print("🔧 Organizing Seismo Framework Project...")
print("=" * 50)

# Create directory structure
directories = [
    'seismo_framework/data/raw',
    'seismo_framework/data/processed',
    'seismo_framework/data/exports',
    'seismo_framework/data/samples',
    'seismo_framework/reports/daily',
    'seismo_framework/reports/weekly',
    'seismo_framework/reports/monthly',
    'seismo_framework/reports/alerts',
    'seismo_framework/reports/archived',
    'seismo_framework/logs',
    'seismo_framework/config/regions',
    'seismo_framework/config/templates',
    'seismo_framework/config/calibrations',
]

print("\n📁 Creating directory structure...")
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"  ✅ Created: {directory}")

# Move existing report files
print("\n📄 Moving existing reports...")
report_files = []

# Find report files in current directory
for file in os.listdir('.'):
    if file.endswith(('.txt', '.csv', '.json', '.html', '.pdf')):
        if any(pattern in file.lower() for pattern in ['report', 'seismo', 'data_']):
            report_files.append(file)

for report_file in report_files:
    try:
        # Determine destination based on filename
        if 'alert' in report_file.lower():
            dest_dir = 'seismo_framework/reports/alerts'
        else:
            dest_dir = 'seismo_framework/reports/daily'
        
        dest_path = os.path.join(dest_dir, report_file)
        shutil.move(report_file, dest_path)
        print(f"  ✅ Moved: {report_file} -> {dest_dir}/")
    except Exception as e:
        print(f"  ⚠️  Could not move {report_file}: {e}")

# Create sample data files
print("\n📊 Creating sample data...")
sample_data = """timestamp,seismic,deformation,hydrogeological,electrical,magnetic,instability,stress,rock_properties
2024-01-01T00:00:00,0.3,0.2,0.4,0.1,0.3,0.2,0.3,0.5
2024-01-01T01:00:00,0.4,0.3,0.5,0.2,0.4,0.3,0.4,0.6
2024-01-01T02:00:00,0.5,0.4,0.6,0.3,0.5,0.4,0.5,0.7
2024-01-01T03:00:00,0.6,0.5,0.7,0.4,0.6,0.5,0.6,0.8
"""

sample_path = 'seismo_framework/data/samples/sample_earthquakes.csv'
with open(sample_path, 'w') as f:
    f.write(sample_data)
print(f"  ✅ Created: {sample_path}")

# Create log file
print("\n📝 Setting up logging...")
log_content = f"""Seismo Framework Log
===================
Start Time: {datetime.now().isoformat()}
Version: 1.0.0
Status: Initialized
"""
log_path = 'seismo_framework/logs/seismo.log'
with open(log_path, 'w') as f:
    f.write(log_content)
print(f"  ✅ Created: {log_path}")

# Create README for organized structure
print("\n📖 Creating documentation...")
readme_content = """# Seismo Framework - Organized Structure

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

Last Organized: {timestamp}
""".format(timestamp=datetime.now().isoformat())

readme_path = 'seismo_framework/ORGANIZED_STRUCTURE.md'
with open(readme_path, 'w') as f:
    f.write(readme_content)
print(f"  ✅ Created: {readme_path}")

print("\n" + "=" * 50)
print("✅ Project Organization Complete!")
print("\n📊 Summary:")
print(f"  Directories created: {len(directories)}")
print(f"  Reports moved: {len(report_files)}")
print(f"  Sample files created: 1")
print(f"  Documentation created: 1")
print("\n🚀 Next steps:")
print("  1. Review the organized structure")
print("  2. Update your code to use new paths")
print("  3. Run your applications")
print("\n📁 Organized structure ready in: seismo_framework/")
