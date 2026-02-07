# Installation Guide

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Storage**: 50 GB free space
- **OS**: Linux, macOS, or Windows 10/11
- **Python**: 3.9 or higher

### Recommended Requirements
- **CPU**: 8+ cores
- **RAM**: 16+ GB
- **Storage**: 100+ GB (for data storage)
- **GPU**: NVIDIA CUDA compatible (optional, for ML features)

## Installation Methods

### Method 1: PyPI Installation (Recommended)
```bash
# Install from PyPI
pip install seismo-framework

# Install with optional dependencies
pip install seismo-framework[full]

# Install for development
pip install seismo-framework[dev]
```

Method 2: Source Installation

```bash
# Clone repository
git clone https://gitlab.com/gitdeeper3/seismo.git
cd seismo

# Install in development mode
pip install -e .

# Install with all features
pip install -e .[full,dev,test]
```

Method 3: Docker Installation

```bash
# Pull Docker image
docker pull gitdeeper3/seismo:latest

# Run container
docker run -d \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  --name seismo-monitoring \
  gitdeeper3/seismo:latest
```

Platform-Specific Instructions

Linux (Ubuntu/Debian)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3-pip \
    python3-venv \
    build-essential \
    libhdf5-dev \
    libnetcdf-dev

# Install Seismo
pip3 install seismo-framework[full]
```

macOS

```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python hdf5 netcdf

# Install Seismo
pip3 install seismo-framework[full]
```

Windows

```powershell
# Install Python from python.org
# Make sure to check "Add Python to PATH"

# Open PowerShell as Administrator
# Install Seismo
pip install seismo-framework

# If you encounter errors, try:
pip install --user seismo-framework
```

Configuration

Initial Setup

```bash
# Generate default configuration
seismo-config init --region global

# Configure database
seismo-config database --type postgresql --host localhost --port 5432

# Setup monitoring stations
seismo-config stations import stations.csv
```

Environment Variables

```bash
# Copy example environment file
cp config/.env.example config/.env

# Edit environment variables
nano config/.env
```

Example .env file:

```ini
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=seismo
DB_USER=seismo_user
DB_PASSWORD=your_password

# API Keys (optional)
USGS_API_KEY=your_usgs_key
GNSS_API_KEY=your_gnss_key

# Monitoring Settings
MONITORING_INTERVAL=60
ALERT_THRESHOLD=0.7
LOG_LEVEL=INFO
```

Verification

Test Installation

```bash
# Check installation
python -c "import seismo; print(seismo.__version__)"

# Run basic test
python -m pytest tests/test_basic.py -v

# Start test server
python -m seismo.monitoring.dashboard --test-mode
```

Verify Dependencies

```bash
# Check all dependencies
python -c "import seismo; seismo.check_dependencies()"

# Output should show:
# ✓ numpy 1.24.0
# ✓ scipy 1.10.0
# ✓ pandas 2.0.0
# ... all dependencies OK
```

Troubleshooting

Common Issues

1. Permission Errors

```bash
# Fix: Use virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install seismo-framework
```

2. Missing Dependencies

```bash
# On Ubuntu/Debian
sudo apt install python3-dev build-essential

# On macOS
brew install gcc
```

3. HDF5/NetCDF Issues

```bash
# Install system libraries
sudo apt install libhdf5-dev libnetcdf-dev  # Ubuntu/Debian
brew install hdf5 netcdf                    # macOS
```

4. Memory Issues

```bash
# Reduce memory usage
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Or configure in Python
import os
os.environ['OMP_NUM_THREADS'] = '1'
```

Getting Help

```bash
# Check documentation
seismo --help

# View logs
tail -f logs/installation.log

# Report issues
seismo bug-report
```

Updates

Update Procedure

```bash
# Update from PyPI
pip install --upgrade seismo-framework

# Update from source
cd seismo
git pull origin main
pip install -e . --upgrade

# Update Docker container
docker pull gitdeeper3/seismo:latest
docker-compose down
docker-compose up -d
```

Version Compatibility

· v0.1.x: Python 3.9+
· v0.2.x: Python 3.10+ (planned)
· Check pyproject.toml for exact requirements

Uninstallation

Complete Removal

```bash
# Uninstall package
pip uninstall seismo-framework

# Remove configuration files
rm -rf ~/.config/seismo
rm -rf ~/.cache/seismo

# Remove Docker containers
docker stop seismo-monitoring
docker rm seismo-monitoring
docker rmi gitdeeper3/seismo:latest
```

Partial Removal (keep data)

```bash
# Keep data but remove application
pip uninstall seismo-framework

# Data remains in:
# ~/.local/share/seismo/data/
# /app/data/ (Docker volumes)
```

Support

For installation issues:

· Email: gitdeeper@gmail.com
· Documentation: https://seismo.netlify.app/installation
· Issues: https://gitlab.com/gitdeeper3/seismo/-/issues
