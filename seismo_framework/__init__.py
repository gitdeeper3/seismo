"""
Seismo Framework - Seismic Monitoring and Analysis
GitLab compatible version with relative imports.
"""

__version__ = "1.0.0"
__author__ = "Seismo Framework Team"
__email__ = "gitdeeper@gmail.com"
__description__ = "Advanced seismic data analysis and monitoring framework"

# Import from core
from .core import (
    SeismicAnalyzer,
    DeformationAnalyzer,
    ParameterIntegrator,
    RealTimeMonitor
)

# Core modules
__all__ = [
    'SeismicAnalyzer',
    'DeformationAnalyzer',
    'ParameterIntegrator', 
    'RealTimeMonitor',
    'core'
]

print(f"✅ Seismo Framework {__version__} loaded successfully")
