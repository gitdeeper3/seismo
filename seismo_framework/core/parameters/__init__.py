"""
Volcano Monitoring Parameters Module
Contains analyzers for all 8 key geophysical parameters.
"""
from .seismic import SeismicAnalyzer
from .deformation import DeformationAnalyzer
from .hydrogeological import HydrogeologicalAnalyzer
from .electrical import ElectricalAnalyzer
from .magnetic import MagneticAnalyzer
from .instability import InstabilityAnalyzer
from .stress import StressAnalyzer
from .rock_properties import RockPropertiesAnalyzer
__all__ = [
    'SeismicAnalyzer',
    'DeformationAnalyzer',
    'HydrogeologicalAnalyzer',
    'ElectricalAnalyzer',
    'MagneticAnalyzer',
    'InstabilityAnalyzer',
    'StressAnalyzer',
    'RockPropertiesAnalyzer'
]
def list_parameters():
    """List all available parameter analyzers."""
    return {
        'seismic': 'Seismic activity analysis',
        'deformation': 'Ground deformation monitoring',
        'hydrogeological': 'Hydrogeological indicators',
        'electrical': 'Electrical signals analysis',
        'magnetic': 'Magnetic anomalies detection',
        'instability': 'Slope instability assessment',
        'stress': 'Tectonic stress analysis',
        'rock_properties': 'Rock properties evaluation'
    }