"""
Core modules of Seismo Framework.
"""

# Placeholder classes for testing
class SeismicAnalyzer:
    def __init__(self):
        self.name = "SeismicAnalyzer"
    
    def analyze(self, data=None):
        return {"seismic_index": 0.5, "event_count": 0}

class DeformationAnalyzer:
    def __init__(self):
        self.name = "DeformationAnalyzer"
    
    def analyze(self, data=None):
        return {"deformation_index": 0.7}

class ParameterIntegrator:
    def __init__(self):
        self.name = "ParameterIntegrator"
    
    def integrate(self, params=None):
        return {"integrated_score": 0.6, "alert_level": "NORMAL"}

class RealTimeMonitor:
    def __init__(self):
        self.name = "RealTimeMonitor"
    
    def start(self):
        return {"status": "monitoring_started"}

# Export classes
__all__ = [
    'SeismicAnalyzer',
    'DeformationAnalyzer', 
    'ParameterIntegrator',
    'RealTimeMonitor'
]
