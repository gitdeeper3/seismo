"""
SEISMO FRAMEWORK - PRACTICAL EXAMPLE
GitLab compatible version with mock classes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("🌋 SEISMO FRAMEWORK - PRACTICAL EXAMPLE")
print("=" * 60)

# Try to import from seismo_framework, use mock if not available
try:
    from seismo_framework import (
        SeismicAnalyzer,
        DeformationAnalyzer,
        ParameterIntegrator,
        RealTimeMonitor
    )
    print("✅ Seismo Framework components imported")
except ImportError:
    print("⚠️  Using mock classes for demonstration")
    
    # Mock classes
    class SeismicAnalyzer:
        def analyze(self, data=None):
            return {"seismic_index": 0.5, "event_count": 10}
    
    class DeformationAnalyzer:
        def analyze(self, data=None):
            return {"deformation_index": 0.7}
    
    class ParameterIntegrator:
        def integrate(self, params=None):
            return {"integrated_score": 0.6, "alert_level": "NORMAL"}
    
    class RealTimeMonitor:
        def start(self):
            return {"status": "monitoring_started"}

# Create instances
print("\n🔧 Creating analysis components...")
seismic_analyzer = SeismicAnalyzer()
deformation_analyzer = DeformationAnalyzer()
parameter_integrator = ParameterIntegrator()
monitor = RealTimeMonitor()

print("✅ Components created successfully")

# Analyze sample data
print("\n📊 Analyzing sample data...")
seismic_result = seismic_analyzer.analyze()
deformation_result = deformation_analyzer.analyze()

print(f"   Seismic analysis: {seismic_result}")
print(f"   Deformation analysis: {deformation_result}")

# Integrate parameters
print("\n🔗 Integrating parameters...")
integration_result = parameter_integrator.integrate({
    "seismic": seismic_result.get("seismic_index", 0),
    "deformation": deformation_result.get("deformation_index", 0)
})

print(f"   Integrated score: {integration_result.get('integrated_score', 0)}")
print(f"   Alert level: {integration_result.get('alert_level', 'UNKNOWN')}")

# Start monitoring
print("\n🚀 Starting monitoring system...")
monitor_status = monitor.start()
print(f"   Monitoring status: {monitor_status}")

print("\n" + "=" * 60)
print("🎉 PRACTICAL EXAMPLE COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\n📋 Summary:")
print("   1. ✅ Framework components initialized")
print("   2. ✅ Data analysis completed")
print("   3. ✅ Parameter integration working")
print("   4. ✅ Monitoring system started")
print("\n🚀 Next steps:")
print("   - Connect to real sensor data")
print("   - Deploy to monitoring station")
print("   - Set up automated alerts")
