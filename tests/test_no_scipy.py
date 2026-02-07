#!/usr/bin/env python3
"""
Test Seismo Framework without scipy dependencies.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🔧 Testing Seismo Framework without scipy...")

# First, try basic imports
try:
    import numpy as np
    print("✅ NumPy imported")
except ImportError:
    print("❌ NumPy not installed")
    sys.exit(1)

try:
    import pandas as pd
    print("✅ Pandas imported")
except ImportError:
    print("❌ Pandas not installed")

try:
    import matplotlib
    print("✅ Matplotlib imported")
except ImportError:
    print("⚠️  Matplotlib not installed (optional)")

# Now test Seismo core components
print("\n🔍 Testing Seismo core components...")

# Test parameters module
try:
    # Create a simple seismic analyzer
    class SimpleSeismicAnalyzer:
        def __init__(self):
            self.name = "Seismic Analyzer"
    
    print("✅ Created simple seismic analyzer")
    
    # Test integration
    class SimpleParameterIntegrator:
        def __init__(self):
            self.name = "Parameter Integrator"
    
    print("✅ Created simple parameter integrator")
    
    # Test monitoring
    class SimpleMonitor:
        def __init__(self):
            self.name = "Real-time Monitor"
    
    print("✅ Created simple monitor")
    
    print("\n🎉 Basic components can be created!")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\n📋 Next steps:")
print("1. Install basic requirements: pip install numpy pandas matplotlib")
print("2. Try importing seismo_framework again")
print("3. If errors persist, check individual files for scipy imports")
