import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

"""
Test basic Seismo Framework functionality without matplotlib.
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("🧪 Testing Seismo Framework Basic Functionality")
print("=" * 50)

# Test if we can import core modules
try:
    # Mock matplotlib to avoid import errors
    class MockMatplotlib:
        pass
    
    sys.modules['matplotlib'] = MockMatplotlib()
    sys.modules['matplotlib.pyplot'] = MockMatplotlib()
    sys.modules['matplotlib.dates'] = MockMatplotlib()
    
    import seismo_framework
    print(f"✅ Seismo Framework v{seismo_framework.__version__} imported")
    
    # Test parameter analyzers
    from seismo_framework.core.parameters import SeismicAnalyzer, DeformationAnalyzer
    print("✅ Parameter analyzers imported")
    
    # Test integration
    from seismo_framework.core.integration import ParameterIntegrator
    print("✅ Parameter integrator imported")
    
    # Test monitoring
    from seismo_framework.core.monitoring import RealTimeMonitor
    print("✅ Real-time monitor imported")
    
    # Test utilities
    from seismo_framework.core.utils import DataValidator, AlertManager
    print("✅ Utilities imported")
    
    print("\n🔧 Testing functionality...")
    
    # 1. Test seismic analyzer
    seismic = SeismicAnalyzer()
    seismic_result = seismic.analyze({
        'events': [
            {'magnitude': 3.5, 'time': '2024-01-01T00:00:00'},
            {'magnitude': 4.0, 'time': '2024-01-01T01:00:00'},
            {'magnitude': 3.8, 'time': '2024-01-01T02:00:00'},
        ]
    })
    print(f"✅ Seismic analysis: index={seismic_result.get('seismic_index', 0):.3f}")
    
    # 2. Test deformation analyzer
    deformation = DeformationAnalyzer()
    deformation_result = deformation.analyze({
        'gps_displacements': {
            'rates': [5.0, 6.0, 7.0, 8.0],
            'stations': ['STA1', 'STA2', 'STA3', 'STA4']
        }
    })
    print(f"✅ Deformation analysis: index={deformation_result.get('deformation_index', 0):.3f}")
    
    # 3. Test parameter integration
    integrator = ParameterIntegrator()
    integrated_result = integrator.integrate({
        'seismic': seismic_result.get('seismic_index', 0.5),
        'deformation': deformation_result.get('deformation_index', 0.5),
        'hydrogeological': 0.4,
        'electrical': 0.3,
        'magnetic': 0.6,
        'instability': 0.5,
        'stress': 0.4,
        'rock_properties': 0.3
    })
    print(f"✅ Parameter integration: score={integrated_result.get('integrated_score', 0):.3f}")
    print(f"✅ Alert level: {integrated_result.get('alert_level', 'normal').upper()}")
    
    # 4. Test alert manager
    alert = AlertManager.create_alert(
        alert_level=integrated_result['alert_level'],
        message=f"Integrated score: {integrated_result['integrated_score']:.3f}",
        parameters={'seismic': seismic_result['seismic_index']},
        location="Test Volcano"
    )
    print(f"✅ Alert created: {alert['alert_level'].upper()}")
    
    # 5. Test data validator
    validator = DataValidator()
    validated = validator.validate_parameter_dict({
        'seismic': 0.75,
        'deformation': 0.60,
        'test_invalid': 'not_a_number'
    })
    print(f"✅ Data validation: {len(validated)} valid parameters")
    
    print("\n🎉 ALL BASIC FUNCTIONALITY TESTS PASSED!")
    print("\n📊 Summary:")
    print(f"   Seismic Index: {seismic_result['seismic_index']:.3f}")
    print(f"   Deformation Index: {deformation_result['deformation_index']:.3f}")
    print(f"   Integrated Score: {integrated_result['integrated_score']:.3f}")
    print(f"   Alert Level: {integrated_result['alert_level'].upper()}")
    
    print("\n💡 Next steps:")
    print("   1. Framework is functional without matplotlib")
    print("   2. You can use text reports and CSV exports")
    print("   3. Install matplotlib later for visualization")
    print("   4. Start using the monitoring system")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n🔧 Troubleshooting:")
    print("   Install basic requirements:")
    print("   pip install numpy pandas python-dateutil pytz tzlocal pyyaml")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
