"""
Test basic import of Seismo Framework.
GitLab compatible with relative paths.
"""

import sys
import os

# Add parent directory to path for relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("🧪 Testing Seismo Framework Import...")

try:
    import seismo_framework
    print("✅ Seismo Framework imported successfully")
    
    # Check version
    version = getattr(seismo_framework, '__version__', '1.0.0')
    print(f"   Version: {version}")
    
    # Check author (optional)
    author = getattr(seismo_framework, '__author__', 'Seismo Framework Team')
    print(f"   Author: {author}")
    
    # Check description
    desc = getattr(seismo_framework, '__description__', 'Seismic monitoring framework')
    print(f"   Description: {desc}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n🔧 Troubleshooting:")
    print("   1. Check if seismo_framework folder exists")
    print("   2. Verify __init__.py exists")
    print("   3. Check current directory:", os.getcwd())
