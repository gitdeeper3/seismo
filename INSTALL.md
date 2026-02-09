
Installation Guide

📦 PyPI Installation (Recommended)

Basic Installation

```bash
pip install seismo-framework==2.0.2
```

Installation with Optional Dependencies

```bash
pip install seismo-framework[dev]==2.0.2
```

🔧 Source Installation

Clone Repository

```bash
git clone https://gitlab.com/gitdeeper3/seismo.git
cd seismo
```

Install in Development Mode

```bash
pip install -e .
```

Install with Development Tools

```bash
pip install -e .[dev]
```

🐳 Docker Installation

Build Docker Image

```bash
docker build -f docker/Dockerfile -t seismo-framework .
```

Run Container

```bash
docker run -p 8000:8000 seismo-framework
```

Access Services

· API Documentation: http://localhost:8000/docs
· Dashboard: http://localhost:8000/dashboard
· REST API: http://localhost:8000/api/v1

📋 System Requirements

Python Versions

· Python 3.8 or higher
· Recommended: Python 3.10+

Operating Systems

· Linux (Ubuntu, Debian, CentOS)
· macOS (10.15+)
· Windows (WSL2 recommended)

Hardware Requirements

· Minimum: 2GB RAM, 1GB disk space
· Recommended: 4GB RAM, 2GB disk space
· For large datasets: 8GB+ RAM, 10GB+ disk space

🔍 Verification

Check Installation

```bash
python -c "import seismo_framework; print(f'Seismo Framework {seismo_framework.__version__} installed successfully')"
```

Run Tests

```bash
./scripts/run_all_tests.sh
```

Check Dependencies

```bash
pip show seismo-framework
```

⚠️ Troubleshooting

Common Issues

1. Permission Errors
   ```bash
   pip install --user seismo-framework==2.0.2
   ```
2. Dependency Conflicts
   ```bash
   pip install seismo-framework==2.0.2 --no-deps
   pip install numpy scipy pandas fastapi uvicorn
   ```
3. Python Version Issues
   ```bash
   python --version  # Should be 3.8+
   ```
4. Network Issues
   ```bash
   pip install seismo-framework==2.0.2 -i https://pypi.org/simple
   ```

Getting Help

· Documentation: https://seismo.netlify.app/documentation
· Issues: https://gitlab.com/gitdeeper3/seismo/-/issues
· Email: gitdeeper@gmail.com

🔄 Upgrading

From Previous Version

```bash
pip install --upgrade seismo-framework==2.0.2
```

Clean Upgrade

```bash
pip uninstall seismo-framework -y
pip install seismo-framework==2.0.2
```

📊 Installation Methods Comparison

Method Pros Cons Use Case
PyPI Simple, automatic updates Limited customization Production, quick start
Source Full control, development Manual updates Development, customization
Docker Isolated, reproducible Larger footprint Deployment, testing

🎯 Quick Verification Script

```bash
#!/bin/bash
# verify_installation.sh

echo "🔍 Verifying Seismo Framework Installation..."

# Check Python version
python --version | grep -q "Python 3" && echo "✅ Python 3.x" || echo "❌ Python 3 required"

# Try to import
python3 -c "import seismo_framework" 2>/dev/null && echo "✅ Import successful" || echo "❌ Import failed"

# Check version
python3 -c "import seismo_framework; print(f'✅ Version: {getattr(seismo_framework, \"__version__\", \"unknown\")}')"

echo "🎉 Verification complete!"
```

---

Last updated: $(date +%Y-%m-%d)
