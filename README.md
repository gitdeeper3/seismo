# Seismo Framework 🌋

[![PyPI version](https://img.shields.io/pypi/v/seismo-framework.svg)](https://pypi.org/project/seismo-framework/)
[![OSF](https://img.shields.io/badge/OSF-Preregistration-blue)](https://osf.io/pm3fq)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18563973.svg)](https://doi.org/10.5281/zenodo.18563973)
[![GitLab](https://img.shields.io/badge/GitLab-Repository-orange)](https://gitlab.com/gitdeeper3/seismo)
[![GitHub](https://img.shields.io/badge/GitHub-Mirror-black)](https://github.com/gitdeeper3/seismo)
[![Bitbucket](https://img.shields.io/badge/Bitbucket-Mirror-blue)](https://bitbucket.org/gitdeeper3/seismo/)
[![Codeberg](https://img.shields.io/badge/Codeberg-Mirror-green)](https://codeberg.org/gitdeeper2/seismo/)
[![Python versions](https://img.shields.io/pypi/pyversions/seismo-framework.svg)](https://pypi.org/project/seismo-framework/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Seismo Framework** is a comprehensive, open-source seismic monitoring and earthquake forecasting system designed for scientific research and geophysical analysis.

## 🚀 Features

### 🔬 Scientific Foundation
- **45 research equations** from peer-reviewed seismic studies
- **4-level alert system**: GREEN/YELLOW/ORANGE/RED
- **Bayesian probability** for uncertainty quantification
- **Stress accumulation models** (Coulomb failure criteria)

### 💻 Technical Capabilities
- **FastAPI REST server** with automatic documentation
- **8-parameter monitoring system**:
  - Seismic frequency-magnitude analysis
  - GPS/InSAR deformation tracking
  - Hydrogeological anomaly detection
  - Electrical and magnetic monitoring
  - Stress and instability assessment
  - Rock properties analysis
- **AI-powered anomaly detection**
- **Real-time processing** (<100ms latency)

## 📦 Installation

```bash
pip install seismo-framework==2.0.2
```

🎯 Quick Start

```python
from seismo_framework import SeismoFramework

# Initialize the framework
seismo = SeismoFramework()

# Analyze seismic data
result = seismo.analyze_v2({
    'seismic': 18.5,
    'deformation': 14.2,
    'magnetic': 47.8,
    'stress': 0.72,
    'region': 'subduction_zone'
})

print(f"Alert Level: {result['alert_level']}")  # GREEN/YELLOW/ORANGE/RED
print(f"Confidence: {result['confidence']}%")
```

🌐 Live Systems

· 🌍 Website: https://seismo.netlify.app/
· 📊 Dashboard: https://seismo.netlify.app/dashboard
· 📚 Documentation: https://seismo.netlify.app/documentation
· 🔬 Research: https://seismo.netlify.app/#research

📁 Project Structure

```
.
├── AUTHORS.md                    # Project contributors
├── CHANGELOG.md                  # Version history
├── CITATION.cff                  # Citation metadata
├── CONTRIBUTING.md               # Contribution guidelines
├── DEPLOY.md                     # Deployment instructions
├── Dockerfile.txt                # Docker documentation
├── INSTALL.md                    # Installation guide
├── LICENSE                       # MIT License
├── MANIFEST.in                   # Package inclusion rules
├── OSF_REGISTRATION.md          # OSF Preregistration docs
├── QUICKSTART.md                 # Quick start guide
├── README.md                     # This file
├── README_PYPI.md               # PyPI package description
├── RELEASE_NOTES.md             # Release notes
├── Seismo.zip                   # Complete project archive
├── config/                       # Configuration files
├── data/                         # Data storage
│   ├── enhanced/                # Enhanced datasets
│   ├── exports/                 # Data exports
│   └── samples/                 # Sample data
├── dist/                         # Built packages
│   ├── seismo_framework-2.0.2-py3-none-any.whl
│   └── seismo_framework-2.0.2.tar.gz
├── docker/                       # Docker configuration
│   └── Dockerfile
├── docs/                         # Documentation
│   ├── api/                     # API documentation
│   ├── research/                # Research papers
│   │   ├── Seismo_Research_Paper.docx
│   │   ├── Seismo_Research_Paper.pdf
│   │   └── zenodo_troubleshooting_guide.md
│   └── user_guide/              # User documentation
│       ├── AI_MODULE_API.md
│       ├── AUTHORS.md
│       ├── CHANGELOG.md
│       ├── CONTRIBUTING.md
│       ├── DEPLOY.md
│       ├── INSTALL.md
│       ├── QUICKSTART.md
│       ├── README.md
│       └── README_PYPI.md
├── pyproject.toml                # Build configuration
├── reports/                      # Generated reports
│   ├── alerts/                  # Alert reports
│   ├── daily/                   # Daily reports
│   ├── enhanced/                # Enhanced analysis
│   ├── validation/              # Validation reports
│   └── weekly/                  # Weekly summaries
├── requirements.txt              # Dependencies
├── requirements_no_gui.txt       # Minimal dependencies
├── scripts/                      # Automation scripts
│   ├── build/                   # Build scripts
│   ├── deployment/              # Deployment scripts
│   ├── run_ai_directly.py       # AI testing
│   ├── run_all_tests.sh         # Test runner
│   ├── run_seismo_simulation.py # Simulation
│   ├── run_tests.py             # Test runner
│   └── utilities/               # Utility scripts
├── src/                          # Source code
│   └── seismo_framework/        # Main package
│       ├── __init__.py          # Package initialization
│       ├── ai_module/           # AI components
│       ├── api/                 # FastAPI server
│       ├── config/              # Configuration
│       ├── core/                # Core modules
│       ├── data/                # Data handling
│       ├── deployment.py        # Deployment
│       ├── integration.py       # Multi-parameter integration
│       ├── test_ai_fixed.py     # AI tests
│       ├── test_ai_module.py    # AI module tests
│       ├── test_local_ai.py     # Local AI tests
│       ├── training.py          # Model training
│       ├── utils/               # Utilities
│       └── web/                 # Web components
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── __pycache__/
│   ├── config/
│   ├── data/
│   ├── enhanced_test.py
│   ├── final_validation.py
│   ├── integration/
│   ├── minimal_test.py
│   ├── organize_project.py
│   ├── practical_example.py
│   ├── reports/
│   ├── run_all_tests.py
│   ├── test_alerts.py
│   ├── test_basic_functionality.py
│   ├── test_complete_model.py
│   ├── test_comprehensive_v2.py
│   ├── test_core.py
│   ├── test_integration.py
│   ├── test_no_scipy.py
│   ├── test_seismo.py
│   └── unit/
└── wiki/
    └── Home.md                  # Project wiki

47 directories, 103 files
```

🔗 Source Code Repositories

· Primary (GitLab): https://gitlab.com/gitdeeper3/seismo
· GitHub Mirror: https://github.com/gitdeeper3/seismo
· Bitbucket Mirror: https://bitbucket.org/gitdeeper3/seismo
· Codeberg Mirror: https://codeberg.org/gitdeeper2/seismo

🐛 Issue Tracking

· Report Issues: https://gitlab.com/gitdeeper3/seismo/-/issues

🧪 Testing

```bash
# Run all tests
./scripts/run_all_tests.sh

# Run specific tests
python -m pytest tests/test_core.py
python -m pytest tests/test_ai_module.py
```

🐳 Docker Deployment

```bash
# Build Docker image
docker build -f docker/Dockerfile -t seismo-framework .

# Run container
docker run -p 8000:8000 seismo-framework

# Access API docs at http://localhost:8000/docs
```

📊 Performance Metrics

Metric Value
Analysis Latency <100ms
Classification Accuracy 82-88%
Test Coverage 100%
Alert Levels 4
Research Equations 45

👥 Contributors

Principal Investigator

· Samir Baladi (@gitdeeper)
· Email: gitdeeper@gmail.com
· ORCID: 0009-0003-8903-0029
· Contact: +16142642074

📝 Citation

If you use Seismo Framework in your research, please cite:

APA Style (Zenodo):

```bibtex
@software{baladi_seismo_2026,
  author = {Baladi, Samir},
  title = {An Eight-Parameter Assessment Framework for Tectonic Stress Evolution and Major Earthquake Probability Forecasting},
  year = {2026},
  publisher = {Zenodo},
  version = {2.0.2},
  doi = {10.5281/zenodo.18563973},
  url = {https://doi.org/10.5281/zenodo.18563973}
}
```

BibTeX (PyPI Package):

```bibtex
@software{seismo_framework_2026,
  author = {Baladi, Samir},
  title = {Seismo Framework: Multi-parameter Seismic Monitoring System},
  year = {2026},
  publisher = {PyPI},
  version = {2.0.2},
  url = {https://pypi.org/project/seismo-framework/2.0.2/}
}
```

Chicago Style (OSF Preregistration):

```bibtex
@software{baladi_osf_2026,
  author = {Baladi, Samir},
  title = {OSF Preregistration: An Eight-Parameter Assessment Framework for Tectonic Stress Evolution and Major Earthquake Probability Forecasting},
  year = {2026},
  url = {https://osf.io/pm3fq},
  note = {OSF Preregistration}
}
```

🔒 License

This project is licensed under the MIT License - see the LICENSE file for details.

🚨 Disclaimer

Seismo Framework is a research tool for scientific investigation of seismic precursors. It is not intended for public earthquake prediction or emergency warnings without proper validation and calibration for specific regions.

---

Latest Release: v2.0.2 (2026-02-09)
PyPI Package: https://pypi.org/project/seismo-framework/2.0.2/
Zenodo DOI: 10.5281/zenodo.18563973
OSF Registration: https://osf.io/pm3fq
GitHub Mirror: https://github.com/gitdeeper3/seismo
Bitbucket Mirror: https://bitbucket.org/gitdeeper3/seismo/
Codeberg Mirror: https://codeberg.org/gitdeeper2/seismo/
