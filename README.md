# Seismo Framework 🌋

[![PyPI version](https://img.shields.io/pypi/v/seismo-framework.svg)](https://pypi.org/project/seismo-framework/)
[![OSF](https://img.shields.io/badge/OSF-Preregistration-blue)](https://osf.io/pm3fq)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18563973.svg)](https://doi.org/10.5281/zenodo.18563973)
[![Python versions](https://img.shields.io/pypi/pyversions/seismo-framework.svg)](https://pypi.org/project/seismo-framework/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitLab](https://img.shields.io/badge/GitLab-Repository-orange)](https://gitlab.com/gitdeeper3/seismo)

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
seismo/
├── src/seismo_framework/     # Source code
│   ├── core/                 # Core monitoring modules
│   ├── ai_module/           # AI integration
│   ├── api/                 # FastAPI REST server
│   └── web/                 # Dashboard components
├── tests/                    # Test suite (25+ tests)
├── docs/                     # Documentation
├── scripts/                  # Automation scripts
├── data/                     # Sample datasets
└── docker/                   # Docker configuration
```

🔗 Source Code Repositories

· Primary (GitLab): https://gitlab.com/gitdeeper3/seismo
· Mirror (Codeberg): https://codeberg.org/gitdeeper2/seismo
· Mirror (Bitbucket): https://bitbucket.org/gitdeeper3/seismo

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


Research Areas:

· Real-time seismic monitoring systems
· Multi-parameter geophysical integration
· Earthquake probability assessment
· Automated decision support frameworks

Repository Access:

· Primary: https://gitlab.com/gitdeeper3/seismo
· Mirror: https://github.com/gitdeeper3/seismo
· Backup: https://bitbucket.org/gitdeeper3/seismo
· Open Source: https://codeberg.org/gitdeeper2/seismo


🌐 Project Links

· Homepage: https://seismo.netlify.app
· Live Dashboard: https://seismo.netlify.app/dashboard
· Documentation: https://seismo.netlify.app/documentation
· PyPI Package: https://pypi.org/project/seismo-framework/
· Issue Tracking: https://gitlab.com/gitdeeper3/seismo/-/issues
· Scientific Paper: In preparation (target: Seismological Research Letters)
>>>>>>> a4bd259a6cce9a127d465808efa3ea03a0748f77

📝 Citation

If you use Seismo Framework in your research, please cite:

APA Style:

```bibtex
@software{baladi_seismo_2026,
  author = {Baladi, Samir},
    title = {An Eight-Parameter Assessment Framework for Tectonic Stress Evolution and Major Earthquake Probability Forecasting},
      year = {2026},
        publisher = {Zenodo},
          version = {2.0.0},
            doi = {10.5281/zenodo.18563973},
              url = {https://doi.org/10.5281/zenodo.18563973}
              }
              ```

              BibTeX:

              ```bibtex
              @software{seismo_framework_2026,
                author = {Samir Baladi},
                  title = {Seismo Framework: Multi-parameter Seismic Monitoring System},
                    year = {2026},
                      month = {February},
                        publisher = {Zenodo},
                          doi = {10.5281/zenodo.18563973},
                            url = {https://zenodo.org/records/18563973},
                              version = {2.0.0},
                                note = {Software for tectonic stress evolution and earthquake probability forecasting}
                                }
                                ```

🔒 License

This project is licensed under the MIT License - see the LICENSE file for details.

🚨 Disclaimer

Seismo Framework is a research tool for scientific investigation of seismic precursors. It is not intended for public earthquake prediction or emergency warnings without proper validation and calibration for specific regions.

---

Latest Release: v2.0.2 (2026-02-09)
PyPI Package: https://pypi.org/project/seismo-framework/
