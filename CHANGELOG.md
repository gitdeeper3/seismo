# Changelog

All notable changes to the Seismo Framework project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.2] - 2026-02-09

### Added
- **Complete PyPI package** with full Markdown description
- **All project URLs** on PyPI (Homepage, Documentation, Repository, Bug Tracker)
- **Comprehensive classifiers** for scientific research
- **Enhanced metadata** with complete package information

### Fixed
- **PyPI description display** with proper Markdown formatting
- **Metadata validation** for PyPI uploads
- **Package dependencies** specification

## [2.0.1] - 2026-02-09

### Fixed
- **Missing metadata_version** in PyPI upload
- **PyPI package validation** issues
- **API token authentication** problems

## [2.0.0] - 2026-02-08

### 🚀 Major Release - Research-Based Enhancement

#### Added
- **45 research equations** from seismic studies implemented
- **New physics module** (`core/physics/`) for scientific algorithms
- **4-level alert system**: GREEN/YELLOW/ORANGE/RED
- **Bayesian probability updating** for uncertainty quantification
- **Stress accumulation models** (Coulomb failure criteria)
- **FastAPI REST server** in new `api/` directory
- **Comprehensive test suite** with 100% coverage
- **DOI reference** via Zenodo (10.5281/zenodo.XXXXXXX)
- **Enhanced parameter thresholds** based on statistical analysis

#### Changed
- **Upgraded 8-parameter system** with improved monitoring
- **Enhanced integration algorithms** with weighted confidence
- **Improved alert generation** from 3 to 4 levels
- **Project structure reorganization** for better modularity
- **Performance optimization** (<100ms analysis latency)
- **Accuracy improvement** (82-88% classification rate)

#### Deprecated
- **3-level alert system** (NORMAL/ELEVATED/WATCH) in favor of 4-level system
- **Legacy parameter thresholds** replaced with research-based values

#### Fixed
- **Parameter normalization** issues
- **Confidence scoring** accuracy
- **Real-time calibration** stability
- **Backward compatibility** with v1.0.9 API

#### Security
- **Input validation** for all API endpoints
- **Data sanitization** for monitoring parameters
- **Error handling** improvements

## [1.0.9] - 2026-02-07

### Added
- **AI Module (`ai_module/`)** with anomaly detection
- **Physics Enhancer** for data quality improvement
- **Adaptive Weights** region-based parameter weighting
- **XAI Explanations** for decision transparency

### Enhanced
- **Main Framework Integration** with AI imports
- **Backward Compatibility** with v1.0.5 functionality
- **System Information** reporting AI availability

## [1.0.6] - 2026-02-07

### Fixed
- Minor bug fixes and documentation updates
- Improved error messages
- Enhanced test coverage

## [1.0.5] - 2026-02-01

### Added
- **Initial release** of Seismo Framework
- **8 specialized parameter analyzers**
- **Multi-parameter integration** system
- **3-level alert generation** (NORMAL/ELEVATED/WATCH)
- **Real-time monitoring** capabilities
- **Comprehensive test suite** (9 tests)
- **GitLab CI/CD pipeline**
- **PyPI package publication**

### Features
- **Seismic Analyzer**: Earthquake frequency-magnitude distribution
- **Deformation Monitoring**: GPS/InSAR displacement
- **Hydrogeological Analysis**: Groundwater and radon anomalies
- **Electrical Signals**: Resistivity and self-potential
- **Magnetic Variations**: Local field monitoring
- **Instability Analysis**: Dynamical system assessment
- **Stress Calculations**: Coulomb stress modeling
- **Rock Properties**: Vp/Vs ratios and attenuation

## Types of Changes

- **Added** for new features.
- **Changed** for changes in existing functionality.
- **Deprecated** for soon-to-be removed features.
- **Removed** for now removed features.
- **Fixed** for any bug fixes.
- **Security** in case of vulnerabilities.

---

**Maintainer**: Samir Baladi (@gitdeeper)  
**Email**: gitdeeper@gmail.com  
**ORCID**: 0009-0003-8903-0029  
**License**: MIT
