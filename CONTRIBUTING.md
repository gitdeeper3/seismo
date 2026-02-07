# Contributing to Seismo Framework

We welcome contributions from scientists, developers, and researchers interested in seismic monitoring and earthquake forecasting.

## How to Contribute

### 1. Reporting Issues
- Use GitLab Issues for bug reports and feature requests
- Include detailed description and steps to reproduce
- Attach relevant data or code snippets

### 2. Development Workflow
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Merge Request

### 3. Coding Standards
- Follow PEP 8 for Python code
- Use type hints for function signatures
- Write docstrings for all public functions/classes
- Include unit tests for new functionality

### 4. Testing Requirements
- Run existing tests before submitting: `pytest tests/`
- Maintain or improve test coverage
- Test with different Python versions (3.9+)

### 5. Documentation
- Update README.md for significant changes
- Document new API functions
- Add example notebooks for new features

## Contribution Areas

### Scientific Contributions
- New monitoring parameter development
- Improved physical models
- Regional adaptation algorithms
- Validation methodologies

### Technical Contributions
- Performance optimization
- Database integration
- Web interface improvements
- Deployment automation

### Data Contributions
- Test datasets
- Regional monitoring data
- Historical earthquake catalogs
- Instrument calibration data

## Development Setup

```bash
# Clone repository
git clone https://gitlab.com/gitdeeper3/seismo.git
cd seismo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install development dependencies
pip install -e .[dev,test]

# Run tests
pytest tests/
```

Code Review Process

1. All merge requests require at least one review
2. Code must pass CI/CD pipeline
3. Documentation must be updated
4. Tests must pass with >=80% coverage

Contact

For questions about contributing, contact:

· Email: gitdeeper@gmail.com
· GitLab Issues: https://gitlab.com/gitdeeper3/seismo/-/issues
