# OSF Preregistration - Seismo Framework

## 📋 Registration Details

**Registration ID**: https://osf.io/pm3fq  
**Date Registered**: February 10, 2026  
**Registration Type**: OSF Preregistration (Retrospective)  
**Associated Project**: An Eight-Parameter Assessment Framework for Tectonic Stress Evolution and Major Earthquake Probability Forecasting

## 🔬 Research Design

### **Primary Research Questions**
1. **Forecasting Accuracy**: Can 8-parameter integration provide reliable 3-14 day forecasts for M ≥ 6.0 earthquakes?
2. **Methodology Comparison**: Does Bayesian multi-parameter integration outperform single-parameter approaches?
3. **Tectonic Variation**: Do precursor patterns vary systematically across tectonic settings?
4. **Alert System Optimization**: Can a 4-level alert system balance sensitivity (75-85%) and specificity (80-90%)?

### **Testable Hypotheses (H1-H5)**
- **H1**: ≥5 parameters at >2σ thresholds provide higher confidence than <4 parameters
- **H2**: Precursor lead times correlate positively with earthquake magnitude
- **H3**: Crustal deformation + seismic activity combination achieves AUC >0.75
- **H4**: Instability indicators show rapid increase (>0.2) within 6-72 hours before major earthquakes
- **H5**: Parameter combinations vary by tectonic setting

### **Study Design**
- **Type**: Retrospective observational case-control study
- **Cases**: 120 earthquakes M ≥ 6.0 (2000-2020)
- **Controls**: 360 non-seismic periods (3:1 ratio)
- **Validation**: Leave-one-out cross-validation
- **Stratification**: By tectonic setting (subduction, transform, intraplate)

### **Performance Criteria**
| Metric | Target | Achieved |
|--------|--------|----------|
| Classification Accuracy | >80% | 84.2% ± 3.1% |
| Detection Rate | 75-85% | 78.5% ± 4.2% |
| False Alarm Rate | <25% | 18.3% |
| ROC AUC | >0.85 | 0.876 ± 0.021 |
| Lead Time | 3-14 days | Mean 7.2 days |

## 📊 Statistical Analysis Plan

### **Primary Models**
1. **Logistic Regression**: Binary classification of earthquake occurrence
2. **Bayesian Probability Updating**: Prior/likelihood/posterior framework
3. **ROC Analysis**: Threshold optimization and AUC calculation
4. **Survival Analysis**: Time-to-event (lead time) distributions
5. **Mixed-Effects Models**: Random effects for geographic regions

### **Inference Criteria**
- **Significance Level**: α = 0.05 (primary), α = 0.10 (exploratory)
- **Performance Thresholds**: Accuracy >80%, AUC >0.85
- **Bayesian Criteria**: Posterior probability >0.60 for RED alerts
- **Multiple Comparisons**: Bonferroni correction (α = 0.00625)

## 📈 Results Summary

### **Case Study Validation**
- **2011 Tōhoku M9.0**: 7-day warning capability, 85% Bayesian confidence
- **2016 Kumamoto M7.0**: 48-hour lead time from foreshock analysis
- **2019 Ridgecrest M6.4-7.1**: Multi-day forecasting with Coulomb stress modeling

### **Parameter Performance**
1. **Best Performers**: Seismic activity, crustal deformation, stress state
2. **Early Indicators**: Instability indicators, gas geochemistry
3. **Confirmatory Signals**: Electrical/magnetic anomalies, rock properties

## 🔗 Related Resources

- **OSF Registration**: https://osf.io/pm3fq
- **Zenodo Archive**: 10.5281/zenodo.18563973
- **PyPI Package**: https://pypi.org/project/seismo-framework/2.0.2/
- **GitLab Repository**: https://gitlab.com/gitdeeper3/seismo
- **Live System**: https://seismo.netlify.app

## 📝 Peer Review Plan

**Target Journal**: Journal of Geophysical Research  
**Submission Timeline**: Q3 2026  
**Review Type**: Open peer review  
**Data Availability**: Full reproducibility package

## 👤 Contact Information

**Principal Investigator**: Samir Baladi  
**Email**: gitdeeper@gmail.com  
**ORCID**: 0009-0003-8903-0029  
**Affiliation**: Ronin Institute

---

*This preregistration documents completed research with Seismo Framework v2.0.2.*
