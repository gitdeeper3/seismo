"""
Integration module for parameter combination and fusion.
"""
from .algorithms import ParameterIntegrator
from .weighting import WeightOptimizer
__all__ = [
    'ParameterIntegrator',
    'WeightOptimizer'
]
def get_integration_methods():
    """List available integration methods."""
    return {
        'weighted_average': 'Weighted average of parameters',
        'pca': 'Principal Component Analysis',
        'fuzzy_logic': 'Fuzzy logic integration',
        'bayesian': 'Bayesian inference'
    }