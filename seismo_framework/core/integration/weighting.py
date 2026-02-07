"""
Parameter weighting algorithms.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
logger = logging.getLogger(__name__)
class WeightOptimizer:
    """
    Optimizes parameter weights based on historical data.
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._setup_defaults()
    def _setup_defaults(self):
        defaults = {
            'optimization_method': 'correlation',
            'min_weight': 0.05,
            'max_weight': 0.30,
            'target_column': 'eruption_occurred',
            'validation_folds': 5,
            'random_state': 42,
            'update_frequency_days': 30
        }
        defaults.update(self.config)
        self.config = defaults
    def optimize(self, data: pd.DataFrame, 
                 parameter_columns: List[str]) -> Dict[str, Any]:
        """
        Optimize weights for parameters.
        Args:
            data: Historical data
            parameter_columns: List of parameter column names
        Returns:
            Dictionary with optimized weights and metrics
        """
        results = {
            'optimized_weights': {},
            'performance_metrics': {},
            'uncertainties': {},
            'metadata': {
                'optimization_time': pd.Timestamp.now().isoformat(),
                'parameters': parameter_columns
            }
        }
        try:
            # Prepare data
            X, y = self._prepare_data(data, parameter_columns)
            if X.shape[0] < 10 or X.shape[1] < 2:
                logger.warning("Insufficient data for optimization")
                return self._get_default_weights(parameter_columns)
            # Apply optimization method
            method = self.config['optimization_method']
            if method == 'correlation':
                optimized = self._correlation_based(X, y, parameter_columns)
            elif method == 'regression':
                optimized = self._regression_based(X, y, parameter_columns)
            elif method == 'entropy':
                optimized = self._entropy_based(X, parameter_columns)
            elif method == 'genetic':
                optimized = self._genetic_optimization(X, y, parameter_columns)
            else:
                optimized = self._correlation_based(X, y, parameter_columns)
            # Validate and normalize weights
            validated_weights = self._validate_weights(optimized['weights'])
            results.update({
                'optimized_weights': validated_weights,
                'performance_metrics': optimized.get('metrics', {}),
                'uncertainties': optimized.get('uncertainties', {}),
                'metadata': {
                    **results['metadata'],
                    'method': method,
                    'data_points': X.shape[0],
                    'success': True
                }
            })
            logger.info(f"Weight optimization completed using {method} method")
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return self._get_default_weights(parameter_columns)
        return results
    def _prepare_data(self, data: pd.DataFrame, 
                     parameter_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for optimization."""
        # Filter valid data
        valid_data = data[parameter_columns + [self.config['target_column']]].copy()
        valid_data = valid_data.dropna()
        if len(valid_data) < 10:
            raise ValueError("Insufficient valid data points")
        # Extract features and target
        X = valid_data[parameter_columns].values
        y = valid_data[self.config['target_column']].values
        # Normalize features
        X_normalized = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0, where=~np.isnan(X))
        X_normalized = np.nan_to_num(X_normalized, nan=0.0)
        return X_normalized, y
    def _correlation_based(self, X: np.ndarray, y: np.ndarray,
                          parameter_names: List[str]) -> Dict[str, Any]:
        """Weight optimization based on correlation with target."""
        weights = {}
        correlations = []
        for i, name in enumerate(parameter_names):
            # Calculate absolute correlation with target
            if len(np.unique(y)) > 1:
                corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
                if np.isnan(corr):
                    corr = 0.0
            else:
                # If no target variation, use variance
                corr = np.var(X[:, i])
            correlations.append(corr)
        # Convert correlations to weights
        if np.sum(correlations) > 0:
            raw_weights = np.array(correlations) / np.sum(correlations)
        else:
            raw_weights = np.ones(len(parameter_names)) / len(parameter_names)
        # Apply min/max constraints
        min_w = self.config['min_weight']
        max_w = self.config['max_weight']
        constrained_weights = np.clip(raw_weights, min_w, max_w)
        constrained_weights = constrained_weights / np.sum(constrained_weights)
        # Assign weights
        for i, name in enumerate(parameter_names):
            weights[name] = float(constrained_weights[i])
        # Calculate performance metrics
        metrics = {
            'average_correlation': float(np.mean(correlations)),
            'max_correlation': float(np.max(correlations)),
            'min_correlation': float(np.min(correlations))
        }
        # Estimate uncertainties
        uncertainties = {}
        for i, name in enumerate(parameter_names):
            # Bootstrap uncertainty estimation
            n_bootstraps = 100
            boot_weights = []
            for _ in range(n_bootstraps):
                # Bootstrap sample
                idx = np.random.choice(len(X), len(X), replace=True)
                X_boot = X[idx, i]
                y_boot = y[idx]
                if len(np.unique(y_boot)) > 1:
                    corr_boot = np.abs(np.corrcoef(X_boot, y_boot)[0, 1])
                else:
                    corr_boot = np.var(X_boot)
                boot_weights.append(corr_boot)
            # Normalize bootstrap weights
            if np.sum(boot_weights) > 0:
                boot_weights_norm = np.array(boot_weights) / np.sum(boot_weights)
                uncertainty = np.std(boot_weights_norm)
            else:
                uncertainty = 0.1
            uncertainties[name] = float(uncertainty)
        return {
            'weights': weights,
            'metrics': metrics,
            'uncertainties': uncertainties
        }
    def _regression_based(self, X: np.ndarray, y: np.ndarray,
                         parameter_names: List[str]) -> Dict[str, Any]:
        """Weight optimization using regression coefficients."""
        weights = {}
        if len(np.unique(y)) < 2:
            logger.warning("Insufficient target variation for regression")
            return self._correlation_based(X, y, parameter_names)
        try:
            # Ridge regression for stable coefficients
            from sklearn.linear_model import RidgeCV
            # Standardize target
            y_std = (y - np.mean(y)) / np.std(y)
            # Fit ridge regression
            ridge = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
            ridge.fit(X, y_std)
            # Use absolute coefficients as weights
            coefficients = np.abs(ridge.coef_)
            if np.sum(coefficients) > 0:
                raw_weights = coefficients / np.sum(coefficients)
            else:
                raw_weights = np.ones(len(parameter_names)) / len(parameter_names)
            # Apply constraints
            min_w = self.config['min_weight']
            max_w = self.config['max_weight']
            constrained_weights = np.clip(raw_weights, min_w, max_w)
            constrained_weights = constrained_weights / np.sum(constrained_weights)
            # Assign weights
            for i, name in enumerate(parameter_names):
                weights[name] = float(constrained_weights[i])
            # Performance metrics
            y_pred = ridge.predict(X)
            r2 = 1 - np.sum((y_std - y_pred) ** 2) / np.sum((y_std - np.mean(y_std)) ** 2)
            metrics = {
                'r2_score': float(r2),
                'best_alpha': float(ridge.alpha_),
                'cv_score': float(ridge.best_score_)
            }
            uncertainties = self._bootstrap_uncertainties(X, y, parameter_names, 'regression')
        except Exception as e:
            logger.warning(f"Regression optimization failed: {e}")
            return self._correlation_based(X, y, parameter_names)
        return {
            'weights': weights,
            'metrics': metrics,
            'uncertainties': uncertainties
        }
    def _entropy_based(self, X: np.ndarray, 
                      parameter_names: List[str]) -> Dict[str, Any]:
        """Weight optimization based on information content."""
        weights = {}
        # Calculate information content (inverse of entropy)
        informations = []
        for i in range(X.shape[1]):
            # Discretize values
            values = X[:, i]
            if len(np.unique(values)) > 1:
                # Calculate entropy
                hist, _ = np.histogram(values, bins=10, density=True)
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log(hist))
                # Information is inverse of entropy
                information = 1.0 / (entropy + 1e-10)
            else:
                information = 0.0
            informations.append(information)
        # Convert to weights
        if np.sum(informations) > 0:
            raw_weights = np.array(informations) / np.sum(informations)
        else:
            raw_weights = np.ones(len(parameter_names)) / len(parameter_names)
        # Apply constraints
        min_w = self.config['min_weight']
        max_w = self.config['max_weight']
        constrained_weights = np.clip(raw_weights, min_w, max_w)
        constrained_weights = constrained_weights / np.sum(constrained_weights)
        # Assign weights
        for i, name in enumerate(parameter_names):
            weights[name] = float(constrained_weights[i])
        metrics = {
            'average_information': float(np.mean(informations)),
            'total_entropy': float(np.sum([1/(info+1e-10) for info in informations]))
        }
        uncertainties = {}
        for name in parameter_names:
            uncertainties[name] = 0.1  # Default uncertainty
        return {
            'weights': weights,
            'metrics': metrics,
            'uncertainties': uncertainties
        }
    def _genetic_optimization(self, X: np.ndarray, y: np.ndarray,
                             parameter_names: List[str]) -> Dict[str, Any]:
        """Genetic algorithm optimization."""
        weights = {}
        n_params = len(parameter_names)
        # Objective function: maximize correlation between weighted sum and target
        def objective(individual):
            # Ensure weights are positive and sum to 1
            individual = np.abs(individual)
            if np.sum(individual) > 0:
                individual = individual / np.sum(individual)
            # Calculate weighted sum
            weighted_sum = np.dot(X, individual)
            # Calculate correlation with target
            if len(np.unique(y)) > 1 and len(np.unique(weighted_sum)) > 1:
                corr = np.corrcoef(weighted_sum, y)[0, 1]
                if np.isnan(corr):
                    return 0.0
                return np.abs(corr)
            else:
                return 0.0
        # Simple optimization using random search
        best_score = -1
        best_weights = None
        for _ in range(1000):
            # Random weights
            random_weights = np.random.dirichlet(np.ones(n_params))
            # Apply constraints
            min_w = self.config['min_weight']
            max_w = self.config['max_weight']
            # Scale to min-max range
            scaled_weights = min_w + (max_w - min_w) * random_weights
            scaled_weights = scaled_weights / np.sum(scaled_weights)
            score = objective(scaled_weights)
            if score > best_score:
                best_score = score
                best_weights = scaled_weights
        if best_weights is not None:
            for i, name in enumerate(parameter_names):
                weights[name] = float(best_weights[i])
        else:
            # Fallback to equal weights
            equal_weight = 1.0 / n_params
            for name in parameter_names:
                weights[name] = equal_weight
        metrics = {
            'best_score': float(best_score),
            'method': 'genetic_algorithm'
        }
        uncertainties = self._bootstrap_uncertainties(X, y, parameter_names, 'genetic')
        return {
            'weights': weights,
            'metrics': metrics,
            'uncertainties': uncertainties
        }
    def _bootstrap_uncertainties(self, X: np.ndarray, y: np.ndarray,
                                parameter_names: List[str],
                                method: str) -> Dict[str, float]:
        """Estimate weight uncertainties using bootstrap."""
        uncertainties = {}
        n_bootstraps = 50
        n_params = len(parameter_names)
        bootstrap_weights = np.zeros((n_bootstraps, n_params))
        for b in range(n_bootstraps):
            # Bootstrap sample
            idx = np.random.choice(len(X), len(X), replace=True)
            X_boot = X[idx]
            y_boot = y[idx]
            # Get weights for bootstrap sample
            if method == 'regression':
                boot_result = self._regression_based(X_boot, y_boot, parameter_names)
            elif method == 'genetic':
                boot_result = self._genetic_optimization(X_boot, y_boot, parameter_names)
            else:
                boot_result = self._correlation_based(X_boot, y_boot, parameter_names)
            boot_weights = list(boot_result['weights'].values())
            bootstrap_weights[b, :] = boot_weights
        # Calculate standard deviation for each parameter
        for i, name in enumerate(parameter_names):
            uncertainties[name] = float(np.std(bootstrap_weights[:, i]))
        return uncertainties
    def _validate_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Validate and normalize weights."""
        validated = {}
        # Check all weights are positive
        positive_weights = {k: max(0.0, v) for k, v in weights.items()}
        # Normalize to sum to 1
        total = sum(positive_weights.values())
        if total > 0:
            for k, v in positive_weights.items():
                validated[k] = v / total
        else:
            # Equal weights if all are zero
            equal_weight = 1.0 / len(weights) if weights else 1.0
            for k in weights.keys():
                validated[k] = equal_weight
        # Apply min/max constraints
        min_w = self.config['min_weight']
        max_w = self.config['max_weight']
        for k in list(validated.keys()):
            validated[k] = np.clip(validated[k], min_w, max_w)
        # Renormalize after clipping
        total = sum(validated.values())
        if total > 0:
            validated = {k: v/total for k, v in validated.items()}
        return validated
    def _get_default_weights(self, parameter_names: List[str]) -> Dict[str, Any]:
        """Return default weights."""
        n_params = len(parameter_names)
        if n_params == 0:
            return {
                'optimized_weights': {},
                'performance_metrics': {},
                'uncertainties': {},
                'metadata': {
                    'optimization_time': pd.Timestamp.now().isoformat(),
                    'method': 'default',
                    'success': False,
                    'error': 'No parameters provided'
                }
            }
        # Equal weights
        equal_weight = 1.0 / n_params
        default_weights = {name: equal_weight for name in parameter_names}
        return {
            'optimized_weights': default_weights,
            'performance_metrics': {
                'method': 'default',
                'note': 'Using equal weights due to insufficient data'
            },
            'uncertainties': {name: 0.15 for name in parameter_names},
            'metadata': {
                'optimization_time': pd.Timestamp.now().isoformat(),
                'method': 'default',
                'parameters': parameter_names,
                'data_points': 0,
                'success': False
            }
        }
    def update_weights_dynamically(self, historical_data: pd.DataFrame,
                                  current_weights: Dict[str, float],
                                  parameter_columns: List[str]) -> Dict[str, float]:
        """
        Update weights dynamically based on recent data.
        Args:
            historical_data: Recent historical data
            current_weights: Current weight values
            parameter_columns: Parameter columns to consider
        Returns:
            Updated weights
        """
        # Use only recent data
        if 'timestamp' in historical_data.columns:
            historical_data = historical_data.sort_values('timestamp')
            recent_cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
            recent_data = historical_data[historical_data['timestamp'] > recent_cutoff]
            if len(recent_data) > 50:  # Enough data for update
                # Optimize on recent data
                recent_result = self.optimize(recent_data, parameter_columns)
                if recent_result['metadata'].get('success', False):
                    # Blend with current weights (70% old, 30% new)
                    blended_weights = {}
                    for name in parameter_columns:
                        old_weight = current_weights.get(name, 1.0/len(parameter_columns))
                        new_weight = recent_result['optimized_weights'].get(name, old_weight)
                        blended = 0.7 * old_weight + 0.3 * new_weight
                        blended_weights[name] = blended
                    # Normalize blended weights
                    total = sum(blended_weights.values())
                    if total > 0:
                        return {k: v/total for k, v in blended_weights.items()}
        # Return current weights if update not possible
        return current_weights
def get_default_weights() -> Dict[str, float]:
    """Get default parameter weights."""
    return {
        'seismic': 0.20,
        'deformation': 0.15,
        'hydrogeological': 0.12,
        'electrical': 0.10,
        'magnetic': 0.10,
        'instability': 0.15,
        'stress': 0.10,
        'rock_properties': 0.08,
    }