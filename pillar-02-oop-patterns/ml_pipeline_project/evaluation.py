"""
ML Pipeline Project - Evaluation Module

This module contains evaluation and metrics components.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class EvaluationResults:
    """
    Dataclass to store evaluation results.
    Demonstrates use of dataclasses for structured data.
    """
    model_name: str
    dataset_name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    ground_truth: Optional[np.ndarray] = None
    
    def add_metric(self, name: str, value: float):
        """Add a metric to the results."""
        self.metrics[name] = value
    
    def get_metric(self, name: str) -> Optional[float]:
        """Get a specific metric."""
        return self.metrics.get(name)
    
    def summary(self) -> str:
        """Generate a summary string."""
        lines = [
            "=" * 60,
            f"Evaluation Results: {self.model_name} on {self.dataset_name}",
            "=" * 60,
        ]
        
        for metric, value in self.metrics.items():
            lines.append(f"  {metric.upper():15s}: {value:.6f}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def __repr__(self):
        """String representation."""
        return f"EvaluationResults(model={self.model_name}, metrics={list(self.metrics.keys())})"


class BaseMetric(ABC):
    """Abstract base class for metrics."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the metric."""
        pass
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Make metric callable."""
        return self.compute(y_true, y_pred)


class MeanSquaredError(BaseMetric):
    """Mean Squared Error metric."""
    
    def __init__(self):
        super().__init__("mse")
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute MSE."""
        return np.mean((y_true - y_pred) ** 2)


class RootMeanSquaredError(BaseMetric):
    """Root Mean Squared Error metric."""
    
    def __init__(self):
        super().__init__("rmse")
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute RMSE."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


class MeanAbsoluteError(BaseMetric):
    """Mean Absolute Error metric."""
    
    def __init__(self):
        super().__init__("mae")
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute MAE."""
        return np.mean(np.abs(y_true - y_pred))


class R2Score(BaseMetric):
    """R-squared (coefficient of determination) metric."""
    
    def __init__(self):
        super().__init__("r2")
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R2 score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)


class MaxError(BaseMetric):
    """Maximum absolute error metric."""
    
    def __init__(self):
        super().__init__("max_error")
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute maximum error."""
        return np.max(np.abs(y_true - y_pred))


class MedianAbsoluteError(BaseMetric):
    """Median Absolute Error metric."""
    
    def __init__(self):
        super().__init__("median_ae")
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute median absolute error."""
        return np.median(np.abs(y_true - y_pred))


class Evaluator:
    """
    Main evaluator class that uses composition to combine multiple metrics.
    Demonstrates composition pattern and magic methods.
    """
    
    def __init__(self, metrics: Optional[List[BaseMetric]] = None):
        self.metrics = metrics if metrics is not None else self._default_metrics()
        self.results_history = []
    
    @staticmethod
    def _default_metrics() -> List[BaseMetric]:
        """Get default set of metrics."""
        return [
            MeanSquaredError(),
            RootMeanSquaredError(),
            MeanAbsoluteError(),
            R2Score(),
        ]
    
    def add_metric(self, metric: BaseMetric) -> 'Evaluator':
        """Add a metric to the evaluator."""
        self.metrics.append(metric)
        return self
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 model_name: str = "Model", dataset_name: str = "Dataset") -> EvaluationResults:
        """
        Evaluate predictions using all metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            model_name: Name of the model
            dataset_name: Name of the dataset
            
        Returns:
            EvaluationResults object containing all metrics
        """
        results = EvaluationResults(
            model_name=model_name,
            dataset_name=dataset_name,
            predictions=y_pred,
            ground_truth=y_true
        )
        
        for metric in self.metrics:
            value = metric(y_true, y_pred)
            results.add_metric(metric.name, value)
        
        self.results_history.append(results)
        return results
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray,
                 model_name: str = "Model", dataset_name: str = "Dataset") -> EvaluationResults:
        """Make evaluator callable."""
        return self.evaluate(y_true, y_pred, model_name, dataset_name)
    
    def compare_results(self, results_list: List[EvaluationResults],
                        metric_name: str = "rmse") -> str:
        """Compare multiple evaluation results."""
        lines = [
            "=" * 60,
            f"Model Comparison ({metric_name.upper()})",
            "=" * 60,
        ]
        
        # Sort by metric value
        sorted_results = sorted(
            results_list,
            key=lambda r: r.get_metric(metric_name) or float('inf')
        )
        
        for i, result in enumerate(sorted_results, 1):
            metric_value = result.get_metric(metric_name)
            if metric_value is not None:
                lines.append(f"{i}. {result.model_name:20s}: {metric_value:.6f}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class ResidualAnalyzer:
    """Analyze prediction residuals."""
    
    def __init__(self):
        self.residuals = None
        self.statistics = {}
    
    def analyze(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Perform residual analysis."""
        self.residuals = y_true - y_pred
        
        self.statistics = {
            'mean_residual': np.mean(self.residuals),
            'std_residual': np.std(self.residuals),
            'min_residual': np.min(self.residuals),
            'max_residual': np.max(self.residuals),
            'median_residual': np.median(self.residuals),
            'q25_residual': np.percentile(self.residuals, 25),
            'q75_residual': np.percentile(self.residuals, 75),
        }
        
        return self.statistics
    
    def summary(self) -> str:
        """Generate residual analysis summary."""
        if not self.statistics:
            return "No analysis performed yet."
        
        lines = [
            "=" * 60,
            "Residual Analysis",
            "=" * 60,
            f"Mean:    {self.statistics['mean_residual']:10.6f}",
            f"Std:     {self.statistics['std_residual']:10.6f}",
            f"Min:     {self.statistics['min_residual']:10.6f}",
            f"Max:     {self.statistics['max_residual']:10.6f}",
            f"Median:  {self.statistics['median_residual']:10.6f}",
            f"Q25:     {self.statistics['q25_residual']:10.6f}",
            f"Q75:     {self.statistics['q75_residual']:10.6f}",
            "=" * 60,
        ]
        
        return "\n".join(lines)


class CrossValidator:
    """Perform cross-validation evaluation."""
    
    def __init__(self, n_folds: int = 5, metrics: Optional[List[BaseMetric]] = None):
        self.n_folds = n_folds
        self.evaluator = Evaluator(metrics)
        self.fold_results = []
    
    def cross_validate(self, trainer, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Perform k-fold cross-validation.
        
        Args:
            trainer: Model trainer (must have train and predict methods)
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary of average metrics across folds
        """
        n_samples = X.shape[0]
        fold_size = n_samples // self.n_folds
        
        self.fold_results = []
        
        for fold in range(self.n_folds):
            # Split data
            val_start = fold * fold_size
            val_end = val_start + fold_size
            
            val_indices = list(range(val_start, val_end))
            train_indices = list(range(0, val_start)) + list(range(val_end, n_samples))
            
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_val = X[val_indices]
            y_val = y[val_indices]
            
            # Train and evaluate
            trainer.train(X_train, y_train)
            y_pred = trainer.predict(X_val)
            
            results = self.evaluator.evaluate(
                y_val, y_pred,
                model_name=trainer.name,
                dataset_name=f"Fold {fold+1}"
            )
            self.fold_results.append(results)
        
        # Compute average metrics
        avg_metrics = {}
        metric_names = self.fold_results[0].metrics.keys()
        
        for metric_name in metric_names:
            values = [r.get_metric(metric_name) for r in self.fold_results]
            avg_metrics[metric_name] = np.mean(values)
            avg_metrics[f"{metric_name}_std"] = np.std(values)
        
        return avg_metrics
    
    def summary(self) -> str:
        """Generate cross-validation summary."""
        if not self.fold_results:
            return "No cross-validation performed yet."
        
        lines = [
            "=" * 60,
            f"{self.n_folds}-Fold Cross-Validation Results",
            "=" * 60,
        ]
        
        # Show results for each fold
        for result in self.fold_results:
            lines.append(f"\n{result.dataset_name}:")
            for metric, value in result.metrics.items():
                lines.append(f"  {metric:10s}: {value:.6f}")
        
        # Show averages
        metric_names = self.fold_results[0].metrics.keys()
        lines.append("\nAverage across folds:")
        
        for metric_name in metric_names:
            values = [r.get_metric(metric_name) for r in self.fold_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            lines.append(f"  {metric_name:10s}: {mean_val:.6f} ± {std_val:.6f}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


if __name__ == "__main__":
    # Test evaluation components
    print("Testing Evaluation Module")
    print("=" * 60)
    
    # Generate test predictions
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.3  # Add some noise
    
    # Test evaluator
    evaluator = Evaluator()
    evaluator.add_metric(MaxError())
    evaluator.add_metric(MedianAbsoluteError())
    
    results = evaluator(y_true, y_pred, model_name="TestModel", dataset_name="TestData")
    print(results.summary())
    
    # Test residual analyzer
    print("\n")
    analyzer = ResidualAnalyzer()
    analyzer.analyze(y_true, y_pred)
    print(analyzer.summary())
    
    # Test comparison
    print("\n")
    y_pred2 = y_true + np.random.randn(100) * 0.5
    results2 = evaluator(y_true, y_pred2, model_name="TestModel2", dataset_name="TestData")
    
    print(evaluator.compare_results([results, results2], metric_name="rmse"))
