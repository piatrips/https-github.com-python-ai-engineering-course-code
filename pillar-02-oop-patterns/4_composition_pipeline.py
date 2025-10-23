"""
Example 4: Composition Pattern for ML Pipeline

This example demonstrates the composition pattern, where complex functionality
is built by combining simpler, reusable components rather than using inheritance.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod


# Component interfaces
class Preprocessor(ABC):
    """Abstract base class for preprocessing components."""
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'Preprocessor':
        """Fit the preprocessor to the data."""
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data."""
        pass
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class Scaler(Preprocessor):
    """Standard scaler component."""
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X: np.ndarray) -> 'Scaler':
        """Calculate mean and standard deviation."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8  # Avoid division by zero
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize the data."""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before transform")
        return (X - self.mean) / self.std


class OutlierRemover(Preprocessor):
    """Component to remove outliers."""
    
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.mean = None
        self.std = None
    
    def fit(self, X: np.ndarray) -> 'OutlierRemover':
        """Calculate mean and std for outlier detection."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip outliers."""
        if self.mean is None or self.std is None:
            raise ValueError("OutlierRemover must be fitted before transform")
        
        z_scores = np.abs((X - self.mean) / self.std)
        # Clip values beyond threshold
        X_clipped = X.copy()
        mask = z_scores > self.threshold
        
        # For each outlier, clip to threshold
        for i in range(X.shape[1]):
            col_mask = mask[:, i]
            upper_bound = self.mean[i] + self.threshold * self.std[i]
            lower_bound = self.mean[i] - self.threshold * self.std[i]
            X_clipped[col_mask, i] = np.clip(X[col_mask, i], lower_bound, upper_bound)
        
        return X_clipped


class FeatureSelector(Preprocessor):
    """Component to select top features."""
    
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.selected_indices = None
    
    def fit(self, X: np.ndarray) -> 'FeatureSelector':
        """Select features with highest variance."""
        variances = np.var(X, axis=0)
        self.selected_indices = np.argsort(variances)[-self.n_features:]
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select the top features."""
        if self.selected_indices is None:
            raise ValueError("FeatureSelector must be fitted before transform")
        return X[:, self.selected_indices]


# Composition: Building complex functionality from simple components
class PreprocessingPipeline:
    """
    A preprocessing pipeline using composition.
    
    Instead of inheriting from multiple classes, we compose functionality
    by containing multiple preprocessor components.
    """
    
    def __init__(self, name: str = "Pipeline"):
        self.name = name
        self.steps: List[tuple[str, Preprocessor]] = []
        self.is_fitted = False
    
    def add_step(self, name: str, preprocessor: Preprocessor) -> 'PreprocessingPipeline':
        """Add a preprocessing step to the pipeline."""
        self.steps.append((name, preprocessor))
        return self  # Enable method chaining
    
    def fit(self, X: np.ndarray) -> 'PreprocessingPipeline':
        """Fit all preprocessing steps."""
        X_transformed = X.copy()
        
        for name, preprocessor in self.steps:
            preprocessor.fit(X_transformed)
            X_transformed = preprocessor.transform(X_transformed)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply all preprocessing steps."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        X_transformed = X.copy()
        for name, preprocessor in self.steps:
            X_transformed = preprocessor.transform(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def get_step_names(self) -> List[str]:
        """Get names of all steps in the pipeline."""
        return [name for name, _ in self.steps]


class Model:
    """Simple model component."""
    
    def __init__(self, name: str):
        self.name = name
        self.coefficients = None
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model."""
        # Simple linear regression
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        self.coefficients = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        return X_with_bias @ self.coefficients


class Evaluator:
    """Component for model evaluation."""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        self.metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        return self.metrics
    
    def print_metrics(self):
        """Print evaluation metrics."""
        print("\nEvaluation Metrics:")
        print("-" * 40)
        for metric, value in self.metrics.items():
            print(f"{metric.upper():8s}: {value:.4f}")


# Main ML Pipeline using composition
class MLPipeline:
    """
    Complete ML pipeline using composition pattern.
    
    This class composes preprocessing, model, and evaluation components
    rather than inheriting from them.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.preprocessor: Optional[PreprocessingPipeline] = None
        self.model: Optional[Model] = None
        self.evaluator: Evaluator = Evaluator()
        self.is_fitted = False
    
    def set_preprocessor(self, preprocessor: PreprocessingPipeline) -> 'MLPipeline':
        """Set the preprocessing pipeline."""
        self.preprocessor = preprocessor
        return self
    
    def set_model(self, model: Model) -> 'MLPipeline':
        """Set the model."""
        self.model = model
        return self
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPipeline':
        """Fit the entire pipeline."""
        if self.model is None:
            raise ValueError("Model must be set before fitting")
        
        print(f"\nFitting {self.name}...")
        print("-" * 60)
        
        # Preprocess if preprocessor is set
        if self.preprocessor is not None:
            print("Fitting preprocessor...")
            X_processed = self.preprocessor.fit_transform(X)
            print(f"  Preprocessing steps: {self.preprocessor.get_step_names()}")
            print(f"  Original shape: {X.shape} -> Processed shape: {X_processed.shape}")
        else:
            X_processed = X
            print("No preprocessing applied")
        
        # Train model
        print(f"\nTraining {self.model.name}...")
        self.model.train(X_processed, y)
        print("  Model training complete!")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        # Preprocess if preprocessor is set
        if self.preprocessor is not None:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X
        
        # Predict
        return self.model.predict(X_processed)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the pipeline."""
        predictions = self.predict(X)
        return self.evaluator.evaluate(y, predictions)
    
    def summary(self) -> str:
        """Generate pipeline summary."""
        lines = [
            "=" * 60,
            f"ML Pipeline: {self.name}",
            "=" * 60,
        ]
        
        if self.preprocessor:
            lines.append(f"Preprocessor: {self.preprocessor.get_step_names()}")
        else:
            lines.append("Preprocessor: None")
        
        if self.model:
            lines.append(f"Model: {self.model.name}")
        else:
            lines.append("Model: Not set")
        
        lines.append(f"Fitted: {self.is_fitted}")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def demonstrate_composition_pattern():
    """Demonstrate the composition pattern for ML pipelines."""
    print("=" * 60)
    print("Composition Pattern for ML Pipeline")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    X_train = np.random.randn(200, 10)
    # Add some outliers
    X_train[::20] += np.random.randn(10, 10) * 5
    
    # True coefficients
    true_coef = np.random.randn(10)
    y_train = X_train @ true_coef + np.random.randn(200) * 0.5
    
    X_test = np.random.randn(50, 10)
    y_test = X_test @ true_coef + np.random.randn(50) * 0.5
    
    print(f"\nDataset:")
    print(f"  Training: X={X_train.shape}, y={y_train.shape}")
    print(f"  Testing: X={X_test.shape}, y={y_test.shape}")
    
    # 1. Build preprocessing pipeline using composition
    print("\n\n1. Building Preprocessing Pipeline (Composition)")
    print("-" * 60)
    
    preprocessing = PreprocessingPipeline("Data Preprocessing")
    preprocessing.add_step("outlier_removal", OutlierRemover(threshold=3.0))
    preprocessing.add_step("scaler", Scaler())
    preprocessing.add_step("feature_selection", FeatureSelector(n_features=7))
    
    print(f"Preprocessing steps: {preprocessing.get_step_names()}")
    
    # 2. Build model
    print("\n\n2. Building Model")
    print("-" * 60)
    
    model = Model("LinearRegression")
    print(f"Model: {model.name}")
    
    # 3. Compose the complete pipeline
    print("\n\n3. Composing Complete ML Pipeline")
    print("-" * 60)
    
    pipeline = MLPipeline("Complete ML Pipeline")
    pipeline.set_preprocessor(preprocessing)
    pipeline.set_model(model)
    
    print(pipeline.summary())
    
    # 4. Train the pipeline
    print("\n\n4. Training Pipeline")
    print("-" * 60)
    
    pipeline.fit(X_train, y_train)
    
    # 5. Evaluate
    print("\n\n5. Evaluating Pipeline")
    print("-" * 60)
    
    train_metrics = pipeline.evaluate(X_train, y_train)
    print("Training Set:")
    pipeline.evaluator.print_metrics()
    
    test_metrics = pipeline.evaluate(X_test, y_test)
    print("\nTest Set:")
    pipeline.evaluator.print_metrics()
    
    # 6. Alternative pipeline without preprocessing
    print("\n\n6. Alternative Pipeline (No Preprocessing)")
    print("-" * 60)
    
    simple_pipeline = MLPipeline("Simple Pipeline")
    simple_pipeline.set_model(Model("SimpleLinearRegression"))
    simple_pipeline.fit(X_train, y_train)
    
    simple_metrics = pipeline.evaluate(X_test, y_test)
    print("\nSimple Pipeline Test Set:")
    pipeline.evaluator.print_metrics()
    
    print("\n" + "=" * 60)
    print("Benefits of Composition Pattern:")
    print("- Flexibility: Easy to swap components")
    print("- Reusability: Components can be used in different pipelines")
    print("- Testability: Each component can be tested independently")
    print("- Maintainability: Changes to one component don't affect others")
    print("- Avoids deep inheritance hierarchies")
    print("- 'Favor composition over inheritance' principle")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_composition_pattern()
