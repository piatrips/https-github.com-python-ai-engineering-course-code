"""
ML Pipeline Project - Preprocessing Module

This module contains all preprocessing components for the ML pipeline.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BasePreprocessor(ABC):
    """Abstract base class for all preprocessors."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BasePreprocessor':
        """Fit the preprocessor to the data."""
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data."""
        pass
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get preprocessor parameters."""
        return {"name": self.name, "is_fitted": self.is_fitted}


class StandardScaler(BasePreprocessor):
    """Standardize features by removing mean and scaling to unit variance."""
    
    def __init__(self):
        super().__init__("StandardScaler")
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'StandardScaler':
        """Compute mean and std for standardization."""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0) + 1e-8  # Avoid division by zero
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize features."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before transform")
        return (X - self.mean_) / self.std_
    
    def get_params(self) -> Dict[str, Any]:
        """Get scaler parameters."""
        params = super().get_params()
        params.update({
            "mean": self.mean_,
            "std": self.std_
        })
        return params


class MinMaxScaler(BasePreprocessor):
    """Scale features to a given range (default [0, 1])."""
    
    def __init__(self, feature_range: tuple = (0, 1)):
        super().__init__("MinMaxScaler")
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.scale_ = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'MinMaxScaler':
        """Compute min and max for scaling."""
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.scale_ = self.max_ - self.min_
        self.scale_[self.scale_ == 0] = 1.0  # Avoid division by zero
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features to range."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before transform")
        
        # Scale to [0, 1]
        X_scaled = (X - self.min_) / self.scale_
        
        # Scale to desired range
        min_val, max_val = self.feature_range
        X_scaled = X_scaled * (max_val - min_val) + min_val
        
        return X_scaled


class MissingValueImputer(BasePreprocessor):
    """Impute missing values using mean, median, or constant."""
    
    def __init__(self, strategy: str = "mean", fill_value: float = 0.0):
        super().__init__("MissingValueImputer")
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'MissingValueImputer':
        """Compute statistics for imputation."""
        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == "constant":
            self.statistics_ = np.full(X.shape[1], self.fill_value)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Impute missing values."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before transform")
        
        X_imputed = X.copy()
        mask = np.isnan(X_imputed)
        
        for i in range(X.shape[1]):
            X_imputed[mask[:, i], i] = self.statistics_[i]
        
        return X_imputed


class OutlierClipper(BasePreprocessor):
    """Clip outliers based on z-score or IQR method."""
    
    def __init__(self, method: str = "zscore", threshold: float = 3.0):
        super().__init__("OutlierClipper")
        self.method = method
        self.threshold = threshold
        self.lower_bound_ = None
        self.upper_bound_ = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'OutlierClipper':
        """Compute bounds for clipping."""
        if self.method == "zscore":
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            self.lower_bound_ = mean - self.threshold * std
            self.upper_bound_ = mean + self.threshold * std
        elif self.method == "iqr":
            q1 = np.percentile(X, 25, axis=0)
            q3 = np.percentile(X, 75, axis=0)
            iqr = q3 - q1
            self.lower_bound_ = q1 - self.threshold * iqr
            self.upper_bound_ = q3 + self.threshold * iqr
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip outliers."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before transform")
        
        return np.clip(X, self.lower_bound_, self.upper_bound_)


class FeatureEngineer(BasePreprocessor):
    """Create polynomial features."""
    
    def __init__(self, degree: int = 2):
        super().__init__("FeatureEngineer")
        self.degree = degree
        self.n_input_features_ = None
        self.n_output_features_ = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'FeatureEngineer':
        """Compute feature counts."""
        self.n_input_features_ = X.shape[1]
        # For degree 2: original + squares + interactions
        self.n_output_features_ = X.shape[1] + X.shape[1] + (X.shape[1] * (X.shape[1] - 1)) // 2
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Create polynomial features."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before transform")
        
        features = [X]  # Original features
        
        if self.degree >= 2:
            # Add squared features
            features.append(X ** 2)
            
            # Add interaction features
            n_features = X.shape[1]
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                    features.append(interaction)
        
        return np.hstack(features)


class PreprocessingPipeline:
    """
    A pipeline that chains multiple preprocessing steps.
    Demonstrates composition pattern.
    """
    
    def __init__(self, steps: list = None):
        self.steps = steps if steps is not None else []
        self.is_fitted = False
    
    def add_step(self, preprocessor: BasePreprocessor) -> 'PreprocessingPipeline':
        """Add a preprocessing step."""
        self.steps.append(preprocessor)
        return self
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'PreprocessingPipeline':
        """Fit all preprocessing steps sequentially."""
        X_transformed = X.copy()
        
        for preprocessor in self.steps:
            preprocessor.fit(X_transformed, y)
            X_transformed = preprocessor.transform(X_transformed)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply all preprocessing steps sequentially."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        X_transformed = X.copy()
        for preprocessor in self.steps:
            X_transformed = preprocessor.transform(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X, y).transform(X)
    
    def get_step_names(self) -> list:
        """Get names of all preprocessing steps."""
        return [step.name for step in self.steps]
    
    def __repr__(self):
        """String representation."""
        return f"PreprocessingPipeline(steps={self.get_step_names()})"


if __name__ == "__main__":
    # Test preprocessing components
    print("Testing Preprocessing Module")
    print("=" * 60)
    
    # Generate test data with outliers and missing values
    np.random.seed(42)
    X = np.random.randn(100, 5)
    X[::10] += np.random.randn(10, 5) * 5  # Add outliers
    X[5:10, 0] = np.nan  # Add missing values
    
    print(f"Original data shape: {X.shape}")
    print(f"Missing values: {np.sum(np.isnan(X))}")
    print(f"Data range: [{np.nanmin(X):.2f}, {np.nanmax(X):.2f}]")
    
    # Create pipeline
    pipeline = PreprocessingPipeline()
    pipeline.add_step(MissingValueImputer(strategy="mean"))
    pipeline.add_step(OutlierClipper(method="zscore", threshold=3.0))
    pipeline.add_step(StandardScaler())
    
    print(f"\nPipeline: {pipeline}")
    
    # Fit and transform
    X_processed = pipeline.fit_transform(X)
    
    print(f"\nProcessed data shape: {X_processed.shape}")
    print(f"Missing values: {np.sum(np.isnan(X_processed))}")
    print(f"Data range: [{X_processed.min():.2f}, {X_processed.max():.2f}]")
    print(f"Mean: {X_processed.mean(axis=0)}")
    print(f"Std: {X_processed.std(axis=0)}")
