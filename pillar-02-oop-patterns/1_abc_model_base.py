"""
Example 1: Abstract Base Classes (ABC) for Model Base Classes

This example demonstrates how to use ABC to create a base class for ML models,
ensuring that all derived models implement required methods.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict


class BaseModel(ABC):
    """
    Abstract base class for ML models.
    All models must implement train, predict, and save methods.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on the given data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the given data."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to disk."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters (optional to override)."""
        return {"name": self.name, "is_trained": self.is_trained}


class LinearRegressionModel(BaseModel):
    """Concrete implementation of a linear regression model."""
    
    def __init__(self, name: str = "LinearRegression"):
        super().__init__(name)
        self.coefficients = None
        self.intercept = None
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train using ordinary least squares."""
        # Add bias term
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Solve normal equations: (X^T X)^-1 X^T y
        self.coefficients = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]
        
        self.is_trained = True
        print(f"{self.name} trained successfully!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return X @ self.coefficients + self.intercept
    
    def save(self, path: str) -> None:
        """Save model parameters to a file."""
        np.savez(path, coefficients=self.coefficients, intercept=self.intercept)
        print(f"Model saved to {path}")
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            "coefficients": self.coefficients,
            "intercept": self.intercept
        })
        return params


class DecisionTreeModel(BaseModel):
    """Concrete implementation of a simple decision tree model."""
    
    def __init__(self, name: str = "DecisionTree", max_depth: int = 3):
        super().__init__(name)
        self.max_depth = max_depth
        self.tree = None
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train a simple decision tree (simplified implementation)."""
        # Simplified: just store the mean as prediction
        # In a real implementation, this would build a tree structure
        self.tree = {"mean": np.mean(y), "depth": self.max_depth}
        self.is_trained = True
        print(f"{self.name} trained successfully with max_depth={self.max_depth}!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained tree."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Simplified: return mean for all samples
        return np.full(X.shape[0], self.tree["mean"])
    
    def save(self, path: str) -> None:
        """Save the tree structure to a file."""
        np.savez(path, tree=self.tree)
        print(f"Model saved to {path}")
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            "max_depth": self.max_depth,
            "tree": self.tree
        })
        return params


def demonstrate_abc_pattern():
    """Demonstrate the ABC pattern with different model implementations."""
    print("=" * 60)
    print("ABC Pattern for Model Base Classes")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    X_train = np.random.randn(100, 3)
    y_train = X_train @ np.array([1.5, -2.0, 0.5]) + 3.0 + np.random.randn(100) * 0.1
    X_test = np.random.randn(20, 3)
    
    # Create and use different models
    models = [
        LinearRegressionModel(),
        DecisionTreeModel(max_depth=5)
    ]
    
    for model in models:
        print(f"\n{'-' * 60}")
        print(f"Training {model.name}...")
        print(f"Initial params: {model.get_params()}")
        
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        
        print(f"Predictions (first 5): {predictions[:5]}")
        print(f"Model params: {model.get_params()}")
        
        # Save model
        model.save(f"/tmp/{model.name.lower()}_model.npz")
    
    print("\n" + "=" * 60)
    print("Benefits of ABC:")
    print("- Enforces interface consistency across models")
    print("- Prevents instantiation of incomplete implementations")
    print("- Makes code more maintainable and extensible")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_abc_pattern()
