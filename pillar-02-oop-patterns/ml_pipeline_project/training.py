"""
ML Pipeline Project - Training Module

This module contains model training components.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable


class BaseTrainer(ABC):
    """Abstract base class for model trainers."""
    
    def __init__(self, name: str):
        self.name = name
        self.model_params = None
        self.is_trained = False
        self.training_history = []
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseTrainer':
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "name": self.name,
            "is_trained": self.is_trained,
            "params": self.model_params
        }


class LinearRegressionTrainer(BaseTrainer):
    """Train a linear regression model using normal equations."""
    
    def __init__(self, fit_intercept: bool = True, regularization: float = 0.0):
        super().__init__("LinearRegression")
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.coefficients_ = None
        self.intercept_ = None
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'LinearRegressionTrainer':
        """Train using ordinary least squares with optional L2 regularization."""
        n_samples, n_features = X.shape
        
        if self.fit_intercept:
            # Add bias term
            X_with_bias = np.column_stack([np.ones(n_samples), X])
        else:
            X_with_bias = X
        
        # Solve: (X^T X + lambda*I)^-1 X^T y
        XtX = X_with_bias.T @ X_with_bias
        
        if self.regularization > 0:
            # Add regularization (don't regularize bias term)
            reg_matrix = self.regularization * np.eye(XtX.shape[0])
            if self.fit_intercept:
                reg_matrix[0, 0] = 0  # Don't regularize intercept
            XtX = XtX + reg_matrix
        
        Xty = X_with_bias.T @ y
        params = np.linalg.solve(XtX, Xty)
        
        if self.fit_intercept:
            self.intercept_ = params[0]
            self.coefficients_ = params[1:]
        else:
            self.intercept_ = 0.0
            self.coefficients_ = params
        
        self.model_params = {
            "coefficients": self.coefficients_,
            "intercept": self.intercept_
        }
        self.is_trained = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return X @ self.coefficients_ + self.intercept_


class GradientDescentTrainer(BaseTrainer):
    """Train a linear model using gradient descent."""
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000,
                 regularization: float = 0.0, verbose: bool = False):
        super().__init__("GradientDescent")
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.verbose = verbose
        self.coefficients_ = None
        self.intercept_ = None
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'GradientDescentTrainer':
        """Train using gradient descent."""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.coefficients_ = np.zeros(n_features)
        self.intercept_ = 0.0
        
        # Training loop
        for iteration in range(self.n_iterations):
            # Predictions
            y_pred = X @ self.coefficients_ + self.intercept_
            
            # Compute gradients
            error = y_pred - y
            grad_coef = (2 / n_samples) * (X.T @ error) + self.regularization * self.coefficients_
            grad_intercept = (2 / n_samples) * np.sum(error)
            
            # Update parameters
            self.coefficients_ -= self.learning_rate * grad_coef
            self.intercept_ -= self.learning_rate * grad_intercept
            
            # Track loss
            loss = np.mean(error ** 2)
            self.training_history.append(loss)
            
            if self.verbose and (iteration % 100 == 0 or iteration == self.n_iterations - 1):
                print(f"  Iteration {iteration}: Loss = {loss:.6f}")
        
        self.model_params = {
            "coefficients": self.coefficients_,
            "intercept": self.intercept_,
            "final_loss": self.training_history[-1]
        }
        self.is_trained = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return X @ self.coefficients_ + self.intercept_


class RidgeRegressionTrainer(BaseTrainer):
    """Train ridge regression (L2 regularization)."""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__("RidgeRegression")
        self.alpha = alpha
        self.coefficients_ = None
        self.intercept_ = None
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'RidgeRegressionTrainer':
        """Train using ridge regression."""
        # Center the data
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        
        X_centered = X - X_mean
        y_centered = y - y_mean
        
        # Solve: (X^T X + alpha*I)^-1 X^T y
        n_features = X.shape[1]
        XtX = X_centered.T @ X_centered
        reg_matrix = self.alpha * np.eye(n_features)
        
        self.coefficients_ = np.linalg.solve(XtX + reg_matrix, X_centered.T @ y_centered)
        self.intercept_ = y_mean - X_mean @ self.coefficients_
        
        self.model_params = {
            "coefficients": self.coefficients_,
            "intercept": self.intercept_,
            "alpha": self.alpha
        }
        self.is_trained = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return X @ self.coefficients_ + self.intercept_


class EnsembleTrainer(BaseTrainer):
    """Train an ensemble of models."""
    
    def __init__(self, trainers: list):
        super().__init__("Ensemble")
        self.trainers = trainers
        self.weights = None
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'EnsembleTrainer':
        """Train all models in the ensemble."""
        print(f"\nTraining ensemble with {len(self.trainers)} models...")
        
        for i, trainer in enumerate(self.trainers):
            print(f"  Training model {i+1}/{len(self.trainers)}: {trainer.name}")
            trainer.train(X, y)
        
        # Equal weights for all models
        self.weights = np.ones(len(self.trainers)) / len(self.trainers)
        self.is_trained = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using weighted average."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        predictions = np.array([trainer.predict(X) for trainer in self.trainers])
        return np.average(predictions, axis=0, weights=self.weights)
    
    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each model."""
        return {
            trainer.name: trainer.predict(X)
            for trainer in self.trainers
        }


class ModelSelector:
    """
    Select the best model from multiple candidates using cross-validation.
    Demonstrates the strategy pattern.
    """
    
    def __init__(self, scoring_fn: Optional[Callable] = None):
        self.scoring_fn = scoring_fn or self._default_scorer
        self.best_trainer = None
        self.best_score = float('inf')
        self.scores = {}
    
    @staticmethod
    def _default_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Default scoring function (MSE)."""
        return np.mean((y_true - y_pred) ** 2)
    
    def select(self, trainers: list, X: np.ndarray, y: np.ndarray,
               n_folds: int = 5) -> BaseTrainer:
        """Select best model using k-fold cross-validation."""
        print(f"\nModel Selection with {n_folds}-fold cross-validation")
        print("-" * 60)
        
        n_samples = X.shape[0]
        fold_size = n_samples // n_folds
        
        for trainer in trainers:
            fold_scores = []
            
            for fold in range(n_folds):
                # Split data
                val_start = fold * fold_size
                val_end = val_start + fold_size
                
                val_indices = list(range(val_start, val_end))
                train_indices = list(range(0, val_start)) + list(range(val_end, n_samples))
                
                X_train_fold = X[train_indices]
                y_train_fold = y[train_indices]
                X_val_fold = X[val_indices]
                y_val_fold = y[val_indices]
                
                # Train and evaluate
                trainer.train(X_train_fold, y_train_fold)
                y_pred = trainer.predict(X_val_fold)
                score = self.scoring_fn(y_val_fold, y_pred)
                fold_scores.append(score)
            
            avg_score = np.mean(fold_scores)
            self.scores[trainer.name] = avg_score
            
            print(f"  {trainer.name:20s}: {avg_score:.6f}")
            
            if avg_score < self.best_score:
                self.best_score = avg_score
                self.best_trainer = trainer
        
        print(f"\nBest model: {self.best_trainer.name} (score: {self.best_score:.6f})")
        
        # Retrain on full dataset
        self.best_trainer.train(X, y)
        
        return self.best_trainer


if __name__ == "__main__":
    # Test training components
    print("Testing Training Module")
    print("=" * 60)
    
    # Generate test data
    np.random.seed(42)
    X = np.random.randn(200, 5)
    true_coef = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
    y = X @ true_coef + 2.0 + np.random.randn(200) * 0.5
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"True coefficients: {true_coef}")
    
    # Test different trainers
    trainers = [
        LinearRegressionTrainer(),
        RidgeRegressionTrainer(alpha=1.0),
        GradientDescentTrainer(learning_rate=0.01, n_iterations=500, verbose=True)
    ]
    
    print("\n" + "=" * 60)
    print("Testing Model Selection")
    selector = ModelSelector()
    best_model = selector.select(trainers, X, y, n_folds=5)
    
    print(f"\nBest model coefficients: {best_model.coefficients_}")
    print(f"Best model intercept: {best_model.intercept_:.4f}")
