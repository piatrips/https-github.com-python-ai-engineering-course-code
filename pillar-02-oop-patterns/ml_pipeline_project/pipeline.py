"""
ML Pipeline Project - Main Pipeline Orchestrator

This module demonstrates the complete integration of preprocessing, training,
and evaluation components using the composition pattern.
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

try:
    # Try relative imports (when used as package)
    from .preprocessing import PreprocessingPipeline, BasePreprocessor
    from .training import BaseTrainer
    from .evaluation import Evaluator, EvaluationResults, ResidualAnalyzer
except ImportError:
    # Fall back to absolute imports (when run as script)
    from preprocessing import PreprocessingPipeline, BasePreprocessor
    from training import BaseTrainer
    from evaluation import Evaluator, EvaluationResults, ResidualAnalyzer


@dataclass
class PipelineConfig:
    """
    Configuration for the ML pipeline using dataclasses.
    Demonstrates use of dataclasses for clean configuration management.
    """
    name: str
    random_seed: int = 42
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    verbose: bool = True
    save_predictions: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        total = self.train_split + self.val_split + self.test_split
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Splits must sum to 1.0, got {total}")


class CompletePipeline:
    """
    Complete ML pipeline using composition pattern.
    
    This class demonstrates:
    - Composition: Uses preprocessing, training, and evaluation components
    - Magic methods: __call__ to make the pipeline callable
    - Clean separation of concerns
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.preprocessor: Optional[PreprocessingPipeline] = None
        self.trainer: Optional[BaseTrainer] = None
        self.evaluator: Evaluator = Evaluator()
        self.residual_analyzer: ResidualAnalyzer = ResidualAnalyzer()
        
        self.is_fitted = False
        self.results = {}
        
        # Set random seed
        np.random.seed(self.config.random_seed)
    
    def set_preprocessor(self, preprocessor: PreprocessingPipeline) -> 'CompletePipeline':
        """Set the preprocessing pipeline."""
        self.preprocessor = preprocessor
        return self
    
    def set_trainer(self, trainer: BaseTrainer) -> 'CompletePipeline':
        """Set the model trainer."""
        self.trainer = trainer
        return self
    
    def _log(self, message: str):
        """Internal logging method."""
        if self.config.verbose:
            print(message)
    
    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Split data into train, validation, and test sets."""
        n_samples = X.shape[0]
        
        # Calculate split indices
        train_end = int(n_samples * self.config.train_split)
        val_end = train_end + int(n_samples * self.config.val_split)
        
        # Create shuffled indices
        indices = np.random.permutation(n_samples)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        return {
            'X_train': X[train_idx],
            'y_train': y[train_idx],
            'X_val': X[val_idx],
            'y_val': y[val_idx],
            'X_test': X[test_idx],
            'y_test': y[test_idx],
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CompletePipeline':
        """
        Fit the complete pipeline.
        
        Steps:
        1. Split data
        2. Fit preprocessor on training data
        3. Transform all splits
        4. Train model on processed training data
        5. Evaluate on validation set
        """
        if self.trainer is None:
            raise ValueError("Trainer must be set before fitting")
        
        self._log("=" * 60)
        self._log(f"Fitting Pipeline: {self.config.name}")
        self._log("=" * 60)
        
        # Step 1: Split data
        self._log("\n1. Splitting data...")
        splits = self._split_data(X, y)
        self._log(f"   Train: {splits['X_train'].shape[0]} samples")
        self._log(f"   Val:   {splits['X_val'].shape[0]} samples")
        self._log(f"   Test:  {splits['X_test'].shape[0]} samples")
        
        # Step 2 & 3: Preprocess
        if self.preprocessor is not None:
            self._log("\n2. Fitting preprocessor...")
            self.preprocessor.fit(splits['X_train'])
            self._log(f"   Steps: {self.preprocessor.get_step_names()}")
            
            self._log("\n3. Transforming data...")
            X_train_processed = self.preprocessor.transform(splits['X_train'])
            X_val_processed = self.preprocessor.transform(splits['X_val'])
            X_test_processed = self.preprocessor.transform(splits['X_test'])
            
            self._log(f"   Original shape: {splits['X_train'].shape}")
            self._log(f"   Processed shape: {X_train_processed.shape}")
        else:
            self._log("\n2-3. No preprocessing")
            X_train_processed = splits['X_train']
            X_val_processed = splits['X_val']
            X_test_processed = splits['X_test']
        
        # Step 4: Train model
        self._log(f"\n4. Training {self.trainer.name}...")
        self.trainer.train(X_train_processed, splits['y_train'])
        self._log("   Training complete!")
        
        # Step 5: Evaluate
        self._log("\n5. Evaluating on validation set...")
        y_val_pred = self.trainer.predict(X_val_processed)
        val_results = self.evaluator.evaluate(
            splits['y_val'], y_val_pred,
            model_name=self.trainer.name,
            dataset_name="Validation"
        )
        
        # Also evaluate on training and test sets
        y_train_pred = self.trainer.predict(X_train_processed)
        train_results = self.evaluator.evaluate(
            splits['y_train'], y_train_pred,
            model_name=self.trainer.name,
            dataset_name="Training"
        )
        
        y_test_pred = self.trainer.predict(X_test_processed)
        test_results = self.evaluator.evaluate(
            splits['y_test'], y_test_pred,
            model_name=self.trainer.name,
            dataset_name="Test"
        )
        
        # Store results
        self.results = {
            'train': train_results,
            'validation': val_results,
            'test': test_results,
        }
        
        # Residual analysis on test set
        self.residual_analyzer.analyze(splits['y_test'], y_test_pred)
        
        self.is_fitted = True
        
        self._log("\n" + "=" * 60)
        self._log("Pipeline fitting complete!")
        self._log("=" * 60)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        # Preprocess if needed
        if self.preprocessor is not None:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X
        
        return self.trainer.predict(X_processed)
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Make the pipeline callable."""
        return self.predict(X)
    
    def get_results_summary(self) -> str:
        """Get a summary of all results."""
        if not self.results:
            return "No results available. Fit the pipeline first."
        
        lines = [
            "=" * 60,
            f"Pipeline Results: {self.config.name}",
            "=" * 60,
        ]
        
        for dataset_name, results in self.results.items():
            lines.append(f"\n{dataset_name.upper()}:")
            for metric, value in results.metrics.items():
                lines.append(f"  {metric:10s}: {value:.6f}")
        
        lines.append("\n" + self.residual_analyzer.summary())
        
        return "\n".join(lines)
    
    def summary(self) -> str:
        """Generate complete pipeline summary."""
        lines = [
            "=" * 60,
            f"ML Pipeline: {self.config.name}",
            "=" * 60,
            f"Configuration:",
            f"  Train/Val/Test split: {self.config.train_split}/{self.config.val_split}/{self.config.test_split}",
            f"  Random seed: {self.config.random_seed}",
        ]
        
        if self.preprocessor:
            lines.append(f"\nPreprocessing steps:")
            for step in self.preprocessor.get_step_names():
                lines.append(f"  - {step}")
        else:
            lines.append(f"\nPreprocessing: None")
        
        if self.trainer:
            lines.append(f"\nModel: {self.trainer.name}")
        else:
            lines.append(f"\nModel: Not set")
        
        lines.append(f"\nFitted: {self.is_fitted}")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def create_sample_pipeline() -> CompletePipeline:
    """
    Create a sample pipeline with common configurations.
    This is a factory function demonstrating a design pattern.
    """
    try:
        # Try relative imports (when used as package)
        from .preprocessing import (
            MissingValueImputer, OutlierClipper, 
            StandardScaler, PreprocessingPipeline
        )
        from .training import LinearRegressionTrainer
    except ImportError:
        # Fall back to absolute imports (when run as script)
        from preprocessing import (
            MissingValueImputer, OutlierClipper, 
            StandardScaler, PreprocessingPipeline
        )
        from training import LinearRegressionTrainer
    
    # Configuration
    config = PipelineConfig(
        name="Sample ML Pipeline",
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        verbose=True
    )
    
    # Preprocessing
    preprocessing = PreprocessingPipeline()
    preprocessing.add_step(MissingValueImputer(strategy="mean"))
    preprocessing.add_step(OutlierClipper(method="zscore", threshold=3.0))
    preprocessing.add_step(StandardScaler())
    
    # Model
    trainer = LinearRegressionTrainer(fit_intercept=True, regularization=0.1)
    
    # Create pipeline
    pipeline = CompletePipeline(config)
    pipeline.set_preprocessor(preprocessing)
    pipeline.set_trainer(trainer)
    
    return pipeline


if __name__ == "__main__":
    print("=" * 60)
    print("Complete ML Pipeline Demonstration")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    
    # Add some noise and outliers
    X[::50] += np.random.randn(10, n_features) * 5  # Outliers
    X[10:15, 0] = np.nan  # Missing values
    
    # True relationship
    true_coef = np.random.randn(n_features)
    y = X @ true_coef + 5.0
    y = np.nan_to_num(y)  # Handle NaN in y from NaN in X
    y += np.random.randn(n_samples) * 0.5  # Add noise
    
    print(f"\nDataset:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Missing values: {np.sum(np.isnan(X))}")
    
    # Create and fit pipeline
    print("\n" + "=" * 60)
    pipeline = create_sample_pipeline()
    print(pipeline.summary())
    
    print("\n" + "=" * 60)
    print("Fitting pipeline...")
    print("=" * 60)
    pipeline.fit(X, y)
    
    # Show results
    print("\n" + pipeline.get_results_summary())
    
    # Make predictions on new data
    print("\n" + "=" * 60)
    print("Making predictions on new data...")
    print("=" * 60)
    X_new = np.random.randn(10, n_features)
    predictions = pipeline(X_new)  # Using __call__ magic method
    
    print(f"New data shape: {X_new.shape}")
    print(f"Predictions: {predictions}")
    
    print("\n" + "=" * 60)
    print("Pipeline demonstration complete!")
    print("=" * 60)
