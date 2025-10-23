"""
Example 2: Magic Methods (__call__, __getitem__)

This example demonstrates the use of magic methods to create intuitive
and Pythonic interfaces for ML components.
"""

import numpy as np
from typing import Any, List, Union


class TransformPipeline:
    """
    A transformation pipeline using __call__ magic method.
    Makes the object callable like a function.
    """
    
    def __init__(self, name: str = "TransformPipeline"):
        self.name = name
        self.transformations = []
    
    def add_transformation(self, transform_fn, name: str):
        """Add a transformation function to the pipeline."""
        self.transformations.append((name, transform_fn))
        return self  # Enable method chaining
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply all transformations in sequence.
        This magic method makes the object callable.
        """
        result = data.copy()
        print(f"\nApplying {self.name}:")
        
        for name, transform_fn in self.transformations:
            result = transform_fn(result)
            print(f"  - Applied {name}: shape {result.shape}")
        
        return result
    
    def __len__(self):
        """Return the number of transformations."""
        return len(self.transformations)
    
    def __repr__(self):
        """String representation of the pipeline."""
        transforms = [name for name, _ in self.transformations]
        return f"TransformPipeline(steps={transforms})"


class ModelEnsemble:
    """
    An ensemble of models using __getitem__ magic method.
    Allows accessing models using bracket notation.
    """
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
    
    def add_model(self, name: str, model: Any):
        """Add a model to the ensemble."""
        self.models[name] = model
        return self
    
    def __getitem__(self, key: str) -> Any:
        """
        Access models using bracket notation.
        This magic method enables ensemble['model_name'].
        """
        if key not in self.models:
            raise KeyError(f"Model '{key}' not found in ensemble")
        return self.models[key]
    
    def __setitem__(self, key: str, value: Any):
        """
        Set models using bracket notation.
        This magic method enables ensemble['model_name'] = model.
        """
        self.models[key] = value
    
    def __contains__(self, key: str) -> bool:
        """
        Check if a model exists in the ensemble.
        This magic method enables 'model_name' in ensemble.
        """
        return key in self.models
    
    def __len__(self):
        """Return the number of models in the ensemble."""
        return len(self.models)
    
    def __iter__(self):
        """
        Make the ensemble iterable.
        This magic method enables for model in ensemble.
        """
        return iter(self.models.items())
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using all models and return the average.
        This magic method makes the ensemble callable.
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        all_predictions = []
        for name, model in self.models.items():
            pred = model.predict(X)
            all_predictions.append(pred)
            self.predictions[name] = pred
        
        # Return average prediction
        return np.mean(all_predictions, axis=0)
    
    def __repr__(self):
        """String representation of the ensemble."""
        model_names = list(self.models.keys())
        return f"ModelEnsemble(models={model_names})"


class DataBatch:
    """
    A data batch container using multiple magic methods.
    Demonstrates advanced indexing and slicing capabilities.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = X
        self.y = y
    
    def __getitem__(self, idx: Union[int, slice, List[int]]) -> 'DataBatch':
        """
        Enable indexing and slicing of the batch.
        Supports integer indexing, slicing, and fancy indexing.
        """
        if self.y is not None:
            return DataBatch(self.X[idx], self.y[idx])
        return DataBatch(self.X[idx])
    
    def __len__(self):
        """Return the number of samples in the batch."""
        return len(self.X)
    
    def __add__(self, other: 'DataBatch') -> 'DataBatch':
        """
        Concatenate two batches using the + operator.
        This magic method enables batch1 + batch2.
        """
        X_combined = np.vstack([self.X, other.X])
        
        if self.y is not None and other.y is not None:
            y_combined = np.concatenate([self.y, other.y])
            return DataBatch(X_combined, y_combined)
        
        return DataBatch(X_combined)
    
    def __repr__(self):
        """String representation of the batch."""
        if self.y is not None:
            return f"DataBatch(X: {self.X.shape}, y: {self.y.shape})"
        return f"DataBatch(X: {self.X.shape})"


class SimpleModel:
    """A simple model for demonstration purposes."""
    
    def __init__(self, name: str, coefficient: float = 1.0):
        self.name = name
        self.coefficient = coefficient
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make simple predictions."""
        return np.sum(X, axis=1) * self.coefficient


def demonstrate_magic_methods():
    """Demonstrate various magic methods in ML contexts."""
    print("=" * 60)
    print("Magic Methods for ML Components")
    print("=" * 60)
    
    # 1. __call__ with TransformPipeline
    print("\n1. __call__ - Making Objects Callable")
    print("-" * 60)
    
    # Create sample data
    data = np.random.randn(100, 5)
    
    # Create a transformation pipeline
    pipeline = TransformPipeline("Preprocessing")
    pipeline.add_transformation(lambda x: x - x.mean(axis=0), "Center Data")
    pipeline.add_transformation(lambda x: x / (x.std(axis=0) + 1e-8), "Normalize")
    pipeline.add_transformation(lambda x: np.clip(x, -3, 3), "Clip Outliers")
    
    print(f"Pipeline: {pipeline}")
    print(f"Number of steps: {len(pipeline)}")
    
    # Call the pipeline like a function
    transformed_data = pipeline(data)
    print(f"Original data shape: {data.shape}")
    print(f"Transformed data shape: {transformed_data.shape}")
    print(f"Transformed data range: [{transformed_data.min():.2f}, {transformed_data.max():.2f}]")
    
    # 2. __getitem__ with ModelEnsemble
    print("\n\n2. __getitem__ - Dictionary-like Access")
    print("-" * 60)
    
    # Create an ensemble
    ensemble = ModelEnsemble()
    ensemble.add_model("model_1", SimpleModel("Model1", coefficient=1.0))
    ensemble.add_model("model_2", SimpleModel("Model2", coefficient=1.5))
    ensemble.add_model("model_3", SimpleModel("Model3", coefficient=0.8))
    
    print(f"Ensemble: {ensemble}")
    print(f"Number of models: {len(ensemble)}")
    
    # Access models using bracket notation
    print(f"\nAccessing model_1: {ensemble['model_1'].name}")
    print(f"Checking if 'model_2' in ensemble: {'model_2' in ensemble}")
    print(f"Checking if 'model_4' in ensemble: {'model_4' in ensemble}")
    
    # Iterate through models
    print("\nIterating through ensemble:")
    for name, model in ensemble:
        print(f"  - {name}: coefficient={model.coefficient}")
    
    # Make predictions using the ensemble
    X_test = np.random.randn(10, 5)
    predictions = ensemble(X_test)
    print(f"\nEnsemble predictions (first 5): {predictions[:5]}")
    
    # 3. Multiple magic methods with DataBatch
    print("\n\n3. Multiple Magic Methods - DataBatch")
    print("-" * 60)
    
    X = np.random.randn(50, 3)
    y = np.random.randint(0, 2, 50)
    
    batch = DataBatch(X, y)
    print(f"Original batch: {batch}")
    print(f"Batch length: {len(batch)}")
    
    # Indexing
    print(f"\nFirst sample: {batch[0]}")
    print(f"Slice [10:20]: {batch[10:20]}")
    print(f"Fancy indexing [0,5,10]: {batch[[0, 5, 10]]}")
    
    # Addition
    batch2 = DataBatch(np.random.randn(30, 3), np.random.randint(0, 2, 30))
    combined = batch + batch2
    print(f"\nCombined batches: {combined}")
    
    print("\n" + "=" * 60)
    print("Benefits of Magic Methods:")
    print("- __call__: Makes objects callable, creating intuitive APIs")
    print("- __getitem__: Enables bracket notation for access")
    print("- __len__: Enables len() function")
    print("- __iter__: Makes objects iterable")
    print("- __add__: Enables + operator")
    print("- Creates Pythonic, intuitive interfaces")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_magic_methods()
