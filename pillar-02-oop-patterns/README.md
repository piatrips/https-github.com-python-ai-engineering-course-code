# Pillar 02: Object-Oriented Programming Patterns for ML

This directory contains examples and a complete project demonstrating essential OOP patterns for machine learning applications.

## 📚 Contents

### Individual Examples

1. **`1_abc_model_base.py`** - Abstract Base Classes (ABC)
   - Demonstrates how to use ABC to create base classes for ML models
   - Ensures all derived models implement required methods
   - Examples: LinearRegressionModel, DecisionTreeModel

2. **`2_magic_methods.py`** - Magic Methods
   - Shows how to use Python magic methods for intuitive APIs
   - `__call__`: Make objects callable like functions
   - `__getitem__`: Enable bracket notation for access
   - `__len__`, `__iter__`, `__add__`: More Pythonic interfaces
   - Examples: TransformPipeline, ModelEnsemble, DataBatch

3. **`3_dataclass_config.py`** - Dataclasses for Configuration
   - Use dataclasses for clean, type-safe configuration
   - Automatic generation of `__init__`, `__repr__`, `__eq__`
   - Validation with `__post_init__`
   - Immutability with `frozen=True`
   - Examples: ModelConfig, DataConfig, PipelineConfig

4. **`4_composition_pipeline.py`** - Composition Pattern
   - Build complex functionality from simple, reusable components
   - "Favor composition over inheritance" principle
   - Examples: Preprocessing pipeline, complete ML pipeline

### Complete Project: `ml_pipeline_project/`

A modular ML pipeline demonstrating all OOP patterns in a real-world context.

**Structure:**
```
ml_pipeline_project/
├── __init__.py          # Package initialization
├── preprocessing.py     # Data preprocessing components
├── training.py          # Model training components
├── evaluation.py        # Model evaluation and metrics
└── pipeline.py          # Main pipeline orchestrator
```

**Key Features:**
- **ABC Pattern**: Base classes for preprocessors, trainers, and metrics
- **Magic Methods**: Callable pipelines, iterable ensembles
- **Dataclasses**: Clean configuration management
- **Composition**: Build complex pipelines from simple components

## 🚀 Quick Start

### Running Individual Examples

Each example is self-contained and can be run directly:

```bash
# ABC Pattern
python 1_abc_model_base.py

# Magic Methods
python 2_magic_methods.py

# Dataclasses
python 3_dataclass_config.py

# Composition Pattern
python 4_composition_pipeline.py
```

### Using the ML Pipeline Project

```python
from ml_pipeline_project import create_sample_pipeline
import numpy as np

# Generate sample data
X = np.random.randn(200, 5)
y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(200) * 0.1

# Create and fit pipeline
pipeline = create_sample_pipeline()
pipeline.fit(X, y)

# Make predictions
X_new = np.random.randn(10, 5)
predictions = pipeline(X_new)  # Using __call__ magic method

# View results
print(pipeline.get_results_summary())
```

### Custom Pipeline Example

```python
from ml_pipeline_project import (
    CompletePipeline, PipelineConfig,
    PreprocessingPipeline, StandardScaler, OutlierClipper,
    LinearRegressionTrainer
)

# Configure pipeline
config = PipelineConfig(
    name="Custom Pipeline",
    train_split=0.7,
    val_split=0.15,
    test_split=0.15
)

# Build preprocessing
preprocessing = PreprocessingPipeline()
preprocessing.add_step(OutlierClipper(method="zscore", threshold=3.0))
preprocessing.add_step(StandardScaler())

# Build pipeline
pipeline = CompletePipeline(config)
pipeline.set_preprocessor(preprocessing)
pipeline.set_trainer(LinearRegressionTrainer(regularization=0.1))

# Fit and evaluate
pipeline.fit(X, y)
```

## 📖 Key OOP Patterns

### 1. Abstract Base Classes (ABC)

**Purpose**: Define interfaces and ensure implementation consistency

**Example**:
```python
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
```

**Benefits**:
- Enforces interface consistency
- Prevents instantiation of incomplete classes
- Improves code maintainability

### 2. Magic Methods

**Purpose**: Create intuitive, Pythonic interfaces

**Common Magic Methods**:
- `__call__`: Make objects callable
- `__getitem__`: Enable bracket notation
- `__len__`: Support len() function
- `__iter__`: Make objects iterable
- `__add__`: Support + operator

**Example**:
```python
class Pipeline:
    def __call__(self, data):
        # Makes pipeline(data) possible
        return self.transform(data)
    
    def __getitem__(self, key):
        # Makes pipeline['step_name'] possible
        return self.steps[key]
```

### 3. Dataclasses

**Purpose**: Clean, type-safe data containers

**Example**:
```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    learning_rate: float = 0.001
    batch_size: int = 32
    
    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
```

**Benefits**:
- Automatic `__init__`, `__repr__`, `__eq__`
- Type hints for better IDE support
- Validation with `__post_init__`
- Immutability option

### 4. Composition Pattern

**Purpose**: Build complex systems from simple, reusable components

**Example**:
```python
class MLPipeline:
    def __init__(self):
        self.preprocessor = None  # Composed
        self.model = None         # Composed
        self.evaluator = None     # Composed
    
    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor
        return self
```

**Benefits**:
- Flexibility: Easy to swap components
- Reusability: Components work independently
- Testability: Test components in isolation
- Avoids deep inheritance hierarchies

## 🎯 Design Principles

1. **Single Responsibility**: Each class has one clear purpose
2. **Open/Closed**: Open for extension, closed for modification
3. **Liskov Substitution**: Derived classes can replace base classes
4. **Interface Segregation**: Clients shouldn't depend on unused methods
5. **Dependency Inversion**: Depend on abstractions, not concretions
6. **Composition over Inheritance**: Build from components

## 📊 Project Architecture

The ML pipeline project follows a modular architecture:

```
Input Data
    ↓
Preprocessing Pipeline (Composition)
    ├── MissingValueImputer
    ├── OutlierClipper
    └── StandardScaler
    ↓
Model Trainer (ABC)
    ├── LinearRegressionTrainer
    ├── RidgeRegressionTrainer
    └── GradientDescentTrainer
    ↓
Evaluator (Composition)
    ├── MeanSquaredError
    ├── RootMeanSquaredError
    ├── MeanAbsoluteError
    └── R2Score
    ↓
Results (Dataclass)
```

## 🔧 Available Components

### Preprocessing
- `StandardScaler`: Standardize features
- `MinMaxScaler`: Scale to range
- `MissingValueImputer`: Handle missing values
- `OutlierClipper`: Remove outliers
- `FeatureEngineer`: Create polynomial features

### Training
- `LinearRegressionTrainer`: OLS regression
- `RidgeRegressionTrainer`: L2 regularization
- `GradientDescentTrainer`: Iterative optimization
- `EnsembleTrainer`: Combine multiple models
- `ModelSelector`: Cross-validation model selection

### Evaluation
- `MeanSquaredError`: MSE metric
- `RootMeanSquaredError`: RMSE metric
- `MeanAbsoluteError`: MAE metric
- `R2Score`: Coefficient of determination
- `MaxError`: Maximum error
- `MedianAbsoluteError`: Median absolute error
- `ResidualAnalyzer`: Residual analysis
- `CrossValidator`: K-fold cross-validation

## 💡 Best Practices

1. **Use ABC for Interfaces**: Define clear contracts
2. **Magic Methods for Intuition**: Make code feel natural
3. **Dataclasses for Data**: Clean configuration management
4. **Composition for Flexibility**: Build from components
5. **Type Hints**: Better IDE support and documentation
6. **Validation**: Fail fast with clear error messages
7. **Documentation**: Clear docstrings for all public methods

## 🧪 Testing Individual Components

Each module can be tested independently:

```bash
# Test preprocessing
python ml_pipeline_project/preprocessing.py

# Test training
python ml_pipeline_project/training.py

# Test evaluation
python ml_pipeline_project/evaluation.py

# Test complete pipeline
python ml_pipeline_project/pipeline.py
```

## 📝 Requirements

- Python 3.7+
- NumPy

Install dependencies:
```bash
pip install numpy
```

## 🤝 Contributing

When adding new components:
1. Follow the ABC pattern for base classes
2. Implement relevant magic methods
3. Use dataclasses for configuration
4. Maintain composability
5. Add docstrings and type hints
6. Include usage examples

## 📚 Further Reading

- [Python ABC Documentation](https://docs.python.org/3/library/abc.html)
- [Python Data Model (Magic Methods)](https://docs.python.org/3/reference/datamodel.html)
- [Dataclasses Documentation](https://docs.python.org/3/library/dataclasses.html)
- [Composition over Inheritance](https://en.wikipedia.org/wiki/Composition_over_inheritance)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)

## 📄 License

This project is part of the Python AI Engineering Course.
