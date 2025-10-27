"""
ML Pipeline Project

A modular machine learning pipeline demonstrating OOP patterns:
- ABC (Abstract Base Classes) for consistent interfaces
- Magic methods (__call__, __getitem__) for intuitive APIs  
- Dataclasses for clean configuration
- Composition pattern for building complex systems from simple components

Modules:
- preprocessing: Data preprocessing components
- training: Model training components
- evaluation: Model evaluation and metrics
- pipeline: Main pipeline orchestrator
"""

__version__ = "1.0.0"

from .preprocessing import (
    BasePreprocessor,
    StandardScaler,
    MinMaxScaler,
    MissingValueImputer,
    OutlierClipper,
    FeatureEngineer,
    PreprocessingPipeline,
)

from .training import (
    BaseTrainer,
    LinearRegressionTrainer,
    GradientDescentTrainer,
    RidgeRegressionTrainer,
    EnsembleTrainer,
    ModelSelector,
)

from .evaluation import (
    EvaluationResults,
    BaseMetric,
    MeanSquaredError,
    RootMeanSquaredError,
    MeanAbsoluteError,
    R2Score,
    MaxError,
    MedianAbsoluteError,
    Evaluator,
    ResidualAnalyzer,
    CrossValidator,
)

from .pipeline import (
    PipelineConfig,
    CompletePipeline,
    create_sample_pipeline,
)

__all__ = [
    # Preprocessing
    'BasePreprocessor',
    'StandardScaler',
    'MinMaxScaler',
    'MissingValueImputer',
    'OutlierClipper',
    'FeatureEngineer',
    'PreprocessingPipeline',
    
    # Training
    'BaseTrainer',
    'LinearRegressionTrainer',
    'GradientDescentTrainer',
    'RidgeRegressionTrainer',
    'EnsembleTrainer',
    'ModelSelector',
    
    # Evaluation
    'EvaluationResults',
    'BaseMetric',
    'MeanSquaredError',
    'RootMeanSquaredError',
    'MeanAbsoluteError',
    'R2Score',
    'MaxError',
    'MedianAbsoluteError',
    'Evaluator',
    'ResidualAnalyzer',
    'CrossValidator',
    
    # Pipeline
    'PipelineConfig',
    'CompletePipeline',
    'create_sample_pipeline',
]
