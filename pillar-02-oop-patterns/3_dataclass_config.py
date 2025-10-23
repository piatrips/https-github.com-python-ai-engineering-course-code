"""
Example 3: Dataclasses for Configuration

This example demonstrates how to use dataclasses to create clean,
type-safe configuration objects for ML pipelines.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from enum import Enum
import json


class OptimizerType(Enum):
    """Enum for optimizer types."""
    SGD = "sgd"
    ADAM = "adam"
    RMSPROP = "rmsprop"


class ActivationType(Enum):
    """Enum for activation functions."""
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"


@dataclass
class ModelConfig:
    """
    Configuration for a machine learning model.
    
    Dataclasses provide:
    - Automatic __init__, __repr__, __eq__
    - Type hints
    - Default values
    - Immutability (with frozen=True)
    """
    model_name: str
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    optimizer: OptimizerType = OptimizerType.ADAM
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = asdict(self)
        config_dict['optimizer'] = self.optimizer.value
        return config_dict
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        if 'optimizer' in config_dict and isinstance(config_dict['optimizer'], str):
            config_dict['optimizer'] = OptimizerType(config_dict['optimizer'])
        return cls(**config_dict)


@dataclass
class DataConfig:
    """Configuration for data processing."""
    data_path: str
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    shuffle: bool = True
    normalize: bool = True
    feature_columns: Optional[List[str]] = None
    target_column: str = "target"
    
    def __post_init__(self):
        """Validate data configuration."""
        total_split = self.train_split + self.validation_split + self.test_split
        if not 0.99 <= total_split <= 1.01:  # Allow small floating point errors
            raise ValueError(f"Splits must sum to 1.0, got {total_split}")
        
        if self.feature_columns is None:
            self.feature_columns = []


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing steps."""
    normalize: bool = True
    remove_outliers: bool = True
    outlier_threshold: float = 3.0
    handle_missing: str = "mean"  # mean, median, drop
    scale_method: str = "standard"  # standard, minmax
    
    def __post_init__(self):
        """Validate preprocessing configuration."""
        valid_missing_methods = ["mean", "median", "drop"]
        if self.handle_missing not in valid_missing_methods:
            raise ValueError(f"handle_missing must be one of {valid_missing_methods}")
        
        valid_scale_methods = ["standard", "minmax", "none"]
        if self.scale_method not in valid_scale_methods:
            raise ValueError(f"scale_method must be one of {valid_scale_methods}")


@dataclass
class NetworkArchitecture:
    """Configuration for neural network architecture."""
    input_dim: int
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    output_dim: int = 1
    activation: ActivationType = ActivationType.RELU
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    
    def __post_init__(self):
        """Validate architecture configuration."""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1)")
        
        for i, size in enumerate(self.hidden_layers):
            if size <= 0:
                raise ValueError(f"Hidden layer {i} size must be positive")
    
    def get_layer_sizes(self) -> List[int]:
        """Get complete layer size specification."""
        return [self.input_dim] + self.hidden_layers + [self.output_dim]


@dataclass(frozen=True)  # Immutable configuration
class ExperimentConfig:
    """
    Immutable experiment configuration.
    
    Using frozen=True makes instances immutable, which is useful for
    ensuring configuration consistency throughout an experiment.
    """
    experiment_name: str
    experiment_id: str
    description: str
    tags: tuple = field(default_factory=tuple)  # Must use tuple for frozen dataclass
    created_at: str = field(default_factory=lambda: "2025-10-23")
    
    def __repr__(self):
        """Custom representation."""
        return (f"ExperimentConfig(name='{self.experiment_name}', "
                f"id='{self.experiment_id}')")


@dataclass
class PipelineConfig:
    """
    Complete ML pipeline configuration combining multiple configs.
    
    This demonstrates composition of dataclass configurations.
    """
    experiment: ExperimentConfig
    model: ModelConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    architecture: Optional[NetworkArchitecture] = None
    save_checkpoints: bool = True
    checkpoint_dir: str = "/tmp/checkpoints"
    log_metrics: bool = True
    
    def summary(self) -> str:
        """Generate a summary of the configuration."""
        lines = [
            "=" * 60,
            "Pipeline Configuration Summary",
            "=" * 60,
            f"Experiment: {self.experiment.experiment_name}",
            f"Model: {self.model.model_name}",
            f"Learning Rate: {self.model.learning_rate}",
            f"Batch Size: {self.model.batch_size}",
            f"Epochs: {self.model.epochs}",
            f"Data Path: {self.data.data_path}",
            f"Train/Val/Test Split: {self.data.train_split}/{self.data.validation_split}/{self.data.test_split}",
            f"Preprocessing: {self.preprocessing.scale_method} scaling",
        ]
        
        if self.architecture:
            lines.extend([
                f"Architecture: {self.architecture.get_layer_sizes()}",
                f"Activation: {self.architecture.activation.value}",
            ])
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire pipeline config to dictionary."""
        return {
            'experiment': asdict(self.experiment),
            'model': self.model.to_dict(),
            'data': asdict(self.data),
            'preprocessing': asdict(self.preprocessing),
            'architecture': asdict(self.architecture) if self.architecture else None,
            'save_checkpoints': self.save_checkpoints,
            'checkpoint_dir': self.checkpoint_dir,
            'log_metrics': self.log_metrics,
        }


def demonstrate_dataclasses():
    """Demonstrate dataclass usage for ML configuration."""
    print("=" * 60)
    print("Dataclasses for ML Configuration")
    print("=" * 60)
    
    # 1. Simple model configuration
    print("\n1. Basic Model Configuration")
    print("-" * 60)
    
    model_config = ModelConfig(
        model_name="ResNet50",
        learning_rate=0.0001,
        batch_size=64,
        epochs=50,
        optimizer=OptimizerType.ADAM
    )
    
    print(f"Model Config: {model_config}")
    print(f"As dict: {model_config.to_dict()}")
    
    # Save and load
    model_config.save("/tmp/model_config.json")
    
    # 2. Data configuration with defaults
    print("\n\n2. Data Configuration with Defaults")
    print("-" * 60)
    
    data_config = DataConfig(
        data_path="/data/training_data.csv",
        feature_columns=["feature1", "feature2", "feature3"]
    )
    
    print(f"Data Config: {data_config}")
    print(f"Feature columns: {data_config.feature_columns}")
    
    # 3. Network architecture
    print("\n\n3. Network Architecture Configuration")
    print("-" * 60)
    
    architecture = NetworkArchitecture(
        input_dim=100,
        hidden_layers=[128, 64, 32],
        output_dim=10,
        activation=ActivationType.RELU,
        dropout_rate=0.3
    )
    
    print(f"Architecture: {architecture}")
    print(f"Layer sizes: {architecture.get_layer_sizes()}")
    
    # 4. Immutable experiment configuration
    print("\n\n4. Immutable Experiment Configuration")
    print("-" * 60)
    
    experiment = ExperimentConfig(
        experiment_name="Image Classification v1",
        experiment_id="exp_001",
        description="Testing new architecture",
        tags=("vision", "classification", "resnet")
    )
    
    print(f"Experiment: {experiment}")
    print(f"Tags: {experiment.tags}")
    
    # Attempting to modify will raise an error (uncomment to see)
    # experiment.experiment_name = "Modified"  # This would raise FrozenInstanceError
    
    # 5. Complete pipeline configuration
    print("\n\n5. Complete Pipeline Configuration")
    print("-" * 60)
    
    preprocessing = PreprocessingConfig(
        normalize=True,
        remove_outliers=True,
        scale_method="standard"
    )
    
    pipeline_config = PipelineConfig(
        experiment=experiment,
        model=model_config,
        data=data_config,
        preprocessing=preprocessing,
        architecture=architecture,
        save_checkpoints=True,
        checkpoint_dir="/tmp/checkpoints/exp_001"
    )
    
    print(pipeline_config.summary())
    
    # Convert to dictionary for serialization
    config_dict = pipeline_config.to_dict()
    print(f"\nConfiguration keys: {list(config_dict.keys())}")
    
    # 6. Validation example
    print("\n\n6. Configuration Validation")
    print("-" * 60)
    
    try:
        invalid_config = ModelConfig(
            model_name="TestModel",
            learning_rate=-0.001  # Invalid: negative learning rate
        )
    except ValueError as e:
        print(f"✓ Validation caught error: {e}")
    
    try:
        invalid_data = DataConfig(
            data_path="/data/test.csv",
            train_split=0.5,
            validation_split=0.3,
            test_split=0.3  # Invalid: splits don't sum to 1.0
        )
    except ValueError as e:
        print(f"✓ Validation caught error: {e}")
    
    print("\n" + "=" * 60)
    print("Benefits of Dataclasses:")
    print("- Automatic generation of __init__, __repr__, __eq__")
    print("- Type hints for better IDE support and validation")
    print("- Default values and factory functions")
    print("- Immutability option with frozen=True")
    print("- Easy serialization with asdict()")
    print("- Post-initialization validation with __post_init__")
    print("- Cleaner, more maintainable configuration code")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_dataclasses()
