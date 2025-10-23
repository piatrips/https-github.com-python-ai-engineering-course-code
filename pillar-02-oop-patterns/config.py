"""
Dataclasses used for configuration of preprocessing, training and evaluation.
Includes simple validation and helpers for serialization.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class PreprocessingConfig:
    normalize: bool = True
    missing_strategy: str = "mean"  # options: 'mean', 'median', 'drop'
    clip_range: tuple[float, float] | None = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PreprocessingConfig":
        return cls(
            normalize=bool(d.get("normalize", True)),
            missing_strategy=str(d.get("missing_strategy", "mean")),
            clip_range=d.get("clip_range"),
        )


@dataclass
class TrainingConfig:
    learning_rate: float = 0.01
    epochs: int = 10
    batch_size: int = 32
    model_name: str = "dummy_linear"

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        return cls(
            learning_rate=float(d.get("learning_rate", 0.01)),
            epochs=int(d.get("epochs", 10)),
            batch_size=int(d.get("batch_size", 32)),
            model_name=str(d.get("model_name", "dummy_linear")),
        )


def dump_config(obj: object) -> dict:
    """Return a serializable dict for configs."""
    return asdict(obj)


if __name__ == "__main__":
    pc = PreprocessingConfig(normalize=False, missing_strategy="drop")
    tc = TrainingConfig(learning_rate=0.1, epochs=5)
    print("preproc:", dump_config(pc))
    print("train:", dump_config(tc))
