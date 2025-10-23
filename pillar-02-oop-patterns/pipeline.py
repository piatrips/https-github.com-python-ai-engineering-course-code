"""
Composition pattern for a modular ML pipeline:
- Preprocessor: transforms raw data
- Trainer: trains a model (expects BaseModel-like object)
- Evaluator: evaluates predictions
- MLPipeline: composes the components and runs end-to-end
"""
from __future__ import annotations
from typing import Callable, Sequence, Any
from .config import PreprocessingConfig, TrainingConfig
from .model_base import BaseModel, DummyLinearModel
from .magic_examples import CallableTransformer


class Preprocessor:
    def __init__(self, config: PreprocessingConfig, transformer: CallableTransformer | None = None) -> None:
        self.config = config
        self.transformer = transformer

    def process(self, raw: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:
        X = [list(row) for row in raw]  # copy to avoid mutating input
        if self.config.normalize and X:
            n_cols = len(X[0])
            mins = [min(row[i] for row in X) for i in range(n_cols)]
            maxs = [max(row[i] for row in X) for i in range(n_cols)]
            for r in X:
                for i in range(n_cols):
                    denom = maxs[i] - mins[i] if maxs[i] != mins[i] else 1.0
                    r[i] = (r[i] - mins[i]) / denom
        if self.transformer:
            X = self.transformer(X)
        return X


class Trainer:
    def __init__(self, model_factory: Callable[[], BaseModel], config: TrainingConfig) -> None:
        self.model = model_factory()
        self.config = config

    def train(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> BaseModel:
        # Example: here you could add batching/epochs; keep it simple for clarity
        self.model.fit(X, y)
        return self.model


class Evaluator:
    @staticmethod
    def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        if not y_true:
            return 0.0
        return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


class MLPipeline:
    def __init__(self, preprocessor: Preprocessor, trainer: Trainer, evaluator: Evaluator) -> None:
        self.preprocessor = preprocessor
        self.trainer = trainer
        self.evaluator = evaluator

    def run(self, raw_X: Sequence[Sequence[float]], y: Sequence[float]) -> dict[str, Any]:
        X_proc = self.preprocessor.process(raw_X)
        model = self.trainer.train(X_proc, y)
        preds = model.predict(X_proc)
        score = self.evaluator.mean_absolute_error(y, preds)
        return {"model": model, "preds": preds, "mae": score}


if __name__ == "__main__":
    # tiny end-to-end example
    raw_X = [[10.0, 20.0], [20.0, 10.0], [30.0, 0.0]]
    y = [30.0, 40.0, 50.0]

    pre_cfg = PreprocessingConfig(normalize=True)
    tran = CallableTransformer(lambda X: [[v * 1.0 for v in row] for row in X], name="identity")
    pre = Preprocessor(pre_cfg, transformer=tran)

    train_cfg = TrainingConfig(learning_rate=0.01, epochs=1)
    trainer = Trainer(lambda: DummyLinearModel(), train_cfg)
    evaluator = Evaluator()

    pipeline = MLPipeline(pre, trainer, evaluator)
    result = pipeline.run(raw_X, y)
    print("MAE:", result["mae"]) 
    print("Preds:", result["preds"])
