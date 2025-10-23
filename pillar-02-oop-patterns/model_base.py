"""
ABC example: model base classes

Defines an abstract BaseModel and a simple concrete implementation.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence


class BaseModel(ABC):
    """Abstract base class for models in the pipeline."""

    @abstractmethod
    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> None:
        """Train the model on X, y."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Sequence[Sequence[float]]) -> Sequence[float]:
        """Return model predictions for X."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Optional: save model to disk (very small placeholder)."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(repr(self.__dict__))


class DummyLinearModel(BaseModel):
    """A tiny deterministic 'linear' model used for examples."""

    def __init__(self) -> None:
        self.coef_: list[float] | None = None
        self.intercept_: float = 0.0

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> None:
        # Toy 'fit': set coefficient per feature as ratio of mean(y)/mean(x_i)
        if not X:
            self.coef_ = []
            self.intercept_ = 0.0
            return
        n_features = len(X[0])
        means_x = [sum(row[i] for row in X) / len(X) for i in range(n_features)]
        mean_y = sum(y) / len(y)
        self.coef_ = [(mean_y / (mx if mx != 0 else 1.0)) for mx in means_x]
        self.intercept_ = 0.0

    def predict(self, X: Sequence[Sequence[float]]) -> Sequence[float]:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted")
        return [sum(f * c for f, c in zip(row, self.coef_)) + self.intercept_ for row in X]


if __name__ == "__main__":
    # quick smoke test
    X = [[1.0, 2.0], [2.0, 1.0], [3.0, 0.0]]
    y = [3.0, 4.0, 5.0]
    m = DummyLinearModel()
    m.fit(X, y)
    print("coef:", m.coef_)
    print("pred:", m.predict([[4.0, 5.0]]))
