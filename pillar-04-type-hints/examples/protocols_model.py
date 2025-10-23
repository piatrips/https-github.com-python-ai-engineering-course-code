from __future__ import annotations
from typing import Protocol, Sequence, runtime_checkable

@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol defining a simple model interface for training and inference."""

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> None: ...

    def predict(self, X: Sequence[Sequence[float]]) -> Sequence[float]: ...


class SimpleEstimator:
    """A concrete estimator that satisfies ModelProtocol without inheriting from it."""

    def __init__(self) -> None:
        self.coefs: list[float] | None = None

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> None:
        if not X:
            self.coefs = []
            return
        n = len(X[0])
        # naive coefficients
        means_x = [sum(row[i] for row in X) / len(X) for i in range(n)]
        mean_y = sum(y) / len(y)
        self.coefs = [(mean_y / (mx if mx != 0 else 1.0)) for mx in means_x]

    def predict(self, X: Sequence[Sequence[float]]) -> Sequence[float]:
        if self.coefs is None:
            raise RuntimeError('not fitted')
        return [sum(a*b for a,b in zip(self.coefs, row)) for row in X]


def accepts_model(m: ModelProtocol) -> None:
    # statically typed to accept any object that implements the protocol
    print('Model accepted, calling fit on a tiny dataset')
    m.fit([[1.0, 2.0]], [3.0])


if __name__ == '__main__':
    est = SimpleEstimator()
    accepts_model(est)
    print('pred:', est.predict([[2.0, 3.0]]))