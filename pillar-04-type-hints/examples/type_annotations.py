from __future__ import annotations
from typing import Sequence, Tuple, Callable

# Type aliases for clarity
Vector = Sequence[float]
Matrix = Sequence[Vector]

# A typical ML training function signature
def train_step(X: Matrix, y: Sequence[float], lr: float) -> Sequence[float]:
    """A single training step that returns updated parameters (toy example)."""
    # toy: return a vector of parameter updates (same length as features)
    if not X:
        return []
    n_features = len(X[0])
    # compute very simple gradient-like values
    grads = [0.0 for _ in range(n_features)]
    for row, target in zip(X, y):
        for i, xi in enumerate(row):
            grads[i] += (sum(row) - target) * xi
    return [g * lr for g in grads]


def predict_fn(params: Sequence[float]) -> Callable[[Vector], float]:
    """Return a statically typed callable that uses params to predict a single example."""
    def predict_one(x: Vector) -> float:
        return sum(p * xi for p, xi in zip(params, x))
    return predict_one


if __name__ == '__main__':
    X = [[1.0, 2.0], [2.0, 1.0]]
    y = [3.0, 4.0]
    params = train_step(X, y, lr=0.01)
    pred = predict_fn(params)([3.0, 1.0])
    print('params', params)
    print('pred', pred)