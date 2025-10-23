"""
Demonstrates magic methods:
- __call__ for a transformer that can be invoked like a function
- __getitem__ for a dataset/container that supports indexing and slicing
"""
from __future__ import annotations
from typing import Any, Callable, Iterable, Iterator, List, Sequence, Union


class CallableTransformer:
    """
    Wraps a function and makes the wrapper callable.
    Useful to keep transformer state and still use it like a function.
    """

    def __init__(self, func: Callable[[Sequence[Sequence[float]]], Sequence[Sequence[float]]], name: str = "") -> None:
        self.func = func
        self.name = name

    def __call__(self, X: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:
        # small side effect for demonstration purposes
        print(f"CallableTransformer({self.name!r}) called with {len(X)} rows")
        return self.func(X)

    def __repr__(self) -> str:
        return f"<CallableTransformer name={self.name!r}>"


class DataContainer:
    """
    Small container that implements __getitem__:
      - int -> single record
      - slice -> DataContainer (view)
      - sequence[int] -> DataContainer with selected indices
      - str -> column projection (for dict-like rows)
    """

    def __init__(self, rows: Iterable[Any]) -> None:
        self._rows = list(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._rows)

    def __getitem__(self, idx: Union[int, slice, Sequence[int], str]) -> Any:
        if isinstance(idx, int):
            return self._rows[idx]
        if isinstance(idx, slice):
            return DataContainer(self._rows[idx])
        if isinstance(idx, (list, tuple)):
            return DataContainer([self._rows[i] for i in idx])
        if isinstance(idx, str):
            # assume rows are dict-like; return list of column values
            return [row.get(idx) if isinstance(row, dict) else None for row in self._rows]
        raise TypeError("Unsupported index type")

    def __repr__(self) -> str:
        return f"<DataContainer len={len(self._rows)}>"


if __name__ == "__main__":
    rows = [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5, "y": 6}]
    dc = DataContainer(rows)
    print(dc[0])         # single record
    print(dc[0:2])       # slice -> DataContainer
    print(dc[[0, 2]])    # DataContainer with selected rows
    print(dc["x"])       # list of x values

    # example callable transformer
    def scale(X: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:
        return [[float(v) * 2.0 for v in row] for row in X]

    t = CallableTransformer(scale, name="double")
    print(t([[1, 2], [3, 4]]))
