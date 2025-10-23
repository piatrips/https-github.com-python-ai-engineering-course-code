from __future__ import annotations
from typing import Generic, Iterable, Iterator, List, TypeVar

T = TypeVar('T', bound=float)

class Tensor(Generic[T]):
    """A minimal generic Tensor wrapper for float tensors.

    This is an educational wrapper — in real code you'd use numpy.typing or libraries' types.
    """

    def __init__(self, data: Iterable[Iterable[T]]) -> None:
        self._data: List[List[T]] = [list(row) for row in data]

    def shape(self) -> tuple[int, int]:
        rows = len(self._data)
        cols = len(self._data[0]) if rows else 0
        return (rows, cols)

    def __iter__(self) -> Iterator[List[T]]:
        return iter(self._data)

    def to_list(self) -> List[List[T]]:
        return [list(row) for row in self._data]


if __name__ == '__main__':
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    print('shape:', t.shape())
    print('as list:', t.to_list())