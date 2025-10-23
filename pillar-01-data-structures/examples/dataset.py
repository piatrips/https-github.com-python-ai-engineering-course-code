from __future__ import annotations
from typing import Iterable, List, Tuple, Union

class CustomDataset:
    """Simple dataset implementing the sequence protocol.

    Supports len(), indexing (int), negative indices, and slicing.
    Each item is a (features, target) tuple.
    """

    def __init__(self, data: Iterable[Tuple[List[float], float]]) -> None:
        self._data = list(data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Tuple[List[float], float], "CustomDataset"]:
        if isinstance(idx, int):
            return self._data[idx]
        if isinstance(idx, slice):
            return CustomDataset(self._data[idx])
        raise TypeError("Indices must be int or slice")

if __name__ == "__main__":
    ds = CustomDataset((([i, i + 1], float(i * 2)) for i in range(5)))
    print(len(ds))
    print(ds[0])
    print(ds[-1])
    print(len(ds[1:4]))
