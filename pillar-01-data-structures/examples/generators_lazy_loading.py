from __future__ import annotations
from typing import Iterator

def number_generator(n: int) -> Iterator[int]:
    """Yield numbers 0..n-1 lazily."""
    i = 0
    while i < n:
        yield i
        i += 1

def load_list(n: int) -> list[int]:
    """Return a materialized list of numbers 0..n-1."""
    return [i for i in range(n)]

if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    # generator uses very little memory; list will allocate full memory
    gen = number_generator(n)
    # consume a few items
    for _ in range(5):
        print(next(gen))

    # show that generator can be iterated lazily
    gen2 = number_generator(5)
    print(list(gen2))
