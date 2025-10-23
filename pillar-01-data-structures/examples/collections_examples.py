from __future__ import annotations
from collections import Counter, defaultdict, deque
from typing import Iterable, List

def count_words(words: Iterable[str]) -> Counter:
    return Counter(words)

def group_by_first_letter(words: Iterable[str]) -> dict[str, List[str]]:
    dd: defaultdict[str, List[str]] = defaultdict(list)
    for w in words:
        if w:
            dd[w[0].lower()].append(w)
    return dict(dd)

def sliding_windows(seq: Iterable[int], window_size: int) -> Iterable[List[int]]:
    d = deque(maxlen=window_size)
    for x in seq:
        d.append(x)
        if len(d) == window_size:
            yield list(d)

if __name__ == "__main__":
    words = ["apple", "banana", "avocado", "berry", "apricot"]
    print(count_words(words))
    print(group_by_first_letter(words))
    print(list(sliding_windows(range(1, 6), 3)))
