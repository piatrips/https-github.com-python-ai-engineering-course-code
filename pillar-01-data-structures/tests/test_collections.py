from __future__ import annotations
from pillar_01_data_structures.examples.collections_examples import count_words, group_by_first_letter, sliding_windows

def test_count_words():
    words = ["a", "b", "a", "c", "b"]
    cnt = count_words(words)
    assert cnt["a"] == 2
    assert cnt["b"] == 2


def test_group_by_first_letter():
    words = ["Apple", "apricot", "banana"]
    grouped = group_by_first_letter(words)
    assert grouped["a"] == ["Apple", "apricot"]


def test_sliding_windows():
    windows = list(sliding_windows(range(1, 6), 3))
    assert windows == [[1,2,3],[2,3,4],[3,4,5]]
