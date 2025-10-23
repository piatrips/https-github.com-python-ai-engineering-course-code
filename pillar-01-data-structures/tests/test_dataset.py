from __future__ import annotations
from pillar_01_data_structures.examples.dataset import CustomDataset

def make_sample():
    return (([i * 1.0, i * 2.0], float(i)) for i in range(5))

def test_len_and_indexing():
    ds = CustomDataset(make_sample())
    assert len(ds) == 5
    f, t = ds[2]
    assert f == [2.0, 4.0]
    assert t == 2.0


def test_negative_index_and_slice():
    ds = CustomDataset(make_sample())
    f, t = ds[-1]
    assert t == 4.0
    sub = ds[1:4]
    assert len(sub) == 3
    sf, st = sub[0]
    assert st == 1.0
