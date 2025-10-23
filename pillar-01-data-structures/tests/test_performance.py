from __future__ import annotations
import pytest
from pillar_01_data_structures.examples.performance_vs_dicts import compare_membership

def test_dict_faster_on_average():
    # Run a few times to reduce noise
    results = [compare_membership(n=3000, checks=1000) for _ in range(3)]
    # take median dict_faster
    dict_faster_counts = sum(1 for r in results if r["dict_faster"])
    assert dict_faster_counts >= 2
