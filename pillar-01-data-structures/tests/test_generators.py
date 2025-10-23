from __future__ import annotations
from pillar_01_data_structures.examples.generators_lazy_loading import number_generator, load_list

def test_generator_yields_correct_sequence():
    n = 10
    gen = number_generator(n)
    assert list(gen) == list(range(n))


def test_list_loader():
    n = 10
    lst = load_list(n)
    assert lst == list(range(n))
