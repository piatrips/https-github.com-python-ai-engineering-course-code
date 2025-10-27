"""
Performance comparison between lists and dictionaries.

This module demonstrates the performance differences between lists and dictionaries
for different operations, highlighting when to use each data structure.
"""
import time
from typing import List, Dict


def measure_time(func):
    """Decorator to measure execution time of a function."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start
    return wrapper


@measure_time
def list_lookup(data: List[int], target: int) -> bool:
    """Search for an element in a list - O(n) time complexity."""
    return target in data


@measure_time
def dict_lookup(data: Dict[int, bool], target: int) -> bool:
    """Search for a key in a dictionary - O(1) average time complexity."""
    return target in data


@measure_time
def list_insert(data: List[int], value: int) -> None:
    """Insert an element at the beginning of a list - O(n) time complexity."""
    data.insert(0, value)


@measure_time
def dict_insert(data: Dict[int, bool], key: int) -> None:
    """Insert a key-value pair in a dictionary - O(1) average time complexity."""
    data[key] = True


def compare_performance(size: int = 10000):
    """
    Compare performance of lists vs dictionaries for lookup and insert operations.
    
    Args:
        size: Number of elements to test with
        
    Returns:
        Dictionary with performance metrics
    """
    # Prepare data
    list_data = list(range(size))
    dict_data = {i: True for i in range(size)}
    
    # Test lookup for an element that doesn't exist (worst case)
    target = size + 1
    
    _, list_lookup_time = list_lookup(list_data, target)
    _, dict_lookup_time = dict_lookup(dict_data, target)
    
    # Test insert operations
    _, list_insert_time = list_insert(list_data, -1)
    _, dict_insert_time = dict_insert(dict_data, -1)
    
    return {
        'size': size,
        'list_lookup_time': list_lookup_time,
        'dict_lookup_time': dict_lookup_time,
        'list_insert_time': list_insert_time,
        'dict_insert_time': dict_insert_time,
        'lookup_speedup': list_lookup_time / dict_lookup_time if dict_lookup_time > 0 else 0,
        'insert_speedup': list_insert_time / dict_insert_time if dict_insert_time > 0 else 0,
    }


def demonstrate_performance():
    """Demonstrate the performance differences between lists and dicts."""
    print("Performance Comparison: Lists vs Dictionaries\n")
    print("=" * 70)
    
    for size in [100, 1000, 10000]:
        results = compare_performance(size)
        print(f"\nData size: {results['size']:,} elements")
        print(f"  List lookup time:  {results['list_lookup_time']:.6f}s")
        print(f"  Dict lookup time:  {results['dict_lookup_time']:.6f}s")
        print(f"  Lookup speedup:    {results['lookup_speedup']:.2f}x")
        print(f"  List insert time:  {results['list_insert_time']:.6f}s")
        print(f"  Dict insert time:  {results['dict_insert_time']:.6f}s")
        print(f"  Insert speedup:    {results['insert_speedup']:.2f}x")


if __name__ == "__main__":
    demonstrate_performance()
