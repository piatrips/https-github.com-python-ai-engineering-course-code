"""
Generators for lazy data loading.

This module demonstrates how to use generators for memory-efficient lazy loading
of data, which is particularly useful when working with large datasets.
"""
import time
from typing import Iterator, List


def load_data_eagerly(n: int) -> List[int]:
    """
    Eagerly load all data into memory at once.
    
    Args:
        n: Number of items to load
        
    Returns:
        List of all items
    """
    print(f"Eagerly loading {n} items...")
    return [i * i for i in range(n)]


def load_data_lazily(n: int) -> Iterator[int]:
    """
    Lazily generate data one item at a time.
    
    Args:
        n: Number of items to generate
        
    Yields:
        One item at a time
    """
    print(f"Lazily generating {n} items...")
    for i in range(n):
        yield i * i


def read_large_file(filepath: str) -> Iterator[str]:
    """
    Read a large file line by line using a generator.
    
    Args:
        filepath: Path to the file
        
    Yields:
        One line at a time
    """
    with open(filepath, 'r') as f:
        for line in f:
            yield line.strip()


def process_data_in_chunks(data: Iterator[int], chunk_size: int) -> Iterator[List[int]]:
    """
    Process data in chunks using a generator.
    
    Args:
        data: Iterator of data items
        chunk_size: Size of each chunk
        
    Yields:
        Chunks of data
    """
    chunk = []
    for item in data:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:  # yield remaining items
        yield chunk


def fibonacci_generator(n: int) -> Iterator[int]:
    """
    Generate Fibonacci sequence up to n numbers.
    
    Args:
        n: Number of Fibonacci numbers to generate
        
    Yields:
        Fibonacci numbers one at a time
    """
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


def demonstrate_generators():
    """Demonstrate the benefits of using generators for lazy loading."""
    print("Generator Demonstrations: Lazy Data Loading\n")
    print("=" * 70)
    
    # Example 1: Memory efficiency
    print("\n1. Memory Efficiency Comparison:")
    n = 10
    
    print(f"\n   Eager loading (creates list of {n} items):")
    eager_data = load_data_eagerly(n)
    print(f"   First 5 items: {eager_data[:5]}")
    
    print(f"\n   Lazy loading (generates items on demand):")
    lazy_data = load_data_lazily(n)
    print(f"   First 5 items: {list(item for i, item in enumerate(lazy_data) if i < 5)}")
    
    # Example 2: Fibonacci generator
    print("\n2. Fibonacci Generator:")
    fib_gen = fibonacci_generator(10)
    print(f"   First 10 Fibonacci numbers: {list(fib_gen)}")
    
    # Example 3: Processing in chunks
    print("\n3. Processing Data in Chunks:")
    data_gen = load_data_lazily(20)
    chunks = list(process_data_in_chunks(data_gen, chunk_size=5))
    print(f"   Number of chunks: {len(chunks)}")
    print(f"   First chunk: {chunks[0]}")
    print(f"   Last chunk: {chunks[-1]}")
    
    # Example 4: Generator expression
    print("\n4. Generator Expression:")
    squared = (x * x for x in range(5))
    print(f"   Type: {type(squared)}")
    print(f"   Values: {list(squared)}")


if __name__ == "__main__":
    demonstrate_generators()
