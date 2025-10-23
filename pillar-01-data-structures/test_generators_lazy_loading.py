"""
Tests for generators_lazy_loading module.
"""
import pytest
from generators_lazy_loading import (
    load_data_eagerly,
    load_data_lazily,
    process_data_in_chunks,
    fibonacci_generator,
)


class TestGenerators:
    """Tests for generator functions."""
    
    def test_load_data_eagerly(self):
        """Test eager data loading returns a list."""
        result = load_data_eagerly(10)
        assert isinstance(result, list)
        assert len(result) == 10
        assert result[0] == 0
        assert result[5] == 25
        assert result[9] == 81
    
    def test_load_data_lazily(self):
        """Test lazy data loading returns a generator."""
        result = load_data_lazily(10)
        # Check it's a generator
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')
        
        # Convert to list and verify
        result_list = list(result)
        assert len(result_list) == 10
        assert result_list[0] == 0
        assert result_list[5] == 25
        assert result_list[9] == 81
    
    def test_load_data_lazily_empty(self):
        """Test lazy loading with zero items."""
        result = list(load_data_lazily(0))
        assert len(result) == 0
    
    def test_fibonacci_generator(self):
        """Test Fibonacci number generation."""
        fib = list(fibonacci_generator(10))
        assert len(fib) == 10
        assert fib == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    
    def test_fibonacci_generator_edge_cases(self):
        """Test Fibonacci generator with edge cases."""
        # Empty sequence
        fib = list(fibonacci_generator(0))
        assert fib == []
        
        # Single number
        fib = list(fibonacci_generator(1))
        assert fib == [0]
        
        # Two numbers
        fib = list(fibonacci_generator(2))
        assert fib == [0, 1]
    
    def test_process_data_in_chunks(self):
        """Test chunking data from a generator."""
        data = load_data_lazily(10)
        chunks = list(process_data_in_chunks(data, chunk_size=3))
        
        assert len(chunks) == 4  # 3 full chunks + 1 partial
        assert len(chunks[0]) == 3
        assert len(chunks[1]) == 3
        assert len(chunks[2]) == 3
        assert len(chunks[3]) == 1
        
        assert chunks[0] == [0, 1, 4]
        assert chunks[3] == [81]
    
    def test_process_data_in_chunks_exact_fit(self):
        """Test chunking when data size is exactly divisible by chunk size."""
        data = load_data_lazily(9)
        chunks = list(process_data_in_chunks(data, chunk_size=3))
        
        assert len(chunks) == 3
        assert all(len(chunk) == 3 for chunk in chunks)
    
    def test_process_data_in_chunks_empty(self):
        """Test chunking with empty data."""
        data = iter([])
        chunks = list(process_data_in_chunks(data, chunk_size=3))
        assert len(chunks) == 0
    
    def test_generator_memory_efficiency(self):
        """Test that generators don't create all data upfront."""
        # Create a generator
        gen = load_data_lazily(1000)
        
        # Get just one item without creating the full list
        first_item = next(gen)
        assert first_item == 0
        
        # Get the next item
        second_item = next(gen)
        assert second_item == 1
    
    def test_generator_expression(self):
        """Test generator expressions work as expected."""
        gen = (x * x for x in range(5))
        result = list(gen)
        assert result == [0, 1, 4, 9, 16]
