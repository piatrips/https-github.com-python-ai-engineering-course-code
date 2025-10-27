"""
Tests for list_vs_dict_performance module.
"""
import pytest
from list_vs_dict_performance import (
    list_lookup,
    dict_lookup,
    list_insert,
    dict_insert,
    compare_performance,
)


class TestPerformanceComparison:
    """Tests for performance comparison functions."""
    
    def test_list_lookup_found(self):
        """Test list lookup for existing element."""
        data = [1, 2, 3, 4, 5]
        result, time_taken = list_lookup(data, 3)
        assert result is True
        assert time_taken >= 0
    
    def test_list_lookup_not_found(self):
        """Test list lookup for non-existing element."""
        data = [1, 2, 3, 4, 5]
        result, time_taken = list_lookup(data, 10)
        assert result is False
        assert time_taken >= 0
    
    def test_dict_lookup_found(self):
        """Test dict lookup for existing key."""
        data = {1: True, 2: True, 3: True}
        result, time_taken = dict_lookup(data, 2)
        assert result is True
        assert time_taken >= 0
    
    def test_dict_lookup_not_found(self):
        """Test dict lookup for non-existing key."""
        data = {1: True, 2: True, 3: True}
        result, time_taken = dict_lookup(data, 10)
        assert result is False
        assert time_taken >= 0
    
    def test_list_insert(self):
        """Test list insert operation."""
        data = [1, 2, 3]
        _, time_taken = list_insert(data, 0)
        assert data[0] == 0
        assert len(data) == 4
        assert time_taken >= 0
    
    def test_dict_insert(self):
        """Test dict insert operation."""
        data = {1: True, 2: True}
        _, time_taken = dict_insert(data, 3)
        assert 3 in data
        assert data[3] is True
        assert len(data) == 3
        assert time_taken >= 0
    
    def test_compare_performance(self):
        """Test compare_performance function."""
        results = compare_performance(size=100)
        
        assert 'size' in results
        assert results['size'] == 100
        assert 'list_lookup_time' in results
        assert 'dict_lookup_time' in results
        assert 'list_insert_time' in results
        assert 'dict_insert_time' in results
        assert 'lookup_speedup' in results
        assert 'insert_speedup' in results
        
        # Verify all times are non-negative
        assert results['list_lookup_time'] >= 0
        assert results['dict_lookup_time'] >= 0
        assert results['list_insert_time'] >= 0
        assert results['dict_insert_time'] >= 0
        
        # Verify speedup calculations
        assert results['lookup_speedup'] >= 0
        assert results['insert_speedup'] >= 0
    
    def test_compare_performance_different_sizes(self):
        """Test compare_performance with different data sizes."""
        for size in [10, 100, 1000]:
            results = compare_performance(size)
            assert results['size'] == size
            assert all(v >= 0 for k, v in results.items() if 'time' in k)
