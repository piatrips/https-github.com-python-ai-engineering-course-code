"""
Tests for collections_usage module.
"""
import pytest
from collections import Counter, defaultdict, deque
from collections_usage import (
    counter_examples,
    defaultdict_examples,
    deque_examples,
)


class TestCounter:
    """Tests for Counter usage."""
    
    def test_counter_basic(self):
        """Test basic Counter functionality."""
        text = "hello"
        counter = Counter(text)
        assert counter['l'] == 2
        assert counter['h'] == 1
        assert counter['e'] == 1
        assert counter['o'] == 1
    
    def test_counter_most_common(self):
        """Test Counter most_common method."""
        words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
        counter = Counter(words)
        most_common = counter.most_common(2)
        assert most_common[0] == ("apple", 3)
        assert most_common[1] == ("banana", 2)
    
    def test_counter_addition(self):
        """Test Counter addition."""
        c1 = Counter({'a': 3, 'b': 1})
        c2 = Counter({'a': 1, 'b': 2})
        result = c1 + c2
        assert result['a'] == 4
        assert result['b'] == 3
    
    def test_counter_subtraction(self):
        """Test Counter subtraction."""
        c1 = Counter({'a': 3, 'b': 2})
        c2 = Counter({'a': 1, 'b': 1})
        result = c1 - c2
        assert result['a'] == 2
        assert result['b'] == 1


class TestDefaultdict:
    """Tests for defaultdict usage."""
    
    def test_defaultdict_list(self):
        """Test defaultdict with list as default factory."""
        dd = defaultdict(list)
        dd['fruits'].append('apple')
        dd['fruits'].append('banana')
        dd['vegetables'].append('carrot')
        
        assert len(dd['fruits']) == 2
        assert 'apple' in dd['fruits']
        assert len(dd['vegetables']) == 1
        assert len(dd['new_key']) == 0  # Auto-creates empty list
    
    def test_defaultdict_int(self):
        """Test defaultdict with int as default factory."""
        dd = defaultdict(int)
        text = "hello"
        for char in text:
            dd[char] += 1
        
        assert dd['l'] == 2
        assert dd['h'] == 1
        assert dd['e'] == 1
        assert dd['o'] == 1
        assert dd['z'] == 0  # Default value for missing key
    
    def test_defaultdict_set(self):
        """Test defaultdict with set as default factory."""
        dd = defaultdict(set)
        dd['odds'].add(1)
        dd['odds'].add(3)
        dd['evens'].add(2)
        dd['evens'].add(4)
        
        assert 1 in dd['odds']
        assert 3 in dd['odds']
        assert 2 in dd['evens']
        assert len(dd['new_key']) == 0  # Auto-creates empty set
    
    def test_nested_defaultdict(self):
        """Test nested defaultdict."""
        nested = defaultdict(lambda: defaultdict(int))
        nested['user1']['count'] = 5
        nested['user2']['count'] = 3
        
        assert nested['user1']['count'] == 5
        assert nested['user2']['count'] == 3
        assert nested['user3']['count'] == 0  # Auto-creates nested structure


class TestDeque:
    """Tests for deque usage."""
    
    def test_deque_basic_operations(self):
        """Test basic deque operations."""
        d = deque([1, 2, 3])
        assert len(d) == 3
        assert list(d) == [1, 2, 3]
    
    def test_deque_append_operations(self):
        """Test append and appendleft operations."""
        d = deque([2, 3])
        d.append(4)
        d.appendleft(1)
        assert list(d) == [1, 2, 3, 4]
    
    def test_deque_pop_operations(self):
        """Test pop and popleft operations."""
        d = deque([1, 2, 3, 4])
        right = d.pop()
        left = d.popleft()
        assert right == 4
        assert left == 1
        assert list(d) == [2, 3]
    
    def test_deque_rotate(self):
        """Test deque rotation."""
        d = deque([1, 2, 3, 4])
        d.rotate(1)
        assert list(d) == [4, 1, 2, 3]
        
        d.rotate(-2)
        assert list(d) == [2, 3, 4, 1]
    
    def test_deque_maxlen(self):
        """Test deque with maximum length (circular buffer)."""
        d = deque(maxlen=3)
        d.append(1)
        d.append(2)
        d.append(3)
        assert list(d) == [1, 2, 3]
        
        d.append(4)  # Should push out 1
        assert list(d) == [2, 3, 4]
        
        d.append(5)  # Should push out 2
        assert list(d) == [3, 4, 5]
    
    def test_deque_as_stack(self):
        """Test using deque as a stack (LIFO)."""
        stack = deque()
        stack.append(1)
        stack.append(2)
        stack.append(3)
        
        assert stack.pop() == 3
        assert stack.pop() == 2
        assert stack.pop() == 1
        assert len(stack) == 0
    
    def test_deque_as_queue(self):
        """Test using deque as a queue (FIFO)."""
        queue = deque()
        queue.append(1)
        queue.append(2)
        queue.append(3)
        
        assert queue.popleft() == 1
        assert queue.popleft() == 2
        assert queue.popleft() == 3
        assert len(queue) == 0
    
    def test_deque_extend(self):
        """Test deque extend operations."""
        d = deque([2, 3])
        d.extend([4, 5])
        d.extendleft([1, 0])
        assert list(d) == [0, 1, 2, 3, 4, 5]
