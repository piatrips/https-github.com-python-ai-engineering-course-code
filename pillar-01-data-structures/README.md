# Pillar 01: Data Structures

This module contains comprehensive examples and demonstrations of Python data structures for AI/ML engineering.

## Contents

### 1. Lists vs Dictionaries Performance Comparison
**File:** `list_vs_dict_performance.py`

Demonstrates performance differences between lists and dictionaries for:
- Lookup operations (O(n) vs O(1))
- Insert operations
- Performance metrics at different data sizes

**Run the example:**
```bash
python list_vs_dict_performance.py
```

### 2. Generators for Lazy Data Loading
**File:** `generators_lazy_loading.py`

Shows how to use generators for memory-efficient data processing:
- Eager vs lazy loading comparison
- Fibonacci sequence generator
- Data chunking with generators
- Generator expressions
- File reading with generators

**Run the example:**
```bash
python generators_lazy_loading.py
```

### 3. Collections Module Usage
**File:** `collections_usage.py`

Demonstrates specialized containers from the collections module:
- **Counter**: Counting hashable objects
- **defaultdict**: Dictionaries with default values
- **deque**: Double-ended queue for efficient operations

**Run the example:**
```bash
python collections_usage.py
```

### 4. Custom Dataset Class
**File:** `custom_dataset.py`

Implements custom Dataset classes with `__len__` and `__getitem__` methods:
- Base Dataset class
- TextDataset with transformations
- ImageDataset for simulated image data
- DataLoader for batching data

**Run the example:**
```bash
python custom_dataset.py
```

## Testing

All modules include comprehensive pytest tests.

### Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

### Run all tests:
```bash
pytest
```

### Run tests with verbose output:
```bash
pytest -v
```

### Run tests for a specific module:
```bash
pytest test_list_vs_dict_performance.py -v
pytest test_generators_lazy_loading.py -v
pytest test_collections_usage.py -v
pytest test_custom_dataset.py -v
```

### Run tests with coverage:
```bash
pytest --cov=. --cov-report=html
```

## Key Concepts

### When to Use Lists vs Dictionaries

**Use Lists when:**
- Order matters
- You need to access elements by index
- You need to iterate over all elements
- You're primarily appending to the end

**Use Dictionaries when:**
- You need fast lookups by key
- You need to check if a key exists
- You need to map keys to values
- Order doesn't matter (or use OrderedDict)

### When to Use Generators

**Use Generators when:**
- Working with large datasets that don't fit in memory
- You only need to iterate once
- You want to pipeline data transformations
- You need lazy evaluation

### Collections Module Best Practices

- **Counter**: Count occurrences, find most common elements
- **defaultdict**: Avoid KeyError when accessing missing keys
- **deque**: Efficient append/pop from both ends, useful for queues

### Dataset Protocol

The `__len__` and `__getitem__` methods allow your custom classes to:
- Work with Python's built-in functions (len(), indexing)
- Be used with data loaders and iterators
- Integrate with ML frameworks like PyTorch and TensorFlow
