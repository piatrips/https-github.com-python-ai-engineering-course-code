"""
Collections module usage examples.

This module demonstrates the usage of specialized container datatypes from the
collections module: Counter, defaultdict, and deque.
"""
from collections import Counter, defaultdict, deque
from typing import List, Dict


def counter_examples():
    """Demonstrate Counter usage for counting hashable objects."""
    print("\n1. Counter Examples:")
    print("-" * 50)
    
    # Count characters in a string
    text = "hello world"
    char_count = Counter(text)
    print(f"   Character counts in '{text}':")
    print(f"   {dict(char_count)}")
    print(f"   Most common: {char_count.most_common(3)}")
    
    # Count words in a list
    words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
    word_count = Counter(words)
    print(f"\n   Word counts: {dict(word_count)}")
    print(f"   'apple' appears {word_count['apple']} times")
    
    # Combine counters
    counter1 = Counter({'a': 3, 'b': 1})
    counter2 = Counter({'a': 1, 'b': 2})
    combined = counter1 + counter2
    print(f"\n   Counter1: {dict(counter1)}")
    print(f"   Counter2: {dict(counter2)}")
    print(f"   Combined: {dict(combined)}")


def defaultdict_examples():
    """Demonstrate defaultdict usage for handling missing keys."""
    print("\n2. defaultdict Examples:")
    print("-" * 50)
    
    # Group items by category
    items = [
        ("fruit", "apple"),
        ("vegetable", "carrot"),
        ("fruit", "banana"),
        ("vegetable", "lettuce"),
        ("fruit", "cherry"),
    ]
    
    # Using defaultdict with list
    grouped = defaultdict(list)
    for category, item in items:
        grouped[category].append(item)
    
    print("   Grouped items:")
    for category, items_list in grouped.items():
        print(f"   {category}: {items_list}")
    
    # Using defaultdict with int for counting
    text = "hello world"
    char_count = defaultdict(int)
    for char in text:
        char_count[char] += 1
    
    print(f"\n   Character counts using defaultdict(int):")
    print(f"   {dict(char_count)}")
    
    # Nested defaultdict
    nested = defaultdict(lambda: defaultdict(int))
    nested['user1']['login_count'] = 5
    nested['user1']['posts'] = 10
    nested['user2']['login_count'] = 3
    
    print(f"\n   Nested defaultdict:")
    for user, stats in nested.items():
        print(f"   {user}: {dict(stats)}")


def deque_examples():
    """Demonstrate deque usage for efficient queue and stack operations."""
    print("\n3. deque Examples:")
    print("-" * 50)
    
    # Basic deque operations
    d = deque([1, 2, 3])
    print(f"   Initial deque: {list(d)}")
    
    d.append(4)  # Add to right
    print(f"   After append(4): {list(d)}")
    
    d.appendleft(0)  # Add to left
    print(f"   After appendleft(0): {list(d)}")
    
    d.pop()  # Remove from right
    print(f"   After pop(): {list(d)}")
    
    d.popleft()  # Remove from left
    print(f"   After popleft(): {list(d)}")
    
    # Rotate deque
    d.rotate(1)  # Rotate right
    print(f"   After rotate(1): {list(d)}")
    
    d.rotate(-2)  # Rotate left
    print(f"   After rotate(-2): {list(d)}")
    
    # Limited-size deque (circular buffer)
    circular = deque(maxlen=3)
    for i in range(5):
        circular.append(i)
        print(f"   After appending {i}: {list(circular)}")
    
    # Using deque as a stack (LIFO)
    stack = deque()
    stack.append(1)
    stack.append(2)
    stack.append(3)
    print(f"\n   Stack (LIFO): {list(stack)}")
    print(f"   Pop: {stack.pop()}")
    print(f"   Remaining: {list(stack)}")
    
    # Using deque as a queue (FIFO)
    queue = deque()
    queue.append(1)
    queue.append(2)
    queue.append(3)
    print(f"\n   Queue (FIFO): {list(queue)}")
    print(f"   Pop from left: {queue.popleft()}")
    print(f"   Remaining: {list(queue)}")


def practical_example():
    """Demonstrate a practical use case combining all three collections."""
    print("\n4. Practical Example: Web Server Log Analysis")
    print("-" * 50)
    
    # Simulated log entries: (url, status_code, response_time)
    logs = [
        ("/home", 200, 0.05),
        ("/api/users", 200, 0.12),
        ("/home", 200, 0.04),
        ("/api/users", 404, 0.02),
        ("/api/data", 200, 0.15),
        ("/home", 200, 0.06),
        ("/api/users", 200, 0.11),
    ]
    
    # Use Counter to count status codes
    status_counts = Counter(log[1] for log in logs)
    print(f"   Status code distribution: {dict(status_counts)}")
    
    # Use defaultdict to group response times by URL
    url_response_times = defaultdict(list)
    for url, status, response_time in logs:
        url_response_times[url].append(response_time)
    
    print("\n   Average response times by URL:")
    for url, times in url_response_times.items():
        avg_time = sum(times) / len(times)
        print(f"   {url}: {avg_time:.3f}s")
    
    # Use deque to keep recent logs (circular buffer)
    recent_logs = deque(maxlen=3)
    for log in logs:
        recent_logs.append(log)
    
    print(f"\n   Most recent 3 logs:")
    for log in recent_logs:
        print(f"   {log}")


def demonstrate_collections():
    """Run all collections demonstrations."""
    print("\nCollections Module Usage Examples\n")
    print("=" * 70)
    
    counter_examples()
    defaultdict_examples()
    deque_examples()
    practical_example()


if __name__ == "__main__":
    demonstrate_collections()
