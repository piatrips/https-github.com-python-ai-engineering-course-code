"""
Custom Dataset class with __len__ and __getitem__ methods.

This module demonstrates how to create a custom Dataset class that can be used
with data loaders and other Python tools that expect sequence-like objects.
"""
from typing import List, Tuple, Optional, Callable
import random


class Dataset:
    """
    Base Dataset class with __len__ and __getitem__ protocol.
    
    This class implements the sequence protocol, making it compatible with
    iteration, indexing, and length operations.
    """
    
    def __init__(self, data: List):
        """
        Initialize the dataset with data.
        
        Args:
            data: List of data items
        """
        self.data = data
    
    def __len__(self) -> int:
        """
        Return the number of items in the dataset.
        
        Returns:
            Length of the dataset
        """
        return len(self.data)
    
    def __getitem__(self, index: int):
        """
        Get an item by index.
        
        Args:
            index: Index of the item to retrieve
            
        Returns:
            Item at the specified index
            
        Raises:
            IndexError: If index is out of bounds
        """
        if index < 0 or index >= len(self.data):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.data)}")
        return self.data[index]
    
    def __repr__(self) -> str:
        """String representation of the dataset."""
        return f"Dataset(size={len(self.data)})"


class TextDataset(Dataset):
    """
    Dataset for text data with optional preprocessing.
    
    This class extends the base Dataset to handle text processing tasks.
    """
    
    def __init__(self, texts: List[str], transform: Optional[Callable] = None):
        """
        Initialize text dataset.
        
        Args:
            texts: List of text strings
            transform: Optional transform function to apply to each text
        """
        super().__init__(texts)
        self.transform = transform
    
    def __getitem__(self, index: int) -> str:
        """
        Get a text item with optional transformation.
        
        Args:
            index: Index of the item
            
        Returns:
            Text item (transformed if transform is set)
        """
        text = super().__getitem__(index)
        if self.transform:
            text = self.transform(text)
        return text


class ImageDataset(Dataset):
    """
    Simulated image dataset with labels.
    
    This class simulates an image dataset with (image, label) pairs.
    """
    
    def __init__(self, size: int, num_classes: int = 10):
        """
        Initialize simulated image dataset.
        
        Args:
            size: Number of samples
            num_classes: Number of different classes/labels
        """
        # Simulate images as tuples of (height, width, channels)
        self.images = [(28, 28, 3) for _ in range(size)]
        self.labels = [random.randint(0, num_classes - 1) for _ in range(size)]
        super().__init__(list(zip(self.images, self.labels)))
    
    def __getitem__(self, index: int) -> Tuple:
        """
        Get an (image, label) pair.
        
        Args:
            index: Index of the item
            
        Returns:
            Tuple of (image_shape, label)
        """
        return super().__getitem__(index)


class DataLoader:
    """
    Simple data loader that batches data from a dataset.
    
    This demonstrates how a Dataset with __len__ and __getitem__ can be used
    by other components.
    """
    
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False):
        """
        Initialize data loader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Number of items per batch
            shuffle: Whether to shuffle data
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
    
    def __iter__(self):
        """Create an iterator over batches."""
        if self.shuffle:
            random.shuffle(self.indices)
        
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield batch
    
    def __len__(self) -> int:
        """Return number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def demonstrate_dataset():
    """Demonstrate custom Dataset usage."""
    print("\nCustom Dataset Class Demonstrations\n")
    print("=" * 70)
    
    # Example 1: Basic Dataset
    print("\n1. Basic Dataset:")
    print("-" * 50)
    data = [1, 2, 3, 4, 5]
    dataset = Dataset(data)
    print(f"   Dataset: {dataset}")
    print(f"   Length: {len(dataset)}")
    print(f"   First item: {dataset[0]}")
    print(f"   Last item: {dataset[-1 + len(dataset)]}")
    
    # Iterate over dataset
    print("   All items:", end=" ")
    for item in [dataset[i] for i in range(len(dataset))]:
        print(item, end=" ")
    print()
    
    # Example 2: TextDataset with transform
    print("\n2. TextDataset with Preprocessing:")
    print("-" * 50)
    texts = ["hello world", "python programming", "data structures"]
    text_dataset = TextDataset(texts, transform=str.upper)
    print(f"   Dataset: {text_dataset}")
    print(f"   Original texts: {texts}")
    print(f"   Transformed texts:")
    for i in range(len(text_dataset)):
        print(f"   {i}: {text_dataset[i]}")
    
    # Example 3: ImageDataset
    print("\n3. Simulated ImageDataset:")
    print("-" * 50)
    image_dataset = ImageDataset(size=5, num_classes=3)
    print(f"   Dataset: {image_dataset}")
    print(f"   Sample items:")
    for i in range(min(3, len(image_dataset))):
        image_shape, label = image_dataset[i]
        print(f"   {i}: image_shape={image_shape}, label={label}")
    
    # Example 4: DataLoader
    print("\n4. DataLoader with Batching:")
    print("-" * 50)
    dataset = Dataset(list(range(10)))
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Batch size: 3")
    print(f"   Number of batches: {len(loader)}")
    print("   Batches:")
    for i, batch in enumerate(loader):
        print(f"   Batch {i}: {batch}")


if __name__ == "__main__":
    demonstrate_dataset()
