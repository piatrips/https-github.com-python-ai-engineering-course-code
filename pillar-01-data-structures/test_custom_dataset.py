"""
Tests for custom_dataset module.
"""
import pytest
from custom_dataset import (
    Dataset,
    TextDataset,
    ImageDataset,
    DataLoader,
)


class TestDataset:
    """Tests for base Dataset class."""
    
    def test_dataset_creation(self):
        """Test creating a basic dataset."""
        data = [1, 2, 3, 4, 5]
        dataset = Dataset(data)
        assert len(dataset) == 5
    
    def test_dataset_len(self):
        """Test __len__ method."""
        dataset = Dataset([1, 2, 3])
        assert len(dataset) == 3
        
        empty_dataset = Dataset([])
        assert len(empty_dataset) == 0
    
    def test_dataset_getitem(self):
        """Test __getitem__ method."""
        data = [10, 20, 30, 40, 50]
        dataset = Dataset(data)
        
        assert dataset[0] == 10
        assert dataset[1] == 20
        assert dataset[4] == 50
    
    def test_dataset_getitem_out_of_bounds(self):
        """Test __getitem__ with out of bounds index."""
        dataset = Dataset([1, 2, 3])
        
        with pytest.raises(IndexError):
            _ = dataset[10]
        
        with pytest.raises(IndexError):
            _ = dataset[-1]
    
    def test_dataset_repr(self):
        """Test string representation."""
        dataset = Dataset([1, 2, 3, 4, 5])
        assert "Dataset" in repr(dataset)
        assert "5" in repr(dataset)
    
    def test_dataset_iteration(self):
        """Test iterating over dataset."""
        data = [1, 2, 3, 4, 5]
        dataset = Dataset(data)
        result = [dataset[i] for i in range(len(dataset))]
        assert result == data


class TestTextDataset:
    """Tests for TextDataset class."""
    
    def test_text_dataset_no_transform(self):
        """Test TextDataset without transformation."""
        texts = ["hello", "world"]
        dataset = TextDataset(texts)
        
        assert len(dataset) == 2
        assert dataset[0] == "hello"
        assert dataset[1] == "world"
    
    def test_text_dataset_with_transform(self):
        """Test TextDataset with transformation."""
        texts = ["hello", "world"]
        dataset = TextDataset(texts, transform=str.upper)
        
        assert len(dataset) == 2
        assert dataset[0] == "HELLO"
        assert dataset[1] == "WORLD"
    
    def test_text_dataset_custom_transform(self):
        """Test TextDataset with custom transformation."""
        texts = ["hello", "world"]
        dataset = TextDataset(texts, transform=lambda x: x[::-1])  # Reverse
        
        assert dataset[0] == "olleh"
        assert dataset[1] == "dlrow"
    
    def test_text_dataset_multiple_transforms(self):
        """Test TextDataset with chained transformations."""
        texts = ["hello world", "python programming"]
        
        def transform(text):
            return text.upper().replace(" ", "_")
        
        dataset = TextDataset(texts, transform=transform)
        assert dataset[0] == "HELLO_WORLD"
        assert dataset[1] == "PYTHON_PROGRAMMING"


class TestImageDataset:
    """Tests for ImageDataset class."""
    
    def test_image_dataset_creation(self):
        """Test creating an image dataset."""
        dataset = ImageDataset(size=10, num_classes=5)
        assert len(dataset) == 10
    
    def test_image_dataset_items(self):
        """Test image dataset returns proper items."""
        dataset = ImageDataset(size=5, num_classes=3)
        
        for i in range(len(dataset)):
            image_shape, label = dataset[i]
            assert image_shape == (28, 28, 3)
            assert 0 <= label < 3
    
    def test_image_dataset_labels_in_range(self):
        """Test that all labels are within valid range."""
        num_classes = 5
        dataset = ImageDataset(size=20, num_classes=num_classes)
        
        labels = [dataset[i][1] for i in range(len(dataset))]
        assert all(0 <= label < num_classes for label in labels)
    
    def test_image_dataset_different_sizes(self):
        """Test creating datasets with different sizes."""
        for size in [1, 10, 100]:
            dataset = ImageDataset(size=size)
            assert len(dataset) == size


class TestDataLoader:
    """Tests for DataLoader class."""
    
    def test_dataloader_creation(self):
        """Test creating a data loader."""
        dataset = Dataset([1, 2, 3, 4, 5])
        loader = DataLoader(dataset, batch_size=2)
        assert len(loader) == 3  # 5 items / 2 per batch = 3 batches
    
    def test_dataloader_batching(self):
        """Test data loader creates proper batches."""
        dataset = Dataset([1, 2, 3, 4, 5])
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        batches = list(loader)
        assert len(batches) == 3
        assert batches[0] == [1, 2]
        assert batches[1] == [3, 4]
        assert batches[2] == [5]  # Last batch has remaining items
    
    def test_dataloader_exact_batches(self):
        """Test data loader when size is exactly divisible by batch size."""
        dataset = Dataset([1, 2, 3, 4, 5, 6])
        loader = DataLoader(dataset, batch_size=3, shuffle=False)
        
        batches = list(loader)
        assert len(batches) == 2
        assert batches[0] == [1, 2, 3]
        assert batches[1] == [4, 5, 6]
    
    def test_dataloader_single_batch(self):
        """Test data loader with batch size equal to dataset size."""
        dataset = Dataset([1, 2, 3])
        loader = DataLoader(dataset, batch_size=10, shuffle=False)
        
        batches = list(loader)
        assert len(batches) == 1
        assert batches[0] == [1, 2, 3]
    
    def test_dataloader_shuffle(self):
        """Test that shuffle parameter affects order."""
        dataset = Dataset(list(range(100)))
        
        # Without shuffle
        loader_no_shuffle = DataLoader(dataset, batch_size=10, shuffle=False)
        batches_no_shuffle = list(loader_no_shuffle)
        first_batch_no_shuffle = batches_no_shuffle[0]
        
        # With shuffle - run multiple times to ensure it eventually differs
        shuffled = False
        for _ in range(10):
            loader_shuffle = DataLoader(dataset, batch_size=10, shuffle=True)
            batches_shuffle = list(loader_shuffle)
            first_batch_shuffle = batches_shuffle[0]
            if first_batch_shuffle != first_batch_no_shuffle:
                shuffled = True
                break
        
        # At least one shuffle should produce different order
        assert shuffled or len(dataset) < 2  # Allow for small datasets
    
    def test_dataloader_empty_dataset(self):
        """Test data loader with empty dataset."""
        dataset = Dataset([])
        loader = DataLoader(dataset, batch_size=2)
        
        batches = list(loader)
        assert len(batches) == 0
    
    def test_dataloader_with_text_dataset(self):
        """Test data loader works with TextDataset."""
        texts = ["hello", "world", "python", "test"]
        dataset = TextDataset(texts, transform=str.upper)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        batches = list(loader)
        assert len(batches) == 2
        assert batches[0] == ["HELLO", "WORLD"]
        assert batches[1] == ["PYTHON", "TEST"]
