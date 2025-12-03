"""
Dataset and DataLoader utilities for maze solver.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image


class MazeDataset(Dataset):
    """
    Dataset for maze images and their solution sequences.
    
    Args:
        entries: List of dicts with keys 'id', 'image', 'sequence'
        tokenizer: Tokenizer instance
        transform: torchvision transforms for images
    """
    
    def __init__(self, entries, tokenizer, transform=None):
        self.entries = entries
        self.tokenizer = tokenizer
        self.transform = transform
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        
        # Load image
        image = Image.open(entry['image']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Tokenize sequence: ['R', 'U', 'U'] -> [1, 4, 5, 5, 2]
        token_ids = self.tokenizer.encode(entry['sequence'])
        
        return {
            'image': image,
            'labels': torch.tensor(token_ids, dtype=torch.long),
            'maze_id': entry['id']
        }


def collate_fn(batch, pad_token_id=0):
    """
    Collate function for DataLoader.
    Pads sequences to the same length in a batch.
    
    Args:
        batch: List of samples from MazeDataset
        pad_token_id: Token ID to use for padding (default: 0)
    
    Returns:
        Dict with batched 'images', 'labels', 'maze_ids'
    """
    images = torch.stack([item['image'] for item in batch])
    
    # Pad label sequences to same length
    labels = torch.nn.utils.rnn.pad_sequence(
        [item['labels'] for item in batch],
        batch_first=True,
        padding_value=pad_token_id
    )
    
    maze_ids = [item['maze_id'] for item in batch]
    
    return {
        'images': images,
        'labels': labels,
        'maze_ids': maze_ids
    }
