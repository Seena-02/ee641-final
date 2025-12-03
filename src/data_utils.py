"""
Data loading utilities for maze dataset.
Handles loading JSON files with metadata.
"""

import json


def load_maze_dataset(json_path, return_metadata=True):
    """
    Load maze dataset from JSON file.
    
    Args:
        json_path: Path to train_sequences.json or test_sequences.json
        return_metadata: If True, returns (entries, metadata), else just entries
    
    Returns:
        If return_metadata=True: (entries, metadata)
        If return_metadata=False: entries
        
    Example:
        train_entries, metadata = load_maze_dataset('data/train_sequences.json')
        GRID_SIZE = metadata['grid_size']
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check if it's the new format with metadata
    if isinstance(data, dict) and 'metadata' in data and 'entries' in data:
        entries = data['entries']
        metadata = data['metadata']
    else:
        # Old format (backward compatibility)
        entries = data
        metadata = {
            'grid_size': 7,  # Default
            'rows': 7,
            'cols': 7,
            'seed': None,
            'variations': None,
            'train_split': None,
            'start_position': [0, 6],
            'goal_position': [6, 0]
        }
        print(f"⚠️  No metadata found in {json_path}, using defaults")
    
    if return_metadata:
        return entries, metadata
    else:
        return entries


def print_dataset_info(metadata, num_entries, dataset_name="Dataset"):
    """
    Pretty print dataset information.
    
    Args:
        metadata: Metadata dictionary from load_maze_dataset
        num_entries: Number of entries in the dataset
        dataset_name: Name to display (e.g., "Training" or "Test")
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name.upper()} SET INFO")
    print(f"{'='*60}")
    print(f"Grid size:       {metadata['rows']}×{metadata['cols']}")
    print(f"Total entries:   {num_entries}")
    print(f"Variations:      {metadata.get('variations', 'N/A')}")
    print(f"Seed:            {metadata.get('seed', 'N/A')}")
    print(f"Start position:  {tuple(metadata.get('start_position', [0, metadata['rows']-1]))}")
    print(f"Goal position:   {tuple(metadata.get('goal_position', [metadata['cols']-1, 0]))}")
    print(f"{'='*60}\n")