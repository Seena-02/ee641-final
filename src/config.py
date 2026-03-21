"""
Grid-size-aware model and training configurations for the maze solver.

Model capacity is motivated by solution space complexity: C(2n-2, n-1).
The solution space grows combinatorially, so required representational
capacity scales roughly logarithmically with it — linear in grid size n.

    Grid  Solutions  log2(solutions)  Config
    4×4   20         4.3 bits         hidden=128, layers=2
    5×5   70         6.1 bits         hidden=128, layers=2
    7×7   924        9.8 bits         hidden=256, layers=4
"""

import math


def num_solutions(n):
    """Number of monotonic (U/R) paths for an n×n grid: C(2n-2, n-1)."""
    return math.comb(2 * (n - 1), n - 1)


# Per-grid-size model and training hyperparameters.
# model_kwargs maps directly to ResNetGPT2PrefixModel constructor args.
GRID_CONFIGS = {
    4: {
        "model_kwargs": {
            "hidden_size": 128,
            "num_layers": 2,
            "num_attention_heads": 2,
            "num_prefix_tokens": 16,
            "dropout": 0.4,
            "resnet_frozen": True,
        },
        "train_kwargs": {
            "epochs": 40,
            "lr": 5e-4,
        },
    },
    5: {
        "model_kwargs": {
            "hidden_size": 128,
            "num_layers": 2,
            "num_attention_heads": 2,
            "num_prefix_tokens": 16,
            "dropout": 0.4,
            "resnet_frozen": True,
        },
        "train_kwargs": {
            "epochs": 75,
            "lr": 5e-4,
        },
    },
    7: {
        "model_kwargs": {
            "hidden_size": 256,
            "num_layers": 4,
            "num_attention_heads": 4,
            "num_prefix_tokens": 16,
            "dropout": 0.4,
            "resnet_frozen": True,
        },
        "train_kwargs": {
            "epochs": 100,
            "lr": 5e-4,
        },
    },
}


def get_config(grid_size):
    """
    Return the config for a given grid size.
    Raises KeyError with a helpful message if the size is not configured.
    """
    if grid_size not in GRID_CONFIGS:
        supported = sorted(GRID_CONFIGS.keys())
        raise KeyError(
            f"No config for grid size {grid_size}. "
            f"Supported sizes: {supported}. "
            f"Add an entry to GRID_CONFIGS in src/config.py."
        )
    return GRID_CONFIGS[grid_size]
