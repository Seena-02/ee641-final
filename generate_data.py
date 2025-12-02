import random
import math
import os
import json
import argparse
from PIL import Image, ImageDraw, ImageFont

def generate_grid_from_path(path, rows, cols, save_path, obstacle_prob=0.6, cell_size=80, seed=None):
    move_map = {"U": (-1,0), "D": (1,0), "L": (0,-1), "R": (0,1)}
    rng_local = random.Random(seed)

    # Starting at bottom-left
    start = (rows-1, 0)
    r, c = start
    path_cells = [start]
    
    # Build path coordinates
    for move in path:
        dr, dc = move_map[move]
        r, c = r + dr, c + dc
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError("Path goes out of grid bounds.")
        path_cells.append((r, c))
    
    goal = path_cells[-1]
    
    # Build grid
    grid = [["0" for _ in range(cols)] for _ in range(rows)]
    grid[start[0]][start[1]] = "S"
    grid[goal[0]][goal[1]] = "G"
    
    # Place obstacles off-path
    for rr in range(rows):
        for cc in range(cols):
            if (rr, cc) not in path_cells and rng_local.random() < obstacle_prob:
                grid[rr][cc] = "X"
    
    # Image colors
    colors = {
        "S": (0,200,0),
        "G": (0,0,200),
        "X": (255,0,0),
        "0": (255,255,255),
    }
    
    img = Image.new("RGB", (cols*cell_size, rows*cell_size), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", int(cell_size / 2))
    except:
        font = ImageFont.load_default()
    
    # Draw the grid
    for rr in range(rows):
        for cc in range(cols):
            cell = grid[rr][cc]
            x1, y1 = cc * cell_size, rr * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size

            draw.rectangle([x1, y1, x2, y2], fill=colors[cell], outline=(100, 100, 100))

            # Draw label for S,G
            if cell in ["S", "G"]:
                bbox = draw.textbbox((0, 0), cell, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]

                draw.text(
                    (x1 + cell_size/2 - w/2, y1 + cell_size/2 - h/2),
                    cell,
                    fill="white",
                    font=font
                )

    img.save(save_path)
    return grid

def generate_random_unique_paths(rows, cols, rng, num_sequences=50):
    """Generate unique random monotonic (U/R only) paths with reproducibility."""
    up_moves = rows - 1
    right_moves = cols - 1
    base_moves = ["U"] * up_moves + ["R"] * right_moves

    paths = set()
    
    while len(paths) < num_sequences:
        m = base_moves[:]
        rng.shuffle(m)
        paths.add(tuple(m))

    return [list(p) for p in paths]

def num_monotonic_paths(rows, cols):
    return math.comb(rows + cols - 2, rows - 1)

def generate_grids_with_variations(paths, rows, cols, base_seed, num_variations=5, train_split=0.8):
    """
    Generate multiple maze variations for each solution path.
    Split by solution pattern, not by individual mazes.
    """
    # Split paths into train and test FIRST
    rng_split = random.Random(base_seed)
    paths_shuffled = paths.copy()
    rng_split.shuffle(paths_shuffled)
    
    split_idx = int(len(paths_shuffled) * train_split)
    train_paths = paths_shuffled[:split_idx]
    test_paths = paths_shuffled[split_idx:]
    
    print(f"\n{'='*60}")
    print(f"DATASET SPLIT")
    print(f"{'='*60}")
    print(f"Total unique solution patterns: {len(paths)}")
    print(f"Train solution patterns: {len(train_paths)}")
    print(f"Test solution patterns: {len(test_paths)}")
    print(f"Variations per solution: {num_variations}")
    print(f"Total train images: {len(train_paths) * num_variations}")
    print(f"Total test images: {len(test_paths) * num_variations}")
    print(f"{'='*60}\n")
    
    train_data = []
    test_data = []
    
    # Generate train set
    print("Generating TRAIN set...")
    for path_idx, path in enumerate(train_paths):
        for var_idx in range(num_variations):
            # Unique seed for each variation
            variation_seed = base_seed + path_idx * 1000 + var_idx
            
            image_path = f"data/grids/train/grid_{len(train_data)}.png"
            generate_grid_from_path(
                path, rows, cols, 
                save_path=image_path,
                seed=variation_seed
            )
            
            train_data.append({
                "id": len(train_data),
                "sequence": path,
                "image": image_path,
                "solution_id": path_idx,
                "variation": var_idx
            })
    
    # Generate test set
    print("Generating TEST set...")
    for path_idx, path in enumerate(test_paths):
        for var_idx in range(num_variations):
            # Different seed space for test to ensure different wall patterns
            variation_seed = base_seed + 500000 + path_idx * 1000 + var_idx
            
            image_path = f"data/grids/test/grid_{len(test_data)}.png"
            generate_grid_from_path(
                path, rows, cols,
                save_path=image_path,
                seed=variation_seed
            )
            
            test_data.append({
                "id": len(test_data),
                "sequence": path,
                "image": image_path,
                "solution_id": path_idx,
                "variation": var_idx
            })
    
    return train_data, test_data

def save_datasets(train_data, test_data):
    """Save train and test datasets to separate JSON files."""
    with open("data/train_sequences.json", "w") as f:
        json.dump(train_data, f, indent=4)
    
    with open("data/test_sequences.json", "w") as f:
        json.dump(test_data, f, indent=4)
    
    print(f"\n✓ Saved {len(train_data)} train examples to data/train_sequences.json")
    print(f"✓ Saved {len(test_data)} test examples to data/test_sequences.json")

if __name__ == "__main__":
    # Create directories
    os.makedirs("data/grids/train", exist_ok=True)
    os.makedirs("data/grids/test", exist_ok=True)

    parser = argparse.ArgumentParser(description='Generate maze dataset with variations')
    parser.add_argument('--seed', type=int, default=641, help='Random seed')
    parser.add_argument('--cols', type=int, default=7, help='Number of cols')
    parser.add_argument('--rows', type=int, default=7, help='Number of rows')
    parser.add_argument('--variations', type=int, default=5, 
                        help='Number of wall variations per solution')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Proportion of solutions for training (0.8 = 80%)')
 
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Generate all unique solution patterns
    max_num_sequences = num_monotonic_paths(args.rows, args.cols)
    paths = generate_random_unique_paths(args.rows, args.cols, rng, max_num_sequences)
    paths = sorted(paths)  # Deterministic order
    
    print(f"Generated {len(paths)} unique solution patterns")
    
    # Generate train and test sets with variations
    train_data, test_data = generate_grids_with_variations(
        paths, args.rows, args.cols, args.seed, 
        num_variations=args.variations,
        train_split=args.train_split
    )
    
    # Save datasets
    save_datasets(train_data, test_data)
    
    print(f"\n{'='*60}")
    print("✓ Dataset generation complete!")
    print(f"{'='*60}")
    print(f"Train images: data/grids/train/")
    print(f"Test images: data/grids/test/")
    print(f"Train JSON: data/train_sequences.json")
    print(f"Test JSON: data/test_sequences.json")
    print(f"{'='*60}\n")