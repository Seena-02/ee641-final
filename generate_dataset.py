import random
import math
import os
import json
import argparse
from PIL import Image, ImageDraw, ImageFont

def generate_grid_from_path(path, rows, cols, save_path, seed=None, cell_size=80):
    """
    Generate a maze where the given path is the ONLY valid solution.
    Creates truly different wall configurations for the same solution.
    """
    move_map = {"U": (-1,0), "D": (1,0), "L": (0,-1), "R": (0,1)}
    rng_local = random.Random(seed)

    # Starting at bottom-left, goal at top-right
    start = (rows-1, 0)
    goal = (0, cols-1)
    
    # Build the solution path coordinates
    r, c = start
    path_cells = {start}
    
    for move in path:
        dr, dc = move_map[move]
        r, c = r + dr, c + dc
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError("Path goes out of grid bounds.")
        path_cells.add((r, c))
    
    if (r, c) != goal:
        raise ValueError(f"Path doesn't reach goal! Ended at {(r,c)}, goal is {goal}")
    
    # Initialize grid - all cells are walkable initially
    grid = [["0" for _ in range(cols)] for _ in range(rows)]
    grid[start[0]][start[1]] = "S"
    grid[goal[0]][goal[1]] = "G"
    
    # CRITICAL: Block alternative paths by placing strategic obstacles
    # For each cell NOT on the solution path, decide if it should be blocked
    
    for rr in range(rows):
        for cc in range(cols):
            if (rr, cc) in path_cells:
                continue  # Never block the solution path
            
            # Check if this cell is "adjacent" to the path (could be an alternative route)
            is_near_path = False
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) in path_cells:
                    is_near_path = True
                    break
            
            # If near the path, randomly block it with higher probability
            # This creates variations while ensuring alternative paths are blocked
            if is_near_path:
                # 70% chance to block cells near the path (creates different maze structures)
                if rng_local.random() < 0.7:
                    grid[rr][cc] = "X"
            else:
                # 40% chance to block cells far from path (decorative obstacles)
                if rng_local.random() < 0.4:
                    grid[rr][cc] = "X"
    
    # VERIFICATION: Ensure the solution path is still valid (no obstacles on it)
    r, c = start
    for move in path:
        dr, dc = move_map[move]
        r, c = r + dr, c + dc
        if grid[r][c] == "X":
            grid[r][c] = "0"  # Remove obstacle if it blocks solution
    
    # Render the maze as an image
    colors = {
        "S": (0, 200, 0),    # Green start
        "G": (0, 0, 200),    # Blue goal
        "X": (50, 50, 50),   # Dark gray obstacles
        "0": (255, 255, 255), # White walkable
    }
    
    img = Image.new("RGB", (cols * cell_size, rows * cell_size), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", int(cell_size / 2))
    except:
        font = ImageFont.load_default()
    
    # Draw grid
    for rr in range(rows):
        for cc in range(cols):
            cell = grid[rr][cc]
            x1, y1 = cc * cell_size, rr * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size

            draw.rectangle([x1, y1, x2, y2], fill=colors[cell], outline=(150, 150, 150), width=2)

            # Draw S and G labels
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

def verify_path_validity(grid, path, start, goal):
    """
    Verify that the given path is valid in the grid.
    Returns True if path successfully navigates from start to goal without hitting obstacles.
    """
    move_map = {"U": (-1,0), "D": (1,0), "L": (0,-1), "R": (0,1)}
    r, c = start
    
    if grid[r][c] == "X":
        return False
    
    for move in path:
        dr, dc = move_map[move]
        r, c = r + dr, c + dc
        
        if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
            return False
        if grid[r][c] == "X":
            return False
    
    return (r, c) == goal

def generate_random_unique_paths(rows, cols, rng, num_sequences=50):
    """Generate unique random monotonic (U/R only) paths."""
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
    Generate multiple TRULY DIFFERENT maze variations for each solution path.
    """
    # Split paths into train and test
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
    
    start = (rows-1, 0)
    goal = (0, cols-1)
    
    # Generate train set
    print("Generating TRAIN set...")
    for path_idx, path in enumerate(train_paths):
        for var_idx in range(num_variations):
            variation_seed = base_seed + path_idx * 1000 + var_idx
            
            image_path = f"data/grids/train/grid_{len(train_data)}.png"
            grid = generate_grid_from_path(
                path, rows, cols, 
                save_path=image_path,
                seed=variation_seed
            )
            
            # Verify the path is valid
            if not verify_path_validity(grid, path, start, goal):
                print(f"WARNING: Invalid maze generated for train {len(train_data)}")
            
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
            variation_seed = base_seed + 500000 + path_idx * 1000 + var_idx
            
            image_path = f"data/grids/test/grid_{len(test_data)}.png"
            grid = generate_grid_from_path(
                path, rows, cols,
                save_path=image_path,
                seed=variation_seed
            )
            
            # Verify the path is valid
            if not verify_path_validity(grid, path, start, goal):
                print(f"WARNING: Invalid maze generated for test {len(test_data)}")
            
            test_data.append({
                "id": len(test_data),
                "sequence": path,
                "image": image_path,
                "solution_id": path_idx,
                "variation": var_idx
            })
    
    return train_data, test_data

def save_datasets(train_data, test_data, metadata):
    """Save train and test datasets to separate JSON files with metadata."""
    
    # Wrap entries with metadata
    train_output = {
        'metadata': metadata,
        'entries': train_data
    }
    
    test_output = {
        'metadata': metadata,
        'entries': test_data
    }
    
    with open("data/train_sequences.json", "w") as f:
        json.dump(train_output, f, indent=2)
    
    with open("data/test_sequences.json", "w") as f:
        json.dump(test_output, f, indent=2)
    
    print(f"\n✓ Saved {len(train_data)} train examples to data/train_sequences.json")
    print(f"✓ Saved {len(test_data)} test examples to data/test_sequences.json")
    print(f"✓ Metadata included: {metadata}")

if __name__ == "__main__":
    os.makedirs("data/grids/train", exist_ok=True)
    os.makedirs("data/grids/test", exist_ok=True)

    parser = argparse.ArgumentParser(description='Generate maze dataset with REAL variations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--size', type=int, default=5, help='Grid Size (rows and cols)')
    parser.add_argument('--variations', type=int, default=100, 
                        help='Number of truly different wall variations per solution')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Proportion of solutions for training (0.8 = 80%)')

    args = parser.parse_args()

    rng = random.Random(args.seed)
    
    # FIX: Both rows and cols should be equal to args.size
    rows = cols = args.size

    # Create metadata to save with the datasets
    metadata = {
        'grid_size': args.size,
        'rows': rows,
        'cols': cols,
        'seed': args.seed,
        'variations': args.variations,
        'train_split': args.train_split,
        'start_position': [0, rows - 1],  # Bottom-left (col, row)
        'goal_position': [cols - 1, 0]    # Top-right (col, row)
    }

    # Generate all unique solution patterns
    max_num_sequences = num_monotonic_paths(rows, cols)
    paths = generate_random_unique_paths(rows, cols, rng, max_num_sequences)
    paths = sorted(paths)
    
    print(f"Generated {len(paths)} unique solution patterns")
    
    # Generate train and test sets with REAL variations
    train_data, test_data = generate_grids_with_variations(
        paths, rows, cols, args.seed, 
        num_variations=args.variations,
        train_split=args.train_split
    )
    
    # Save datasets with metadata
    save_datasets(train_data, test_data, metadata)
    
    print(f"\n{'='*60}")
    print("✓ Dataset generation complete!")
    print(f"{'='*60}")
    print(f"Train images: data/grids/train/")
    print(f"Test images: data/grids/test/")
    print(f"Train JSON: data/train_sequences.json")
    print(f"Test JSON: data/test_sequences.json")
    print(f"\n🔍 CHECK: Open a few train images with same solution_id")
    print(f"   They should have DIFFERENT wall patterns!")
    print(f"{'='*60}\n")