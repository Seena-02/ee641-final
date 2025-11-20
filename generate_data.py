import random
import math
import os
import json
import argparse
from PIL import Image, ImageDraw, ImageFont

def generate_grid_from_path(path, rows, cols, save_path, obstacle_prob=0.6, cell_size=80, seed=None):
    move_map = {"U": (-1,0), "D": (1,0), "L": (0,-1), "R": (0,1)}
    rng_local = random.Random(seed if seed is not None else args.seed)

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

            # Draw label for S,G,X
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
    """
    Generate unique random monotonic (U/R only) paths with reproducibility.
    """
    
    up_moves = rows - 1
    right_moves = cols - 1
    base_moves = ["U"] * up_moves + ["R"] * right_moves

    paths = set()
    
    while len(paths) < num_sequences:
        m = base_moves[:]          # copy
        rng.shuffle(m)             # use the local RNG
        paths.add(tuple(m))        # store immutable, unique

    return [list(p) for p in paths]

def num_monotonic_paths(rows, cols):
    return math.comb(rows + cols - 2, rows - 1)

def generate_grids(paths):
    # for i in range(len(paths)):
    #     generate_grid_from_path(paths[i], rows, cols, save_path=f"data/grids/grid_{i}.png")
    for i, path in enumerate(paths):
        generate_grid_from_path(
            path, args.rows, args.cols, save_path=f"data/grids/grid_{i}.png",
            seed=args.seed + i   # unique but deterministic per grid
    )

def map_sequences_to_images(sequences, image_prefix="grid", output_file="data/grid_sequences.json"):
    """
    sequences: list of paths, e.g. [["U","R","U"...], ...]
    image_prefix: name of image files like path_0.png
    """

    grid_sequence = []

    for i, seq in enumerate(sequences):
        grid_sequence.append({
            "id": i,
            "sequence": seq,
            "image": f"grids/{image_prefix}_{i}.png"
        })

    with open(output_file, "w") as f:
        json.dump(grid_sequence, f, indent=4)

    return grid_sequence

def print_sequences(res):
        for seq in res:
            print(seq, "\n")

        unique_count = len({tuple(p) for p in res})
        total = len(res)
        print("total:", total, "unique:", unique_count)

if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), "data", "grids")
    os.makedirs(data_path, exist_ok=True)

    parser = argparse.ArgumentParser(description='Generate addition dataset')
    parser.add_argument('--seed', type=int, default=641,
                        help='Random seed')
    parser.add_argument('--cols', type=int, default=5, help='Number of cols')
    parser.add_argument('--rows', type=int, default=5, help='Number of rows')
 
    args = parser.parse_args()

    rng = random.Random(args.seed)

    max_num_sequences = num_monotonic_paths(args.rows, args.cols)
    paths = generate_random_unique_paths(args.rows, args.cols, rng, max_num_sequences)
    paths = sorted(paths)  # sort lexicographically for deterministic order
    grid_sequence = map_sequences_to_images(paths)
    generate_grids(paths)

    print("Data has been successfully generated at", data_path)


