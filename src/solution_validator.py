"""
Solution Validator for Maze Solver
Checks if LLM-generated solutions are valid, not just exact matches
"""

import torch


def validate_solution(maze_grid, sequence, start_pos=None, goal_pos=None):
    """
    Check if a sequence of moves reaches the goal from start position.
    
    Args:
        maze_grid: 2D array where 0=path, 1=wall
        sequence: List of tokens like ['R', 'R', 'U', 'U']
        start_pos: (col, row) starting position (default: bottom-left = (0, n-1))
        goal_pos: (col, row) goal position (default: top-right = (n-1, 0))
    
    Returns:
        dict: {
            'is_valid': bool,
            'reached_goal': bool,
            'final_position': (col, row),
            'num_moves': int,
            'hit_wall': bool,
            'out_of_bounds': bool
        }
    """
    grid_height, grid_width = maze_grid.shape
    
    # Calculate default positions based on grid size if not provided
    if start_pos is None:
        start_pos = (0, grid_height - 1)  # Bottom-left: (0, n-1)
    if goal_pos is None:
        goal_pos = (grid_width - 1, 0)    # Top-right: (n-1, 0)
    
    current_pos = list(start_pos)  # [col, row]
    
    move_map = {
        'R': (1, 0),   # Right: +1 col
        'U': (0, -1),  # Up: -1 row
    }
    
    result = {
        'is_valid': True,
        'reached_goal': False,
        'final_position': tuple(current_pos),
        'num_moves': 0,
        'hit_wall': False,
        'out_of_bounds': False,
        'invalid_token': False,
    }
    
    for i, token in enumerate(sequence):
        # Skip special tokens
        if token in ['<pad>', '<s>', '</s>', '<unk>']:
            continue
        
        # Check if token is valid
        if token not in move_map:
            result['is_valid'] = False
            result['invalid_token'] = True
            break
        
        # Calculate new position
        delta_col, delta_row = move_map[token]
        new_col = current_pos[0] + delta_col
        new_row = current_pos[1] + delta_row
        
        # Check bounds
        if new_col < 0 or new_col >= grid_width or new_row < 0 or new_row >= grid_height:
            result['is_valid'] = False
            result['out_of_bounds'] = True
            break
        
        # Check if it's a wall (assuming 1=wall, 0=path)
        if maze_grid[new_row, new_col] == 1:
            result['is_valid'] = False
            result['hit_wall'] = True
            break
        
        # Move is valid, update position
        current_pos = [new_col, new_row]
        result['num_moves'] += 1
        
        # Check if reached goal
        if tuple(current_pos) == goal_pos:
            result['reached_goal'] = True
            break
    
    result['final_position'] = tuple(current_pos)
    
    return result


def evaluate_with_validation(model, data_loader, device, tokenizer, maze_grids=None):
    """
    Enhanced evaluation that tracks both exact matches and valid solutions.
    
    Args:
        model: The trained model
        data_loader: DataLoader with test/train data
        device: torch device
        tokenizer: Tokenizer with decode functionality
        maze_grids: Dict mapping maze_id to grid array (required for validation)
    
    Returns:
        dict with detailed metrics
    """
    model.eval()
    
    results = {
        'total': 0,
        'exact_match': 0,
        'valid_solution': 0,
        'invalid_solution': 0,
        'details': []
    }
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            maze_ids = batch.get('maze_ids', [None] * len(images))
            
            # Generate predictions
            outputs = model.generate(
                images,
                max_length=20,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
            )
            
            for i in range(len(images)):
                # Decode sequences
                expected_tokens = tokenizer.decode(labels[i].tolist())
                predicted_tokens = tokenizer.decode(outputs[i].tolist())
                
                # Remove special tokens for comparison
                expected_clean = [t for t in expected_tokens if t not in ['<pad>', '<s>', '</s>', '<unk>']]
                predicted_clean = [t for t in predicted_tokens if t not in ['<pad>', '<s>', '</s>', '<unk>']]
                
                # Check exact match
                is_exact_match = expected_clean == predicted_clean
                
                # Validate solution (if maze grid available)
                is_valid = False
                validation_result = None
                
                if maze_grids is not None and maze_ids[i] is not None:
                    maze_grid = maze_grids[maze_ids[i]]
                    validation_result = validate_solution(maze_grid, predicted_clean)
                    is_valid = validation_result['reached_goal']
                
                # Update counters
                results['total'] += 1
                if is_exact_match:
                    results['exact_match'] += 1
                if is_valid:
                    results['valid_solution'] += 1
                else:
                    results['invalid_solution'] += 1
                
                # Store detailed result
                results['details'].append({
                    'maze_id': maze_ids[i],
                    'expected': expected_clean,
                    'predicted': predicted_clean,
                    'exact_match': is_exact_match,
                    'valid_solution': is_valid,
                    'validation': validation_result,
                })
    
    # Calculate percentages
    results['exact_match_pct'] = 100 * results['exact_match'] / results['total']
    results['valid_solution_pct'] = 100 * results['valid_solution'] / results['total']
    
    return results


def print_evaluation_results(results, dataset_name="Test"):
    """
    Pretty print evaluation results with both exact match and valid solution metrics.
    """
    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} SET RESULTS - DETAILED ANALYSIS")
    print("=" * 70)
    print(f"Total mazes evaluated: {results['total']}")
    print()
    print(f"Exact Match Accuracy:   {results['exact_match']}/{results['total']} "
          f"({results['exact_match_pct']:.1f}%)")
    print(f"Valid Solution Rate:    {results['valid_solution']}/{results['total']} "
          f"({results['valid_solution_pct']:.1f}%)")
    print(f"Invalid Solutions:      {results['invalid_solution']}/{results['total']} "
          f"({100 * results['invalid_solution'] / results['total']:.1f}%)")
    print()
    
    # Calculate the "creative solutions" - valid but not exact match
    creative_solutions = results['valid_solution'] - results['exact_match']
    creative_pct = 100 * creative_solutions / results['total']
    
    print(f"Creative Solutions:     {creative_solutions}/{results['total']} "
          f"({creative_pct:.1f}%)")
    print("  ↳ Valid paths that differ from training solution")
    print("=" * 70)
    
    # Show some examples
    if len(results['details']) > 0:
        print("\nSample Results:")
        print("-" * 70)
        
        # Show first 5 examples
        for i, detail in enumerate(results['details'][:5]):
            status = "✓ EXACT" if detail['exact_match'] else \
                     ("✓ VALID" if detail['valid_solution'] else "✗ INVALID")
            
            print(f"\nMaze {detail['maze_id']}: {status}")
            print(f"  Expected:  {' '.join(detail['expected'])}")
            print(f"  Predicted: {' '.join(detail['predicted'])}")
            
            if detail['validation']:
                val = detail['validation']
                if not val['is_valid']:
                    reason = "hit wall" if val['hit_wall'] else \
                            ("out of bounds" if val['out_of_bounds'] else \
                            ("invalid token" if val['invalid_token'] else "unknown"))
                    print(f"  Failure: {reason} at position {val['final_position']}")
        
        print("-" * 70)