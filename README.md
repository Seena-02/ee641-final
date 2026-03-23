# High-Dimensional Visual Embedding for Language-Aligned Perception

## Overview

This project investigates whether high-dimensional visual embeddings can preserve procedural image structure well enough for an LLM to reconstruct the sequence of steps that generated an image. A Vision Encoder (ResNet18) extracts features from maze grid images, a Prefix Generator converts them to soft prompts, and a GPT2 decoder generates solution sequences (e.g., `R R U U R U`).

- **Actions**: R (Right) and U (Up) only
- **Start**: Bottom-left corner → **Goal**: Top-right corner
- **Grid sizes**: 4x4 (20 unique solutions), 5x5 (70 unique solutions), 7x7 (924 unique solutions)

Example data entry:

```json
{
  "id": 0,
  "sequence": ["R", "U", "U", "R", "R", "U", "U", "R"],
  "image": "data/5x5/grids/test/grid_0.png",
  "solution_id": 0,
  "variation": 0
}
```

## Architecture

```
Maze Image (224x224)
    │
    ▼
ResNet18 (pretrained, frozen) → 512-dim features
    │
    ▼
Projection Layer → hidden_size-dim
    │
    ▼
Prefix Generator (MLP: hidden → hidden*2 → num_prefix * hidden)
    │  Tanh activation
    ▼
Soft Prefix Embeddings (num_prefix_tokens × hidden_size)
    │
    ▼
GPT2 Decoder (custom) → autoregressive token generation
    │
    ▼
Solution Sequence: R U U R R U
```

**Vocabulary**: 6 tokens — `<pad>` (0), `<s>` (1), `</s>` (2), `<unk>` (3), `R` (4), `U` (5)

### Per-Grid Configuration

Model capacity scales with task complexity. The number of unique monotonic paths through an NxN grid is C(2n-2, n-1).

| Grid | Unique Solutions | Hidden Size | Layers | Heads | Prefix Tokens | Epochs |
|------|-----------------|-------------|--------|-------|---------------|--------|
| 4x4  | 20              | 128         | 2      | 2     | 16            | 40     |
| 5x5  | 70              | 128         | 2      | 2     | 16            | 75     |
| 7x7  | 924             | 256         | 4      | 4     | 16            | 100    |

### Training Configuration

- **Optimizer**: AdamW with differentiated learning rates
  - ResNet features: lr/10 (5e-5)
  - Prefix generator: lr (5e-4)
  - GPT2 decoder: lr/5 (1e-4)
- **Scheduler**: Cosine Annealing (min lr: 1e-6)
- **Batch size**: 32
- **Gradient clipping**: max_norm=1.0
- **Weight decay**: 0.01
- **Dropout**: 0.4

## Installation

### PyTorch with GPU Support

For CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Dependencies

```bash
pip install -r requirements.txt
```

### Verify

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Training Workflow

### 1. Generate Dataset

```bash
python generate_dataset.py --size 5 --variations 100 --seed 69
```

This creates `data/5x5/` with maze images and `train_sequences.json` / `test_sequences.json`. Mazes use strategic obstacle placement (70% block probability near the solution path) to create structurally diverse variations.

### 2. Train Model

Open `train.ipynb` (works locally or on Colab), set `grid_size`, and run all cells. The model checkpoint is saved to `models/NxN/resnet_gpt2_prefix.pth`.

### 3. Analyze Results

Open `analyze_results.ipynb` and run all cells. It automatically discovers all trained models in `models/`, generates test datasets, evaluates each model, and saves results to `results/NxN/`.

## Results

| Metric | 4x4 | 5x5 | 7x7 |
|--------|-----|-----|-----|
| Test Samples | 600 | 1,400 | TBD |
| Exact Match | 5.7% | 17.7% | TBD |
| Valid Solution Rate | 54.2% | 82.6% | TBD |
| Creative Solutions | 48.5% | 64.9% | TBD |
| Train Accuracy | 73.2% | 74.8% | TBD |
| Training Loss | 0.1377 | 0.1139 | TBD |
| Generalization Gap | 67.6% | 57.2% | TBD |

7x7 is pending training with a scaled-up architecture (256-dim, 4-layer, 4 heads).

### Understanding the Metrics

**Exact match** only counts predictions identical to the training solution. But mazes have multiple valid paths:

```
Training path:  R R R U U U  (goes right first, then up)
Model's path:   R U R R U U  (alternates right and up)
                ↑ Different but VALID — both reach the goal!
```

We validate solutions by simulating the predicted path on the maze grid:
- **Valid Solution**: path stays in bounds, avoids walls, and reaches the goal
- **Exact Match**: valid solution identical to the training data
- **Creative Solution**: valid solution that differs from training — the model found an alternative path

This is why valid solution rate is the true performance metric. Exact match underestimates performance by 40-60 percentage points.

### Failure Analysis

The dominant failure mode is wall collision:

| Failure Type | 4x4 | 5x5 |
|-------------|-----|-----|
| Hit Wall | 270 (98.2%) | 219 (90.1%) |
| Out of Bounds | 5 (1.8%) | 24 (9.9%) |

The model understands general maze navigation but occasionally misjudges wall positions.

## Key Discoveries

1. **Valid solution rate is the true metric** — exact match accuracy severely underestimates model performance. A model scoring 5.7% exact match actually solves 54.2% of mazes correctly.

2. **Creative solutions prove spatial reasoning** — 48-65% of valid solutions use different paths than training data, demonstrating the model learned navigation principles rather than memorizing sequences.

3. **CNNs beat ViT for grid tasks** — ViT pretrained on ImageNet achieved 0% accuracy after 300 epochs. ResNet18's convolutional architecture naturally captures local spatial patterns (walls, paths, corridors).

4. **Data quality over quantity** — initial maze generation placed obstacles decoratively (away from paths), making all variations structurally identical. Strategic obstacle placement near the solution path creates meaningful diversity.

5. **Model capacity must match task complexity** — the 128-dim, 2-layer architecture works well for 4x4/5x5 (20-70 solutions) but 7x7 (924 solutions) requires scaling to 256-dim, 4-layer.

## Results Directory

```
results/
├── comparison.png              # side-by-side performance chart
├── 4x4/
│   ├── summary.json            # model config + performance metrics
│   ├── report.txt              # formatted text report
│   ├── detailed_results.json   # per-maze exact match & creative solutions
│   ├── failure_analysis.json   # failure type breakdown
│   └── creative_solutions.png  # visualization of alternative valid paths
├── 5x5/
│   └── (same structure)
└── 7x7/
    └── (TBD — pending training)
```
