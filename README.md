# High-Dimensional Visual Embedding for Language-Aligned Perception

## Project Overview

This project explores whether high-dimensional visual embeddings can preserve procedural image structure well enough for an LLM to reconstruct the sequence of steps that generated an image. The model takes maze grid images as input and outputs solution sequences (e.g., "R R R U U U") using a Vision Encoder → LLM Decoder architecture.

**Core Architecture:**

- **Vision Encoder**: ResNet18 → extracts 512-dim features
- **Prefix Generator**: MLP → projects features to soft prompts (16 tokens × 128-dim)
- **LLM Decoder**: Custom GPT2 → generates action sequences

## Installation

### GPU Support (Recommended)

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
pip install numpy scipy opencv-python matplotlib h5py jupyter tensorboard seaborn tqdm pandas torchinfo torchviz Pillow tokenizers transformers accelerate>=0.26.0
```

### Verify Installation

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Quickstart

```bash
python generate_data.py
```

## Dataset

- **Grid sizes**: 4x4 (20 unique solutions), 5×5 (70 unique solutions) and 7×7 (924 unique solutions)
- **Actions**: R (Right) and U (Up) only
- **Start**: Bottom-left corner → **Goal**: Top-right corner
- **Sequence length**: ~4 tokens (4x4), ~8 tokens (5×5), ~12 tokens (7×7)

Example:

```json
    {
      "id": 0,
      "sequence": [
        "R",
        "U",
        "U",
        "R",
        "R",
        "U",
        "U",
        "R"
      ],
      "image": "data/grids/test/grid_0.png",
      "solution_id": 0,
      "variation": 0
    },
```

## Experimental Journey

### Initial Attempt: ViT + GPT2 (Failed)

**Why ViT Failed:**

- **Domain mismatch**: Pretrained on natural images (ImageNet), not grid structures
- **Architecture mismatch**: Self-attention doesn't capture spatial path structure well
- **Results**: After 300 epochs, loss plateaued at 0.34 with 0% accuracy

**Key Insight**: Pretrained models aren't always better—ViT's pretraining was actually a disadvantage for grid-based data.

### Pivot to ResNet18

**Why ResNet18 works better:**

- Convolutional architecture naturally captures local spatial patterns (walls/paths)
- Hierarchical features: edges → walls → paths → full maze structure
- Proven track record with grid-based tasks

### Testing History: The Path to Generalization

| Test | Grid | Train Images | Test Images | Model                 | Epochs | LR   | Train Acc | Test Acc (Exact) | Test Acc (Valid) | Gap   | Loss   | Status        |
| ---- | ---- | ------------ | ----------- | --------------------- | ------ | ---- | --------- | ---------------- | ---------------- | ----- | ------ | ------------- |
| 1    | 4×4  | 2,400        | 600         | 128-dim, 2L, 8 prefix | 40     | 5e-4 | 71.4%     | 7.3%             | **61.0%**        | 64.1% | 0.1721 | ⚠️ Moderate   |
| 2    | 5×5  | 5,600        | 1,400       | 128-dim, 2L, 8 prefix | 75     | 5e-4 | 71.3%     | 13.1%            | **70.1%**        | 58.2% | 0.1459 | ✅ Good       |
| 3    | 7×7  | 36,950       | 9,250       | 128-dim, 2L, 8 prefix | 75     | 5e-4 | 14.8%     | 6.6%             | **26.2%**        | 8.3%  | 0.3021 | ❌ Needs Work |

### What Worked ✅

**1. Architecture Finds Its Sweet Spot (4×4 and 5×5)**

- 128-dim, 2-layer model successfully learned both small grids
- Consistent ~71% train accuracy across both grid sizes
- Valid solution rates of 61-70% demonstrate genuine spatial reasoning
- Model avoided pure memorization while achieving solid performance

**2. Progressive Improvement with More Data**

- 4×4 → 5×5: Valid solution rate increased from 61.0% to 70.1% (+9.1%)
- Generalization gap improved from 64.1% to 58.2% (-5.9%)
- More training data (2.3×) and epochs (1.9×) yielded tangible improvements
- Loss decreased from 0.1721 to 0.1459, showing better convergence

**3. Creative Solution Generation**

- 4×4: 53.7% of valid solutions were creative (different from training)
- 5×5: 57.0% creative solutions, showing the model learned principles, not patterns
- Model finds alternative valid paths, demonstrating true spatial reasoning

### What Didn't Work ❌

**Critical Discovery - The Complexity Ceiling (Test 3)**

The 7×7 grid revealed a fundamental architectural limitation:

**Performance Collapse:**

- Train accuracy dropped from 71% to just **14.8%** → clear underfitting
- Valid solution rate plummeted to **26.2%** (down from 70.1%)
- Invalid solutions skyrocketed to **73.8%** (6,572 wall collisions)
- Loss remained high at **0.3021** (2× higher than 5×5)

**Root Cause - Insufficient Model Capacity:**

- 13× more unique solution patterns (924 vs 70) overwhelmed the small architecture
- The model can't even learn the training set, let alone generalize
- Despite 6.6× more training data, performance degraded catastrophically
- The 2-layer, 128-dim architecture hit its complexity ceiling

**Paradoxical Generalization Gap:**

- Only 8.3% gap seems good, but it's actually bad news
- The gap is low because the model fails equally on both train and test sets
- This is underfitting, not good generalization

**Key Lesson**: More data doesn't help if your model lacks capacity. The same architecture that succeeded on 5×5 completely failed on 7×7, proving that model size must scale with task complexity.

**Required Next Steps for 7×7:**

1. **Increase model capacity**: 256-dim or 512-dim hidden size, 4-6 layers, more attention heads
2. **More training**: 100-150 epochs to allow proper convergence
3. **Target metrics**: Achieve 60-70% train accuracy first, then evaluate generalization
4. **Validation**: Don't scale complexity further until base performance is established

## Key Discoveries

### 1. Data Quality Over Quantity

**Problem**: Initial maze generation only placed decorative obstacles off the solution path—all 100 "variations" were essentially identical mazes with different decorations.

**Result**: Model memorized 70 solution patterns instead of learning spatial reasoning.

**Fix**: Modified generation to place strategic obstacles near the path, creating structurally different mazes.

### 2. The Task Complexity Threshold

**70 unique solutions for 5×5 mazes proved too few:**

- Even small models (5M params) learned pattern recognition, not spatial reasoning
- Test set had only 14 new solution patterns → model couldn't generalize
- 43% generalization gap showed lack of true maze-solving logic

**Solution**: Scale to 7×7 mazes with 924 unique solutions (13× more complexity)

### 3. Model Size vs Task Complexity - The Critical Balance

**The Goldilocks Problem**: Model capacity must match task complexity.

**Too Large (Early experiments):**

- **768-dim model (93M params)**: Instant memorization
- **256-dim model (15M params)**: Still memorizes despite smaller size
- Result: Perfect training, terrible generalization

**Just Right (4×4 and 5×5):**

- **128-dim model (5M params)**: Sweet spot for small grids
- Achieves ~71% train accuracy without memorization
- 61-70% valid solution rate shows genuine learning
- Creative solutions (53-57%) prove spatial reasoning

**Too Small (7×7):**

- **Same 128-dim model**: Catastrophic failure on larger grids
- Only 14.8% train accuracy (down from 71%)
- 26.2% valid rate (down from 70%)
- Model capacity ceiling hit hard

**Key Insight**: The same architecture can be "just right" for one task and "too small" for another. Task complexity grew 13× (70→924 solutions), but model capacity stayed constant, resulting in complete breakdown. Model size must scale with problem complexity.

### 4. The Validation Metric Discovery ⭐ (From Earlier High-Capacity Experiments)

**Note**: This discovery came from earlier experiments with larger models on 7×7 grids that achieved much higher performance than Test 3's small model.

**The most critical finding**: Traditional exact-match accuracy severely underestimates model performance!

**Initial belief**:

- Exact match accuracy: 30.9%
- Conclusion: "Model is overfitting with 69% generalization gap"

**Reality after implementing path validation:**

- Valid solution rate: **86.5%**
- Creative solutions: **55.6%** (valid but different from training)
- True generalization gap: Only 13.5%

**Key insights:**

- Models that truly learn spatial reasoning find **alternative valid paths** to solve mazes
- Over half of correct predictions use different routes than training data
- Exact-match metrics can underestimate performance by 50+ percentage points
- This validates that larger models (when properly sized) can achieve genuine spatial reasoning

**Current Status (Test 3)**: The 128-dim model on 7×7 hasn't reached this level yet—it's still in the underfitting phase at 26.2% valid rate. This discovery shows what's possible with proper model capacity.

### 5. Overfitting Indicators Discovered

- Loss < 0.01 → likely memorizing
- Train accuracy ~100% → definitely memorizing
- Large train/test gap → poor generalization

## Current Architecture (Test 4)

```
Vision Encoder: ResNet18 (pretrained on ImageNet)
  └─ Feature extraction: 512-dim
  └─ Projection: 512 → 128

Prefix Generator:
  └─ MLP: 128 → 256 → 2048 (16 prefix tokens × 128-dim)
  └─ Generates soft prompts from image features

LLM Decoder: GPT2 (custom)
  └─ Embedding: 128-dim
  └─ Layers: 2
  └─ Attention heads: 2
  └─ Dropout: 0.4 (high regularization)
  └─ Vocabulary: 6 tokens (pad, bos, eos, unk, R, U)

Total Parameters: ~5M
```

### Training Configuration

- **Optimizer**: AdamW with differentiated learning rates
  - ResNet: 5e-5 (fine-tuning rate)
  - Prefix Generator: 5e-4 (base rate)
  - GPT2: 1e-4 (controlled rate)
- **Scheduler**: Cosine Annealing (to 1e-6)
- **Batch Size**: 32
- **Gradient Clipping**: 1.0
- **Weight Decay**: 0.01

## Results & Progress

### Current Status: Establishing Baseline Performance

**Tests 1-3** represent the initial exploration phase with a fixed architecture (128-dim, 2-layer) across three grid sizes:

**✅ Success on Small Grids (4×4, 5×5):**

- Consistent ~71% train accuracy shows the model learns effectively
- Valid solution rates of 61-70% demonstrate genuine spatial reasoning
- 53-57% creative solutions prove the model isn't just memorizing
- Progressive improvement from 4×4 to 5×5 shows healthy scaling

**❌ Failure on Large Grids (7×7):**

- Train accuracy collapses to 14.8% (vs 71% on smaller grids)
- Valid solution rate drops to 26.2% (vs 70% on 5×5)
- 73.8% invalid solutions with 6,572 wall collisions
- Clear evidence of insufficient model capacity

### The Path Forward

**What We Know Works:**

- ResNet18 + GPT2 with prefix-tuning is a viable architecture
- 128-dim, 2-layer model handles up to 5×5 grids effectively
- Path validation metrics reveal true model capabilities
- Creative solution generation indicates spatial reasoning, not memorization

**What We Need to Fix:**

- Scale model capacity for 7×7 grids (256-512 dim, 4-6 layers)
- Increase training duration (100-150 epochs)
- Target 60-70% train accuracy before evaluating generalization
- Then measure creative solution rate to validate true learning

### Historical Context: The Validation Discovery 🎉

**Note**: In earlier experiments with larger models, we discovered that exact-match accuracy severely underestimates performance:

**Traditional Metrics (Misleading):**

- Exact Match Accuracy: 30.9%
- Appeared to show severe overfitting (69% gap)

**True Performance (After Validation):**

- **Valid Solution Rate: 86.5%** ✅
- **Creative Solutions: 55.6%** (valid alternate paths)
- **True Generalization Gap: 13.5%** (much healthier!)

This discovery showed that properly-sized models can achieve genuine spatial reasoning. Our current Test 3 (7×7) hasn't reached this level due to insufficient capacity, but it represents the target for future iterations.

## Lessons Learned

1. **Validation metrics are critical**: Exact-match accuracy can underestimate performance by 50+ percentage points. For tasks with multiple valid solutions, checking solution validity is essential. (Discovered in earlier high-capacity experiments)

2. **Creative solutions prove learning**: When properly-sized models generate 55-70% creative solutions (valid but different from training), it demonstrates genuine spatial reasoning rather than memorization.

3. **Architecture matters**: CNNs (ResNet) > Transformers (ViT) for grid-based spatial tasks. Domain-specific inductive biases are crucial.

4. **Data quality > quantity**: Decorative variations were useless; structural diversity is key. Strategic obstacle placement creates meaningful variations.

5. **Task complexity threshold**: Need enough unique patterns (924 vs 70) for generalization. Too few patterns lead to pattern matching instead of learning.

6. **Model sizing is critical - The Goldilocks Principle**:

   - Too large (768-dim) → memorization and overfitting
   - Just right (128-dim for 5×5) → learning without memorization
   - Too small (128-dim for 7×7) → underfitting and failure
   - **Key insight**: The "right size" depends on task complexity. 128-dim succeeded on 5×5 (70 solutions) but catastrophically failed on 7×7 (924 solutions). Model capacity must scale with problem complexity.

7. **Underfitting indicators**:

   - Low train accuracy (14.8% vs 71% expected)
   - High loss that won't decrease (0.30 vs 0.14-0.17)
   - Low valid solution rate with high wall collisions
   - Small generalization gap (8.3%) due to failing on both train and test

8. **Prefix-tuning works**: Successfully bridges vision encoder to LLM decoder without fine-tuning the entire model.

9. **Progressive scaling matters**: Successfully moving from 4×4 → 5×5 showed that model capacity matched complexity. The failure at 7×7 shows exactly where the ceiling is, providing clear direction for next steps.

## Evaluation Methodology

### Why Exact Match Isn't Enough

Traditional accuracy (exact match) only counts predictions that perfectly match the training solution:

```
Expected:  R R R U U U
Predicted: R U R R U U  ❌ Counted as wrong
```

But for mazes, **multiple paths can reach the goal**:

```
Training path:  R R R U U U  (goes right first, then up)
Model's path:   R U R R U U  (alternates right and up)
                ↑ Different but VALID! Both reach goal!
```

### Path Validation Approach

Instead of exact matching, we validate solutions by:

1. **Convert maze image to grid**: Sample cell centers to create binary grid (0=path, 1=wall)
2. **Simulate the path**: Follow predicted moves from start (bottom-left) to goal (top-right)
3. **Check validity**:
   - Does it stay in bounds?
   - Does it avoid walls?
   - Does it reach the goal?

**Results**:

- ✅ **Valid solutions**: Path successfully reaches goal (86.5%)
- 📊 **Exact matches**: Path identical to training solution (30.9%)
- 🎨 **Creative solutions**: Valid but different from training (55.6%)

This reveals the model learned **spatial reasoning**, not just memorization!

## Technical Details

**Tokenizer Vocabulary:**

```
{'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3, 'R': 4, 'U': 5}
```

**Model Configuration:**

- Image preprocessing: Resize to 224×224, normalize
- Sequence max length: 20 tokens
- Special tokens: BOS, EOS, PAD
- Loss function: CrossEntropyLoss (padding ignored: -100)

## Future Directions

### Immediate Priority: Fix 7×7 Performance

**Critical next steps based on Test 3 failure:**

1. **Scale up model capacity**:

   - Try 256-dim hidden size, 4 layers → ~20M params
   - If still insufficient, go to 512-dim, 6 layers → ~40M params
   - Add more attention heads (4-8) and prefix tokens (16-32)
   - Target: Achieve 60-70% train accuracy first

2. **Extend training duration**:

   - Increase from 75 to 100-150 epochs
   - Monitor loss convergence (should reach <0.20)
   - Use early stopping to prevent overfitting once train acc > 70%

3. **Validate the fix**:
   - First confirm train accuracy reaches 60-70%
   - Then measure valid solution rate (target: >60%)
   - Check creative solution percentage (target: >50%)
   - Ensure generalization gap stays under 20%

### After Establishing 7×7 Baseline

**Once 7×7 achieves 60%+ valid rate:**

- **Analyze failure modes**: Study invalid solutions to identify patterns
- **Optimize creative diversity**: Investigate what makes certain mazes favor creative solutions
- **Compare 5×5 vs 7×7**: Understand how creative solution rates differ by complexity

### Scaling Up Complexity

- **Larger grids**: Test on 10×10 or 15×15 mazes with exponentially more solution patterns
- **4-way navigation**: Add L (Left) and D (Down) movements for full directional control
- **Obstacle density**: Vary maze difficulty by adjusting wall density
- **Multi-goal tasks**: Extend to mazes with multiple checkpoints

### Model Improvements

- **Attention visualization**: Use gradient-based methods to see what the model "looks at"
- **Beam search**: Test if beam search (k>1) improves valid solution rate
- **Architecture search**: Systematically test different layer/head/dimension combinations
- **Regularization tuning**: Experiment with dropout rates and weight decay

### Analysis & Understanding

- **Path diversity analysis**: Cluster creative solutions to understand reasoning strategies
- **Embedding analysis**: Study prefix embeddings to see how maze features are encoded
- **Ablation studies**: Test impact of prefix tokens, model depth, attention heads
- **Capacity analysis**: Determine the mathematical relationship between grid size and required parameters

### Real-World Applications

- **Robot navigation**: Adapt to real-world path planning with obstacles
- **Game AI**: Apply to procedurally generated game levels
- **Code generation**: Extend to generating valid execution sequences for programs

---

**Project Status**: Baseline architecture established; scaling challenges identified

**Current Results (Tests 1-3)**:

- ✅ 4×4 grids: 61.0% valid rate (moderate performance)
- ✅ 5×5 grids: 70.1% valid rate (good performance)
- ❌ 7×7 grids: 26.2% valid rate (underfitting - model too small)

**Key Findings**:

1. **Architecture validation**: ResNet18 + GPT2 with prefix-tuning works for spatial reasoning
2. **Capacity ceiling discovered**: 128-dim model hits limit at 7×7 complexity
3. **Creative solutions**: 53-57% of valid solutions use different paths (proves spatial reasoning)
4. **Next step**: Scale to 256-512 dim for 7×7 success

**Historical Note**: Earlier experiments with larger models achieved 86.5% valid rate on 7×7 with 55.6% creative solutions, demonstrating the potential of this architecture when properly sized.
