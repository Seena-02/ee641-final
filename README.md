# High-Dimensional Visual Embedding for Language-Aligned Perception

## About

This project explores whether high-dimensional visual embeddings from a Vision Transformer (ViT) can preserve procedural image structure well enough for a Large Language Model (LLM) to reconstruct the sequence of steps that generated an image. Instead of predicting labels or captions, the LLM outputs program-like tokens describing how the image was built. By varying embedding dimensionality and projection mechanisms, the project investigates how representation size affects multimodal reasoning, interpretability, and the model’s ability to generate both correct and novel reconstruction sequences.

## Quickstart

```bash
python3 generate_data.py
```


<!-- Sources:\
Pytordch Vision transformer -> Vec2ext\
https://ankur3107.github.io/blogs/the-illustrated-image-captioning-using-transformers/ \
https://github.com/lucidrains/MaMMUT-pytorch \
https://github.com/google-deepmind/mammut/tree/main -->

# Installation

## GPU Support (Recommended)

For CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Check your CUDA version with `nvidia-smi` to determine which version to install.

## Other Dependencies

After installing PyTorch with GPU support, install the remaining dependencies:
```bash
pip install numpy scipy opencv-python matplotlib h5py jupyter tensorboard seaborn tqdm pandas torchinfo torchviz Pillow tokenizers transformers accelerate>=0.26.0
```

## Verify GPU Installation
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## CPU-Only Installation (Not Recommended)

If you don't have a GPU, you can install the CPU version:
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt  # (excluding torch packages)
```

Note: Training will be significantly slower on CPU.
```

# Maze Solver: Vision-to-Sequence Model

## Project Goal
Build a model that takes maze grid images as input and outputs the solution sequence (e.g., "R R R U U U") by:
1. Encoding images into high-dimensional representations using a vision encoder
2. Decoding these representations into action sequences using an LLM-style tokenizer/decoder

## Dataset
- **Grid size**: 7x7 mazes
- **Total examples**: 924
- **Actions**: R (Right) and U (Up) only
- **Start**: Bottom-left corner
- **Goal**: Top-right corner
- **Sequence length**: ~12 tokens per maze

Example:
```json
{
  "id": 0,
  "sequence": ["R", "R", "R", "R", "R", "R", "U", "U", "U", "U", "U", "U"],
  "image": "data/grids/grid_0.png"
}
```

## Experiments & Results

### Attempt 1: ViT + GPT2 (Frozen Encoder)
**Setup**: 
- Encoder: ViT-Base (google/vit-base-patch16-224) - FROZEN
- Decoder: GPT2 with custom 6-token vocabulary
- Training: 200 epochs, 50 examples, LR=5e-4

**Results**:
- Final Loss: 0.439
- Accuracy: 0/10 (0%)
- **Issue**: Predictions were close in length and structure but not exact matches

### Attempt 2: ViT + GPT2 (Unfrozen Encoder)
**Setup**:
- Encoder: ViT-Base - UNFROZEN (fine-tuned)
- Decoder: GPT2 with custom vocabulary
- Training: 300 epochs, 100 examples, LR=1e-5

**Results**:
- Final Loss: 0.344
- Accuracy: 0/15 (0%)
- **Issue**: Model still not converging after 300 epochs

**Sample Predictions**:
```
Maze 0:
  Expected:  'R R R R R R U U U U U U'
  Predicted: 'R R R R R U U U U U'  ✗

Maze 7:
  Expected:  'R R R R U R R U U U U U'
  Predicted: 'R R R U R U R U U U'  ✗
```

### Key Findings

**Why ViT Failed**:
1. **Domain Mismatch**: ViT was pretrained on natural images (ImageNet: cats, dogs, cars), not grid-based maze structures
2. **Feature Extraction**: The self-attention mechanism in ViT doesn't naturally capture the spatial path structure critical for maze solving
3. **Loss Plateau**: After 300 epochs with 100 examples, loss remained at 0.34 (target: <0.05 for convergence)
4. **Training Instability**: Even with very low learning rates (1e-5) and proper configuration, the model couldn't learn the task

**What Worked**:
- ✅ Data preprocessing and tokenization were correct
- ✅ Model architecture setup was proper
- ✅ Training loop was stable
- ✅ Predictions had correct format (only R and U tokens, reasonable lengths)

**What Didn't Work**:
- ❌ ViT couldn't extract discriminative features for different maze paths
- ❌ Model couldn't distinguish between similar mazes
- ❌ Loss never converged below 0.34 even after extensive training

## Decision: Pivot to ResNet

### Why ResNet18 Instead of ViT?

**ResNet18 is better suited for this task because**:
1. **Convolutional architecture**: Better at capturing local spatial patterns (maze walls and paths)
2. **Hierarchical features**: Progressively builds from edges → walls → paths → full maze structure
3. **Proven for grid-based tasks**: CNNs excel at structured, grid-like data
4. **Lighter weight**: Faster training and convergence

### Project Goal Still Intact ✅

The core concept remains unchanged:
- **Image Encoding**: ResNet18 encodes maze images into high-dimensional features (512-dim)
- **Sequence Decoding**: GPT2 decodes these features into action sequences using a custom tokenizer
- **Architecture**: Vision Encoder → Language Decoder (same paradigm)

**The only change**: Swapping ViT for ResNet18 as the vision encoder

## Next Steps

1. Implement ResNet18 + GPT2 architecture
2. Train on 100 examples first to verify convergence
3. Scale to full 924-example dataset
4. Target: >90% accuracy with loss <0.05

## Technical Details

**Tokenizer Vocabulary**:
```
{'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3, 'R': 4, 'U': 5}
```

**Model Configuration**:
- Image preprocessing: Resize to 224x224, normalize
- Sequence max length: 20 tokens
- Special tokens: BOS (<s>), EOS (</s>), PAD (<pad>)
- Loss function: CrossEntropyLoss with padding ignored (-100)

## Lessons Learned

1. **Pretrained models aren't always better**: ViT's pretraining on natural images was actually a disadvantage
2. **Task-specific architecture matters**: Grid-based data needs convolutional inductive biases
3. **Loss is the truth**: No amount of hyperparameter tuning helped when the architecture was fundamentally mismatched
4. **Start simple, then scale**: Testing on small datasets (50-100 examples) saved significant time

---

**Status**: Pivoting to ResNet18 + GPT2 architecture  
**Expected Outcome**: Successful convergence and >90% accuracy on maze solving task


# Maze Solver: Vision→LLM Model - Complete Testing History

## Project Goal
Build a model that uses a **Vision Encoder (ResNet18)** to encode maze images into high-dimensional features and an **LLM Decoder (GPT2)** to generate the solution sequence.

---

## Testing History

| Test # | Grid Size | Unique Solutions | Model Architecture | Variations per Solution | Total Train Images | Total Test Images | Epochs | Train Accuracy | Test Accuracy | Generalization Gap | Final Loss | Status |
|--------|-----------|------------------|-------------------|------------------------|-------------------|------------------|--------|---------------|---------------|-------------------|------------|---------|
| 1 | 5×5 | 70 | 768-dim, 6 layers, 32 prefix | 100 (fake*) | 5,600 | 1,400 | 100 | ~100% | 1.9% | ~98% | 0.001 | ❌ Severe overfitting |
| 2 | 5×5 | 70 | 256-dim, 2 layers, 32 prefix | 100 (fake*) | 5,600 | 1,400 | 100 | ~100% | 14.8% | ~85% | 0.001 | ❌ Still overfitting |
| 3 | 5×5 | 70 | 128-dim, 2 layers, 16 prefix | 100 (real) | 5,600 | 1,400 | 100 | 53.0% | 9.6% | 43.4% | 0.214 | ⚠️ Better, but gap too large |
| 4 | 7×7 | ~924 | 128-dim, 2 layers, 16 prefix | 50 (real) | ~37,000 | ~9,200 | 75 | **TBD** | **TBD** | **TBD** | **TBD** | 🔄 **Next test** |

\* *"Fake variations" = Only decorative obstacles off the solution path, not structural maze differences*

---

## Key Discoveries

### 1. Data Quality Issue (Tests 1-2)
**Problem**: Initial maze generation only placed random obstacles off the solution path.
- All 100 "variations" of the same solution were essentially the same maze with different decorations
- Model could memorize the 70 solution patterns easily
- Result: Perfect training accuracy, terrible test accuracy

**Fix**: Modified maze generation to place strategic obstacles near the path, creating truly different maze structures for each variation.

### 2. Model Size vs Task Complexity (Tests 1-3)
**Observations**:
- 768-dim model (93M params): Memorizes everything instantly
- 256-dim model (15M params): Still memorizes despite smaller size
- 128-dim model (5M params): Can't memorize anymore (53% train acc), but...

**Conclusion**: With only 70 unique solution patterns, even tiny models learn pattern recognition instead of spatial reasoning.

### 3. The Fundamental Limitation
**70 unique solutions for 5×5 mazes is too few:**
- Even with 100 real variations each (5,600 images), the model learns to recognize solution patterns
- Test set has only 14 new solution patterns → model fails on unseen patterns
- Gap of 43% shows model hasn't learned general maze-solving logic

---

## Progress Metrics

### Overfitting Reduction Progress:
| Test | Train Acc | Test Acc | Gap | Improvement |
|------|-----------|----------|-----|-------------|
| 1 | ~100% | 1.9% | ~98% | Baseline |
| 2 | ~100% | 14.8% | ~85% | Test +12.9% |
| 3 | 53.0% | 9.6% | 43.4% | Gap -54.6% |

### Model Complexity Reduction:
| Test | Hidden Size | Layers | Prefix Tokens | Total Params | Dropout |
|------|-------------|--------|---------------|--------------|---------|
| 1 | 768 | 6 | 32 | ~93M | 0.1 |
| 2 | 256 | 2 | 32 | ~15M | 0.3 |
| 3 | 128 | 2 | 16 | ~5M | 0.4 |

---

## Next Test: 7×7 Mazes

### Rationale:
- **13× more unique solutions** (924 vs 70)
- **Longer sequences** (~12 tokens vs ~8 tokens)
- **More complex spatial reasoning** required
- **Harder to memorize** patterns

### Expected Results:
With proper scaling:
- **Training Accuracy**: 40-60% (healthy learning range)
- **Test Accuracy**: 25-45% (would be SUCCESS!)
- **Generalization Gap**: <20% (good generalization)
- **Final Loss**: 0.10-0.20 (not overfitting)

### Success Criteria:
✅ Test accuracy > 30%
✅ Generalization gap < 25%
✅ Model learns spatial reasoning, not pattern matching

---

## Architecture Details

### Current Best Model (Test 3 & 4):
```
Vision Encoder: ResNet18 (pretrained on ImageNet)
  └─ Feature extraction: 512-dim
  └─ Projection to hidden size: 512 → 128

Prefix Generator:
  └─ MLP: 128 → 256 → 2048 (16 prefix tokens × 128-dim)
  └─ Generates soft prompts from image features

LLM Decoder: GPT2 (custom configuration)
  └─ Embedding size: 128-dim
  └─ Layers: 2
  └─ Attention heads: 2
  └─ Dropout: 0.4 (high regularization)
  └─ Vocabulary: 6 tokens (pad, bos, eos, unk, R, U)

Total Parameters: ~5M (trainable)
```

### Training Configuration:
- Optimizer: AdamW with differentiated learning rates
  - ResNet: 5e-5 (1/10 of base)
  - Prefix Generator: 5e-4 (base)
  - GPT2: 1e-4 (1/5 of base)
- Scheduler: Cosine Annealing (to 1e-6)
- Batch Size: 32
- Gradient Clipping: 1.0
- Weight Decay: 0.01

---

## Lessons Learned

1. **Data quality > Data quantity**: 5,600 fake variations were useless, but proper variations matter
2. **Model size must match task complexity**: Too large = memorization, too small = underfitting
3. **Task complexity matters**: 70 unique patterns are too few for a deep learning model to learn generalization
4. **Overfitting indicators**: 
   - Loss < 0.01 = likely memorizing
   - Train accuracy ~100% = definitely memorizing
   - Large train/test gap = not generalizing
5. **Prefix-tuning works**: Successfully connects vision encoder to LLM decoder
6. **ResNet18 extracts features well**: Even with frozen weights initially, features are usable

---

## Current Status
✅ Architecture: Vision→LLM working correctly
✅ Data generation: Fixed to create real maze variations
✅ Model size: Optimized to prevent memorization
🔄 Next: Scale to 7×7 mazes with more unique solutions

**Goal**: Achieve >30% test accuracy with <25% generalization gap on 7×7 mazes.