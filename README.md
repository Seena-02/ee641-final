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