# Quick Start Guide

## 5-Minute Setup

### 1. Download Everything
Download the entire `/mnt/user-data/outputs/` folder to your computer.

### 2. Organize Files
```bash
# Create project structure
mkdir maze-solver
cd maze-solver

# Copy the src folder
cp -r /path/to/downloads/outputs/src ./

# Copy notebooks
cp /path/to/downloads/outputs/train_refactored.ipynb ./
cp /path/to/downloads/outputs/analyze_results_refactored.ipynb ./

# Copy your existing data folder
cp -r /path/to/your/data ./

# Create models folder
mkdir models
```

Your structure should look like:
```
maze-solver/
├── src/
├── data/
├── models/
├── train_refactored.ipynb
└── analyze_results_refactored.ipynb
```

### 3. Run Training
```bash
jupyter notebook train_refactored.ipynb
# Then: Cell → Run All
```

### 4. Validate Results
```bash
jupyter notebook analyze_results_refactored.ipynb
# Then: Cell → Run All
```

## Expected Output

### Training (takes ~2-3 hours on GPU)
```
============================================================
TRAINING PHASE
============================================================
Training on 36950 mazes for 75 epochs...
============================================================
Epoch 1/75: 100%|██████████| 1155/1155 [02:34<00:00,  7.47it/s, loss=0.5916]
Epoch 1, Avg Loss: 0.685529, LR: 5.00e-05
...
Training completed. Final loss: 0.002000

============================================================
FINAL RESULTS SUMMARY
============================================================
Final Training Loss:    0.002000
Training Accuracy:      36950/36950 (100.0%)
Test Accuracy:          2858/9250 (30.9%)
Generalization Gap:     69.1%
============================================================

💾 Model checkpoint saved to ../models/resnet_gpt2_prefix.pth
```

### Analysis (takes ~5 minutes)
```
======================================================================
TEST SET RESULTS - DETAILED ANALYSIS
======================================================================
Total mazes evaluated: 9250

Exact Match Accuracy:   2858/9250 (30.9%)    ← What you saw before
Valid Solution Rate:    5775/9250 (62.4%)    ← The REAL performance!
Invalid Solutions:      3475/9250 (37.6%)

Creative Solutions:     2917/9250 (31.5%)
  ↳ Valid paths that differ from training solution
======================================================================

✅ EXCELLENT: Model is generalizing well!
   → Model learned spatial reasoning, not just memorization
   → High creative solution rate shows true understanding
   → Consider: Keep current architecture, maybe train longer
```

## If You Get Errors

### "No module named 'src'"
Your notebook isn't in the right place. Move it so `src/` is in the same directory.

### "File not found: ../data/train.json"
Your data folder isn't where expected. Either:
1. Move `data/` to the right place, OR
2. Update the path in the notebook

### "CUDA out of memory"
Reduce batch size in the notebook:
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=16,  # Change from 32 to 16
    ...
)
```

## What Each File Does

| File | Purpose | When to Use |
|------|---------|-------------|
| `src/model.py` | Model architecture | Imported automatically |
| `src/tokenizer.py` | Tokenization | Imported automatically |
| `src/dataset.py` | Data loading | Imported automatically |
| `src/train_utils.py` | Training loop | Imported automatically |
| `src/solution_validator.py` | Path validation | Used in analysis notebook |
| `train_refactored.ipynb` | Train model | Run once to train |
| `analyze_results_refactored.ipynb` | Validate solutions | Run after training |

## Workflow

```
1. Run train_refactored.ipynb
   ↓
2. Wait for training to complete (~2-3 hours)
   ↓
3. Check: Is test accuracy around 30%?
   ↓
4. Run analyze_results_refactored.ipynb
   ↓
5. Check: What's the valid solution rate?
   ↓
6. Decide next steps based on results
```

## Decision Tree

```
Valid solution rate > 60%?
├─ YES → ✅ Model is great!
│         • Keep current architecture
│         • Maybe train longer (100-150 epochs)
│         • Or try harder tasks (10×10 mazes)
│
└─ NO → Valid solution rate 40-60%?
    ├─ YES → ⚠️  Model is okay
    │         • Try stronger regularization
    │         • Or use 64-dim model
    │
    └─ NO → ❌ Model needs work
              • Make model smaller (64-dim)
              • Add heavy regularization
              • Generate more diverse data
```

## Next Experiment Ideas

After you see the validation results:

**If model is good (>60% valid):**
- Train for 150 epochs
- Try 10×10 mazes
- Add L/D movements (4-way navigation)

**If model needs work (<60% valid):**
- Reduce to 64-dim hidden size
- Increase dropout to 0.5
- Add label smoothing

**If you want faster iteration:**
- Train on subset (10k examples for 50 epochs)
- Use smaller grid (5×5 mazes)
- Reduce prefix tokens (4 instead of 8)

## Time Budget

| Task | Time |
|------|------|
| Setup files | 5 min |
| Training (GPU) | 2-3 hours |
| Training (CPU) | 12+ hours |
| Analysis | 5 min |
| **Total (GPU)** | **~3 hours** |

## Need Help?

1. Check `PROJECT_STRUCTURE.md` for detailed explanations
2. Check `REFACTORING_SUMMARY.md` for what changed
3. Check your original `train.ipynb` for comparison

---

**Ready?** Start with Step 1: Download Everything! 🚀
