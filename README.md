# High-Dimensional Visual Embedding for Language-Aligned Perception

## About

This project explores whether high-dimensional visual embeddings from a Vision Transformer (ViT) can preserve procedural image structure well enough for a Large Language Model (LLM) to reconstruct the sequence of steps that generated an image. Instead of predicting labels or captions, the LLM outputs program-like tokens describing how the image was built. By varying embedding dimensionality and projection mechanisms, the project investigates how representation size affects multimodal reasoning, interpretability, and the model’s ability to generate both correct and novel reconstruction sequences.

## Quickstart

```bash
python3 generate_data.py
```


Sources:\
Pytordch Vision transformer -> Vec2ext\
https://ankur3107.github.io/blogs/the-illustrated-image-captioning-using-transformers/ \
https://github.com/lucidrains/MaMMUT-pytorch \
https://github.com/google-deepmind/mammut/tree/main

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

And update your `requirements.txt` or `pip-requirements` file to remove the torch packages since they need special installation:
```
# Install PyTorch separately with GPU support - see README
# torch
# torchvision  
# torchaudio
numpy
scipy
opencv-python
matplotlib
h5py
jupyter
tensorboard
seaborn
tqdm
pandas
torchinfo
torchviz
Pillow
tokenizers
transformers
accelerate>=0.26.0