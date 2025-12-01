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
