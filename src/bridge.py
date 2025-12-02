import torch.nn as nn

# Since ViT embeddings are of size 768 and LM expects 1024, we need a projection layer
class Bridge(nn.Module):
    def __init__(self, vit_dim=768, lm_dim=1024):
        super().__init__()
        self.proj = nn.Linear(vit_dim, lm_dim)

    def forward(self, x):
        return self.proj(x)
    