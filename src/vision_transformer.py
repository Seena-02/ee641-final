import torch
from torchvision import models, transforms
from PIL import Image
import os

# Load pre-trained ViT
weights = models.ViT_B_16_Weights.DEFAULT
vit_model = models.vit_b_16(weights=weights)
vit_model.eval()

# Use the built-in transforms for this model
preprocess = weights.transforms()

# Folder with grid images
data_path = "data/grids"
image_files = [f for f in os.listdir(data_path) if f.endswith(".png")]

# Optional: store encodings in a dictionary
encodings = {}

with torch.no_grad():  # no gradient needed
    for img_file in image_files:
        img_path = os.path.join(data_path, img_file)
        image = Image.open(img_path).convert("RGB")
        tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

        # Forward pass through ViT
        outputs = vit_model._process_input(tensor)  # prepare patches + CLS token
        for blk in vit_model.encoder.layers:
            outputs = blk(outputs)
        cls_embedding = vit_model.encoder.ln(outputs[:, 0, :])  # CLS token
        encodings[img_file] = cls_embedding.squeeze(0)  # remove batch dim

# Convert embeddings dict to tensor
embedding_list = torch.stack(list(encodings.values()))  # shape: [num_images, 768]


print(embedding_list.shape)

