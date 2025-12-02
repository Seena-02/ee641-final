import vec2text
import torch

device = torch.device("mps")  # for macOS GPU via Metal
# If you want CPU fallback:
# device = torch.device("cpu")

corrector = vec2text.load_pretrained_corrector("text-embedding-ada-002")

# Move the corrector's model to the device
corrector.inversion_model.to(device)
corrector.corrector_model.to(device)
print(corrector)
print("Corrector loaded successfully.")
