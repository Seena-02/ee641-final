"""
Training and evaluation utilities for maze solver.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


def train_model(model, train_loader, epochs=75, lr=5e-4, device='cuda'):
    """
    Train the model.
    
    Args:
        model: ResNetGPT2PrefixModel instance
        train_loader: DataLoader for training data
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        final_loss: Final average loss
    """
    model.train()
    
    # Optimizer with differentiated learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.resnet_features.parameters(), 'lr': lr / 10},  # Lower LR for pretrained
        {'params': model.feature_projection.parameters(), 'lr': lr},
        {'params': model.prefix_generator.parameters(), 'lr': lr},
        {'params': model.gpt2.parameters(), 'lr': lr / 5},  # Lower LR for GPT2
    ], weight_decay=0.01)
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs, 
        eta_min=1e-6
    )
    
    final_loss = 0.0
    
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            
            # Prepare input_ids (labels without EOS) and labels
            input_ids = labels[:, :-1].clone()
            labels = labels[:, 1:].clone()
            
            # Mask padding tokens in labels
            labels[labels == 0] = -100
            
            # Forward pass
            outputs = model(images, input_ids, labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
        
        final_loss = avg_loss
    
    return final_loss


def test_model(model, test_loader, device, tokenizer, num_samples=None):
    """
    Test the model and count exact matches.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run on
        tokenizer: Tokenizer instance
        num_samples: Number of samples to test (None = all)
    
    Returns:
        correct_count: Number of exact matches
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            
            # Generate predictions
            outputs = model.generate(
                images,
                max_length=20,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
            )
            
            # Compare predictions with ground truth
            for i in range(len(images)):
                expected = tokenizer.decode(labels[i].tolist())
                predicted = tokenizer.decode(outputs[i].tolist())
                
                # Remove special tokens for comparison
                expected_clean = [t for t in expected if t not in ['<pad>', '<s>', '</s>', '<unk>']]
                predicted_clean = [t for t in predicted if t not in ['<pad>', '<s>', '</s>', '<unk>']]
                
                if expected_clean == predicted_clean:
                    correct += 1
                
                total += 1
                
                if num_samples and total >= num_samples:
                    return correct
    
    return correct
