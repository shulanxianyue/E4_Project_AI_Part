import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
import time

# Import custom modules
from dataset import CarlaSegmentationDataset
from model import get_carla_model

# ==========================================
# Hyperparameters (Optimized for High-Resolution ResNet101)
# ==========================================
DATA_DIR = "./datasets/split_data_by_map"
# CRITICAL: Batch size MUST be small (e.g., 2 or even 1) 
# to prevent CUDA Out-Of-Memory (OOM) on 800x600 images with ResNet101
BATCH_SIZE = 2           
LEARNING_RATE = 1e-4     
NUM_EPOCHS = 100         
NUM_CLASSES = 29         

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Using device: {device} ===")

    # 2. Load Datasets and DataLoaders
    # Note: No img_size parameter is passed; dataset uses original 800x600 resolution
    print("Loading datasets in Original Resolution (800x600)...")
    train_dataset = CarlaSegmentationDataset(root_dir=DATA_DIR, split='train')
    val_dataset = CarlaSegmentationDataset(root_dir=DATA_DIR, split='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # 3. Initialize Model, Loss, Optimizer, Scheduler, and AMP Scaler
    print("Initializing DeepLabV3-ResNet101 model and advanced training tools...")
    model = get_carla_model(num_classes=NUM_CLASSES)
    model = model.to(device)

    # CrossEntropyLoss with ignore_index=255 handles out-of-bounds/unlabeled pixels
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # AdamW includes weight_decay for regularization (prevents overfitting on high-res details)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Cosine Annealing scheduler smoothly reduces the learning rate for precise fine-tuning
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    # GradScaler enables Automatic Mixed Precision (AMP) to save VRAM and accelerate training
    scaler = GradScaler()

    # 4. Training Loop
    print("\n=== Starting High-Res Training ===")
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train() 
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Filter out invalid or experimental material classes in CARLA
            masks[masks >= NUM_CLASSES] = 255

            # Zero the gradients
            optimizer.zero_grad()

            # Enable AMP context manager for the forward pass
            with autocast():
                outputs = model(images)['out'] 
                loss = criterion(outputs, masks)
            
            # Backward pass and optimization using the Scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            # Print progress every 100 batches to keep terminal clean
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        
        # Step the learning rate scheduler at the end of each epoch
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # --- VALIDATION PHASE ---
        model.eval() 
        val_loss = 0.0
        
        # Disable gradient calculation during validation
        with torch.no_grad(): 
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                masks[masks >= NUM_CLASSES] = 255
                
                # Use AMP in validation to save memory and speed up inference
                with autocast(): 
                    outputs = model(images)['out']
                    loss = criterion(outputs, masks)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - start_time

        print(f"\n-> Epoch [{epoch+1}/{NUM_EPOCHS}] Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
        print(f"   Time Taken: {epoch_time:.2f} seconds")

        # 5. Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save with a specific name for the high-res ResNet101 model
            save_path = "best_carla_model_resnet101_highres.pth" 
            torch.save(model.state_dict(), save_path)
            print(f"   [!] Val loss improved. Model saved to {save_path}\n")
        else:
            print("\n")

    print("=== Training Complete! ===")

if __name__ == "__main__":
    main()