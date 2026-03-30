import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

# Import custom modules
from dataset import CarlaSegmentationDataset
from model import get_carla_model

# ==========================================
# Hyperparameters (Optimized for 10-Epoch Quick Run)
# ==========================================
DATA_DIR = "./datasets/split_data_by_map"
BATCH_SIZE = 2           
LEARNING_RATE = 1e-4     
# [MODIFIED] Quick training run, only 10 epochs to test our new strategy
NUM_EPOCHS = 10          
NUM_CLASSES = 29         

def get_class_weights(device):
    """
    [NEW] Custom class weights for the highly imbalanced CARLA dataset.
    Forces the model to focus on critical autonomous driving classes 
    that occupy very few pixels but are vital for safety.
    """
    weights = torch.ones(NUM_CLASSES, dtype=torch.float32)
    
    # 1. High Penalty Group (Weight x10): Loss will spike if the model misses these
    weights[4] = 10.0   # Pedestrian
    weights[10] = 10.0  # Vehicles
    weights[12] = 10.0  # Traffic sign
    weights[18] = 10.0  # Traffic Light
    
    # 2. Medium Penalty Group (Weight x5): Thin or rare objects (Addressing your 0% issue)
    weights[5] = 5.0    # Pole
    weights[6] = 5.0    # Road line
    weights[15] = 5.0   # Bridge
    
    # 3. Reduced Weight Group (Weight x0.5): Massive background classes
    weights[1] = 0.5    # Building
    weights[9] = 0.5    # Vegetation
    weights[11] = 0.5   # Wall
    
    return weights.to(device)

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Using device: {device} ===")

    # 2. Load Datasets and DataLoaders
    print("Loading datasets in Original Resolution (800x600)...")
    train_dataset = CarlaSegmentationDataset(root_dir=DATA_DIR, split='train')
    val_dataset = CarlaSegmentationDataset(root_dir=DATA_DIR, split='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 3. Initialize Model, Loss, Optimizer, Scheduler, and AMP Scaler
    print("Initializing ResNet101 model with Weighted Loss...")
    model = get_carla_model(num_classes=NUM_CLASSES)
    model = model.to(device)

    # [MODIFIED] Inject the custom class weights into the Loss function
    class_weights = get_class_weights(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # T_max is automatically adapted to 10, ensuring LR drops to minimum at the end
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    # [MODIFIED] Fixed the PyTorch FutureWarning by using the updated syntax
    scaler = torch.amp.GradScaler('cuda')

    # 4. Training Loop
    print(f"\n=== Starting 10-Epoch Weighted Training ===")
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train() 
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            masks[masks >= NUM_CLASSES] = 255

            optimizer.zero_grad()

            # [MODIFIED] Using the latest recommended official syntax to eliminate warnings
            with torch.amp.autocast('cuda'):
                outputs = model(images)['out'] 
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # --- VALIDATION PHASE ---
        model.eval() 
        val_loss = 0.0
        
        with torch.no_grad(): 
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                masks[masks >= NUM_CLASSES] = 255
                
                with torch.amp.autocast('cuda'): 
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
            # [MODIFIED] New name to avoid overwriting your previous unweighted checkpoint
            save_path = "best_carla_model_10epoch_weighted.pth" 
            torch.save(model.state_dict(), save_path)
            print(f"   [!] Val loss improved. Model saved to {save_path}\n")
        else:
            print("\n")

    print("=== 10-Epoch Training Complete! ===")

if __name__ == "__main__":
    main()