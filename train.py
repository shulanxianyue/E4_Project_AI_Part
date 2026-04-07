import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import matplotlib.pyplot as plt 

from dataset import CarlaSegmentationDataset
from model import get_carla_model

# ==========================================
# Hyperparameters 
# ==========================================
DATA_DIR = "./datasets/explicit_map_split"
BATCH_SIZE = 2           
LEARNING_RATE = 1e-4     
NUM_EPOCHS = 20       
NUM_CLASSES = 13    # [MODIFIED] Updated to 13 classes
EARLY_STOPPING_PATIENCE = 5     
    
def get_class_weights(device):
    """
    Highly customized class weights based on the 13-class autonomous driving scheme.
    Prioritizes drivable areas, dynamic objects, and traffic signals.
    """
    weights = torch.ones(NUM_CLASSES, dtype=torch.float32)
    
    # --- High Priority: Dynamic & Safety Critical (Weight x3 ~ x5) ---
    weights[5] = 5.0  # Pedestrian
    weights[4] = 3.0  # Vehicles
    weights[6] = 4.0  # TrafficLight
    weights[7] = 4.0  # TrafficSign
    weights[3] = 4.0  # RoadLine
    
    # --- Medium Priority (Weight x1.0) ---
    weights[8] = 1.0  # Pole
    weights[12] = 1.0 # Obstacles/Misc
    
    # --- Low Priority: Massive Backgrounds (Weight < 0.5) ---
    weights[1] = 0.5  # Road
    weights[2] = 0.5  # Sidewalk
    weights[9] = 0.5 # Structures
    weights[10] = 0.5 # Nature/Terrain
    weights[11] = 0.2 # Sky
    
    return weights.to(device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Using device: {device} ===")

    print("Loading datasets with Augmentation...")
    train_dataset = CarlaSegmentationDataset(root_dir=DATA_DIR, split='train')
    val_dataset = CarlaSegmentationDataset(root_dir=DATA_DIR, split='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Initializing ResNet101 model for {NUM_CLASSES} classes with AD Weighted Loss...")
    model = get_carla_model(num_classes=NUM_CLASSES)
    model = model.to(device)

    class_weights = get_class_weights(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    print(f"\n=== Starting {NUM_EPOCHS}-Epoch Training with Early Stopping (Patience: {EARLY_STOPPING_PATIENCE}) ===")
    
    best_val_loss = float('inf')
    patience_counter = 0  
    actual_epochs = 0     
    
    history_train_loss = []
    history_val_loss = []

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        actual_epochs += 1
        
        # --- TRAINING PHASE ---
        model.train() 
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            masks[(masks >= NUM_CLASSES) & (masks != 255)] = 255

            optimizer.zero_grad()

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
                masks[(masks >= NUM_CLASSES) & (masks != 255)] = 255
                
                with torch.amp.autocast('cuda'): 
                    outputs = model(images)['out']
                    loss = criterion(outputs, masks)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - start_time

        history_train_loss.append(avg_train_loss)
        history_val_loss.append(avg_val_loss)

        print(f"\n-> Epoch [{epoch+1}/{NUM_EPOCHS}] Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
        print(f"   Time Taken: {epoch_time:.2f} seconds")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  
            save_path = "best_carla_model_13classes_weighted.pth"  # [MODIFIED] Save name updated
            torch.save(model.state_dict(), save_path)
            print(f"   [!] Val loss improved. Model saved to {save_path}\n")
        else:
            patience_counter += 1
            print(f"   [!] Val loss did not improve. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}\n")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"🛑 Early stopping triggered! Validation loss hasn't improved for {EARLY_STOPPING_PATIENCE} epochs.")
            break

    print(f"=== Training Completed after {actual_epochs} Epochs! ===")

    # ==========================================
    # Generate and Save Loss Curve
    # ==========================================
    print("Generating Loss Curve...")
    plt.figure(figsize=(10, 6))
    
    epochs_range = range(1, actual_epochs + 1) 
    
    plt.plot(epochs_range, history_train_loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, history_val_loss, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('Training and Validation Loss Curve (13 Classes)', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(epochs_range)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plot_path = "loss_curve.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    print(f"Loss curve saved successfully to: {plot_path}")

if __name__ == "__main__":
    main()