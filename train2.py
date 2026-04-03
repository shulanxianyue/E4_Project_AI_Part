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
NUM_CLASSES = 13    
# [NEW] Early Stopping Patience (Stop if Val Loss doesn't improve for 5 consecutive epochs)
EARLY_STOPPING_PATIENCE = 5     
    
def get_class_weights(device):
    """
    Highly customized class weights based on the actual mIoU evaluation report.
    Forces the model to focus on 0% classes and ignore easily learned background classes.
    """
    weights = torch.ones(NUM_CLASSES, dtype=torch.float32)
    weights[3] = 2.0  # Pole
    weights[4] = 3.0  # TrafficLight
    weights[5] = 3.0  # TrafficSign
    weights[8] = 5.0  # Pedestrian
    weights[9] = 3.0  # Vehicles
    weights[10] = 3.0 # Two-Wheelers
    
    weights[1] = 0.5  # Flat Ground
    weights[2] = 0.5  # Structures
    weights[6] = 0.5  # Vegetation
    weights[7] = 0.2  # Sky

    return weights.to(device)   

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Using device: {device} ===")

    print("Loading datasets in Original Resolution (800x600) with Augmentation...")
    train_dataset = CarlaSegmentationDataset(root_dir=DATA_DIR, split='train')
    val_dataset = CarlaSegmentationDataset(root_dir=DATA_DIR, split='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Initializing ResNet101 model for {NUM_CLASSES} classes with Weighted Loss...")
    model = get_carla_model(num_classes=NUM_CLASSES)
    model = model.to(device)

    # Apply the highly customized weights
    class_weights = get_class_weights(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    print(f"\n=== Starting {NUM_EPOCHS}-Epoch Training with Early Stopping (Patience: {EARLY_STOPPING_PATIENCE}) ===")
    
    best_val_loss = float('inf')
    patience_counter = 0  # [NEW] Tracks epochs without improvement
    actual_epochs = 0     # [NEW] Tracks how many epochs actually completed
    
    # Lists to store loss values for plotting
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
            
            # Catch out-of-bounds IDs
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

        # Record losses for the plot
        history_train_loss.append(avg_train_loss)
        history_val_loss.append(avg_val_loss)

        print(f"\n-> Epoch [{epoch+1}/{NUM_EPOCHS}] Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
        print(f"   Time Taken: {epoch_time:.2f} seconds")

        # [MODIFIED] Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset patience since we found a better model
            save_path = "best_carla_model_13classes_weighted.pth" 
            torch.save(model.state_dict(), save_path)
            print(f"   [!] Val loss improved. Model saved to {save_path}\n")
        else:
            patience_counter += 1
            print(f"   [!] Val loss did not improve. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}\n")

        # Trigger Early Stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"🛑 Early stopping triggered! Validation loss hasn't improved for {EARLY_STOPPING_PATIENCE} epochs.")
            break

    print(f"=== Training Completed after {actual_epochs} Epochs! ===")

    # ==========================================
    # Generate and Save Loss Curve (Dynamic X-Axis)
    # ==========================================
    print("Generating Loss Curve...")
    plt.figure(figsize=(10, 6))
    
    # [MODIFIED] Use actual_epochs to match the length of history lists
    epochs_range = range(1, actual_epochs + 1) 
    
    plt.plot(epochs_range, history_train_loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, history_val_loss, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('Training and Validation Loss Curve', fontsize=16)
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