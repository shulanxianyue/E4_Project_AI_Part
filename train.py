import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

# Import your custom modules
from dataset import CarlaSegmentationDataset
from model import get_carla_model

# ==========================================
# Hyperparameters (Adjust these based on your laptop's capability)
# ==========================================
DATA_DIR = "./datasets/split_data_by_map"
BATCH_SIZE = 4           # If CUDA out of memory, reduce this to 2
LEARNING_RATE = 1e-4     # How fast the model learns
NUM_EPOCHS = 10          # Number of times to loop over the entire dataset
NUM_CLASSES = 29         # CARLA standard semantic classes
IMG_SIZE = (400, 300)    # Resized image dimensions

def main():
    # 1. Setup Device (Use GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Using device: {device} ===")

    # 2. Load Datasets and DataLoaders
    print("Loading datasets...")
    train_dataset = CarlaSegmentationDataset(root_dir=DATA_DIR, split='train', img_size=IMG_SIZE)
    val_dataset = CarlaSegmentationDataset(root_dir=DATA_DIR, split='val', img_size=IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # 3. Initialize Model, Loss Function, and Optimizer
    print("Initializing model...")
    model = get_carla_model(num_classes=NUM_CLASSES)
    model = model.to(device) # Move model to GPU

    # CrossEntropyLoss is standard for multi-class classification
    # CrossEntropyLoss is standard for multi-class classification
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    # AdamW is a robust optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    print("\n=== Starting Training ===")
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train() # Set model to training mode
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            # Move data to GPU
            images = images.to(device)
            masks = masks.to(device)

            masks[masks >= NUM_CLASSES] = 255

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)['out'] 
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval() # Set model to evaluation mode (turns off Dropout/BatchNorm updates)
        val_loss = 0.0
        
        with torch.no_grad(): # Disable gradient calculation to save memory/compute
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - start_time

        print(f"\n-> Epoch [{epoch+1}/{NUM_EPOCHS}] Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   Time Taken: {epoch_time:.2f} seconds")

        # 5. Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = "best_carla_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"   [!] Val loss improved. Model saved to {save_path}\n")
        else:
            print("\n")

    print("=== Training Complete! ===")

if __name__ == "__main__":
    main()