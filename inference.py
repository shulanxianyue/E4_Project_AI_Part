import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Import the updated ResNet101 model definition
from model import get_carla_model

# ==========================================
# Configuration (Updated for High-Res ResNet101)
# ==========================================
TEST_DIR = "./datasets/split_data_by_map/test"
# Ensure this matches the save_path in your train.py
MODEL_WEIGHTS = "best_carla_model_resnet101_highres.pth" 
NUM_CLASSES = 29
# Note: IMG_SIZE is completely removed because we are doing inference on the original 800x600 images!

def decode_segmap(image, nc=29):
    """
    Decodes the 2D tensor of class indices into a color image for visualization.
    Assigns a unique RGB color to each class ID.
    """
    # Create a basic color palette (RGB) for up to 29 classes
    label_colors = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], 
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        [0, 64, 128], [128, 64, 128], [0, 192, 128], [128, 192, 128],
        [64, 64, 0], [192, 64, 0], [64, 192, 0], [192, 192, 0],
        [64, 64, 128]
    ])
    
    # Initialize an empty RGB image
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    # Map each class index to its corresponding color
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
        
    return np.stack([r, g, b], axis=2)

def main():
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # 2. Initialize ResNet101 model and load trained high-res weights
    print(f"Loading model weights from {MODEL_WEIGHTS}...")
    model = get_carla_model(num_classes=NUM_CLASSES)
    
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"Error: {MODEL_WEIGHTS} not found. Please wait for the training to finish.")
        return
        
    # Load the state dictionary (weights) into the model
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model = model.to(device)
    model.eval() # CRITICAL: Set model to evaluation mode

    # 3. Define standard image transformations (Normalization only, no resizing)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 4. Pick random images from the Test set (Town10)
    rgb_dir = os.path.join(TEST_DIR, 'rgb')
    mask_dir = os.path.join(TEST_DIR, 'mask')
    
    all_images = [f for f in os.listdir(rgb_dir) if f.endswith('.png')]
    # Select 3 random images for visualization
    sample_images = random.sample(all_images, 3)

    # 5. Setup Matplotlib figure (Increased figsize for high-res images)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    plt.suptitle("CARLA High-Resolution Semantic Segmentation (ResNet101 - Town10)", fontsize=18)

    with torch.no_grad(): # Disable gradient calculation to save memory
        for i, img_name in enumerate(sample_images):
            # --- Load Original High-Res RGB Image ---
            rgb_path = os.path.join(rgb_dir, img_name)
            original_img = cv2.imread(rgb_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # --- Load Original High-Res Ground Truth Mask ---
            mask_path = os.path.join(mask_dir, img_name)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Handle out-of-bounds classes by mapping them to 0 (Unlabeled) for visualization
            gt_mask[gt_mask >= NUM_CLASSES] = 0 
            
            # --- Model Prediction ---
            # Prepare tensor and add batch dimension [1, C, H, W]
            input_tensor = transform(original_img).unsqueeze(0).to(device) 
            
            # Forward pass through the ResNet101 model
            output = model(input_tensor)['out']
            
            # Get the class with the highest probability for each pixel
            pred_mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

            # --- Visualization ---
            # Plot Original Image
            axes[i, 0].imshow(original_img)
            axes[i, 0].set_title("Original RGB (800x600)")
            axes[i, 0].axis('off')

            # Plot Ground Truth (Colorized)
            axes[i, 1].imshow(decode_segmap(gt_mask))
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 1].axis('off')

            # Plot Prediction (Colorized)
            axes[i, 2].imshow(decode_segmap(pred_mask))
            axes[i, 2].set_title("ResNet101 Prediction")
            axes[i, 2].axis('off')

    plt.tight_layout()
    # Save the high-res result as an image file
    save_path = "inference_results_highres.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Inference complete! High-resolution visualization saved to {save_path}")

if __name__ == "__main__":
    main()