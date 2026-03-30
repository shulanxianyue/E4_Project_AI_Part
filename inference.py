import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Import your custom model definition
from model import get_carla_model

# ==========================================
# Configuration
# ==========================================
TEST_DIR = "./datasets/split_data_by_map/test"
MODEL_WEIGHTS = "best_carla_model.pth"
NUM_CLASSES = 29
IMG_SIZE = (400, 300)

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

    # 2. Initialize model and load trained weights
    model = get_carla_model(num_classes=NUM_CLASSES)
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"Error: {MODEL_WEIGHTS} not found. Please train the model first.")
        return
        
    # Load the state dictionary (weights) into the model
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model = model.to(device)
    model.eval() # CRITICAL: Set model to evaluation mode

    # 3. Define image transformations (must match training pipeline)
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

    # 5. Setup Matplotlib figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    plt.suptitle("CARLA Semantic Segmentation Inference (Test Set - Town10)", fontsize=16)

    with torch.no_grad(): # No need to track gradients for inference
        for i, img_name in enumerate(sample_images):
            # --- Load RGB Image ---
            rgb_path = os.path.join(rgb_dir, img_name)
            original_img = cv2.imread(rgb_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            original_img = cv2.resize(original_img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            
            # --- Load Ground Truth Mask ---
            mask_path = os.path.join(mask_dir, img_name)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.resize(gt_mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
            # Handle out-of-bounds classes by capping them
            gt_mask[gt_mask >= NUM_CLASSES] = 0 
            
            # --- Model Prediction ---
            # Prepare tensor
            input_tensor = transform(original_img).unsqueeze(0).to(device) # Add batch dimension
            
            # Forward pass
            output = model(input_tensor)['out']
            # Get the class with the highest probability for each pixel
            pred_mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

            # --- Visualization ---
            # Plot Original Image
            axes[i, 0].imshow(original_img)
            axes[i, 0].set_title("Original RGB")
            axes[i, 0].axis('off')

            # Plot Ground Truth (Colorized)
            axes[i, 1].imshow(decode_segmap(gt_mask))
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 1].axis('off')

            # Plot Prediction (Colorized)
            axes[i, 2].imshow(decode_segmap(pred_mask))
            axes[i, 2].set_title("Model Prediction")
            axes[i, 2].axis('off')

    plt.tight_layout()
    # Save the result as an image file so you can view it easily
    save_path = "inference_results.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Inference complete! Visualization saved to {save_path}")

if __name__ == "__main__":
    main()