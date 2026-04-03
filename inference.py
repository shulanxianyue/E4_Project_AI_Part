import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from model import get_carla_model

# ==========================================
# Configuration 
# ==========================================
TEST_DIR = "./datasets/explicit_map_split/test"
MODEL_WEIGHTS = "best_carla_model_13classes_weighted.pth" # Ensure this matches your 13-class saved model name
NUM_CLASSES = 13

# ==========================================
# [NEW] Ultimate Class Merging Mapping 
# Transforms raw 0-28 CARLA IDs into 0-12 Super-Class IDs
# ==========================================
LABEL_MAPPING = np.full(29, 255, dtype=np.uint8)
LABEL_MAPPING[0] = 0
LABEL_MAPPING[[1, 2, 10, 24, 25]] = 1      # Flat Ground
LABEL_MAPPING[[3, 4, 5, 26, 27, 28]] = 2   # Structures
LABEL_MAPPING[6] = 3                       # Pole
LABEL_MAPPING[7] = 4                       # TrafficLight
LABEL_MAPPING[8] = 5                       # TrafficSign
LABEL_MAPPING[9] = 6                       # Vegetation
LABEL_MAPPING[11] = 7                      # Sky
LABEL_MAPPING[12] = 8                      # Pedestrian
LABEL_MAPPING[[14, 15, 16, 17]] = 9        # Vehicles
LABEL_MAPPING[[18, 19]] = 10               # Two-Wheelers
LABEL_MAPPING[[20, 21, 22]] = 11           # Obstacles
LABEL_MAPPING[23] = 12                     # Water

# ==========================================
# [NEW] Refined 13-Color Palette for Visualization
# Distinct colors for merged categories to make it easy on the eyes
# ==========================================
SUPER_CLASS_COLORS = np.array([
    [0, 0, 0],       # 0: Unlabeled (Black)
    [128, 64, 128],  # 1: Flat Ground (Purple/Grey Road color)
    [70, 70, 70],    # 2: Structures (Dark Grey)
    [153, 153, 153], # 3: Pole (Light Grey)
    [250, 170, 30],  # 4: TrafficLight (Orange)
    [220, 220, 0],   # 5: TrafficSign (Yellow)
    [107, 142, 35],  # 6: Vegetation (Green)
    [70, 130, 180],  # 7: Sky (Blue)
    [220, 20, 60],   # 8: Pedestrian (Red)
    [0, 0, 142],     # 9: Vehicles (Dark Blue)
    [119, 11, 32],   # 10: Two-Wheelers (Dark Red)
    [110, 190, 160], # 11: Obstacles (Olive/Teal)
    [45, 60, 150]    # 12: Water (Deep Blue)
], dtype=np.uint8)

def decode_segmap(image):
    """
    Decodes the 2D tensor of 13 super-class indices into a vivid RGB image.
    """
    color_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for class_id in range(NUM_CLASSES):
        color_mask[image == class_id] = SUPER_CLASS_COLORS[class_id]
    return color_mask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    print(f"Loading model weights from {MODEL_WEIGHTS}...")
    model = get_carla_model(num_classes=NUM_CLASSES)
    
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"Error: {MODEL_WEIGHTS} not found.")
        return
        
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model = model.to(device)
    model.eval() 

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    rgb_dir = os.path.join(TEST_DIR, 'rgb')
    mask_dir = os.path.join(TEST_DIR, 'mask')
    
    all_images = [f for f in os.listdir(rgb_dir) if f.endswith('.png')]
    sample_images = random.sample(all_images, 3)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    plt.suptitle("CARLA Semantic Segmentation (ResNet101 - 13 Classes)", fontsize=18)

    with torch.no_grad(): 
        for i, img_name in enumerate(sample_images):
            # --- Load Original RGB Image ---
            rgb_path = os.path.join(rgb_dir, img_name)
            original_img = cv2.imread(rgb_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # --- Load and Remap Ground Truth Mask ---
            mask_path = os.path.join(mask_dir, img_name)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Apply Super-Class mapping immediately
            gt_mask = LABEL_MAPPING[gt_mask]
            
            # Map ignored classes (255) to 0 (Unlabeled) to prevent visualization errors
            gt_mask[gt_mask == 255] = 0 
            
            # --- Model Prediction ---
            input_tensor = transform(original_img).unsqueeze(0).to(device) 
            output = model(input_tensor)['out']
            pred_mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

            # --- Visualization ---
            axes[i, 0].imshow(original_img)
            axes[i, 0].set_title("Original RGB (800x600)")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(decode_segmap(gt_mask))
            axes[i, 1].set_title("Ground Truth Mask (13 Classes)")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(decode_segmap(pred_mask))
            axes[i, 2].set_title("ResNet101 Prediction")
            axes[i, 2].axis('off')

    plt.tight_layout()
    save_path = "inference_results_13classes.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Inference complete! High-resolution visualization saved to {save_path}")

if __name__ == "__main__":
    main()