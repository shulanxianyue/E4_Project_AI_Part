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
MODEL_WEIGHTS = "best_carla_model_13classes_weighted.pth" 
NUM_CLASSES = 13

# ==========================================
# Ultimate Class Merging Mapping (13 Classes)
# ==========================================
LABEL_MAPPING = np.full(29, 255, dtype=np.uint8)

LABEL_MAPPING[0] = 0                           # Unlabeled
LABEL_MAPPING[1] = 1                           # Roads
LABEL_MAPPING[2] = 2                           # SideWalks
LABEL_MAPPING[24] = 3                          # RoadLine
LABEL_MAPPING[[14, 15, 16, 18, 19]] = 4        # Vehicles/Agents (Car, Truck, Bus, otorcycle, Bicycle)
LABEL_MAPPING[12] = 5                          # Pedestrian
LABEL_MAPPING[7] = 6                           # TrafficLight
LABEL_MAPPING[8] = 7                           # TrafficSign
LABEL_MAPPING[6] = 8                           # Pole
LABEL_MAPPING[[3, 4, 5, 26, 27, 28]] = 9      # Structures (Building, Wall, Fence, Bridge, RailTrack, GuardRail)
LABEL_MAPPING[[9, 10, 25]] = 10                # Nature/Terrain (Vegetation, Terrain, Ground)
LABEL_MAPPING[11] = 11                         # Sky
LABEL_MAPPING[[20, 21, 22, 23]] = 12           # Obstacles/Misc (Static, Dynamic, Other, Water)

# ==========================================
#  AD-Focused Palette for 13 Classes
# ==========================================
SUPER_CLASS_COLORS = np.array([
    [0, 0, 0],       # 0: Unlabeled (Black)
    [128, 64, 128],  # 1: Road (Purple)
    [244, 35, 232],  # 2: Sidewalk (Pink)
    [157, 234, 50],  # 3: RoadLine (Bright Green/Yellow)
    [0, 0, 142],     # 4: Vehicles (Dark Blue)
    [220, 20, 60],   # 5: Pedestrian (Crimson)
    [250, 170, 30],  # 6: TrafficLight (Orange)
    [220, 220, 0],   # 7: TrafficSign (Yellow)
    [153, 153, 153], # 8: Pole (Light Grey)
    [70, 70, 70],    # 9: Structures (Dark Grey)
    [107, 142, 35],  # 10: Nature/Terrain (Forest Green)
    [70, 130, 180],  # 11: Sky (Sky Blue)
    [110, 190, 160]  # 12: Obstacles/Misc (Teal)
], dtype=np.uint8)

def decode_segmap(image):
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
    plt.suptitle("CARLA Semantic Segmentation (ResNet101 - 13 AD Classes)", fontsize=18)

    with torch.no_grad(): 
        for i, img_name in enumerate(sample_images):
            rgb_path = os.path.join(rgb_dir, img_name)
            original_img = cv2.imread(rgb_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            mask_path = os.path.join(mask_dir, img_name)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            gt_mask = LABEL_MAPPING[gt_mask]
            gt_mask[gt_mask == 255] = 0 
            
            input_tensor = transform(original_img).unsqueeze(0).to(device) 
            output = model(input_tensor)['out']
            pred_mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

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
    print(f"Inference complete! Visualizations saved to {save_path}")

if __name__ == "__main__":
    main()