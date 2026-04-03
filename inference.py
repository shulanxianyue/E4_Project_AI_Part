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
MODEL_WEIGHTS = "best_carla_model_10epoch_weighted.pth"
NUM_CLASSES = 27 # [MODIFIED] Updated to 27

# [NEW] Exact mapping array to fix GT visualizer
LABEL_MAPPING = np.full(29, 255, dtype=np.uint8)
new_id = 0
for old_id in range(29):
    if old_id == 13 or old_id == 17:
        continue
    LABEL_MAPPING[old_id] = new_id
    new_id += 1

# [MODIFIED] Official Palette updated for the shifted 27 classes
TRUE_COLORS = np.array([
    [0, 0, 0],       # 0: Unlabeled
    [128, 64, 128],  # 1: Roads
    [244, 35, 232],  # 2: SideWalks
    [70, 70, 70],    # 3: Building
    [102, 102, 156], # 4: Wall
    [190, 153, 153], # 5: Fence
    [153, 153, 153], # 6: Pole
    [250, 170, 30],  # 7: TrafficLight
    [220, 220, 0],   # 8: TrafficSign
    [107, 142, 35],  # 9: Vegetation
    [152, 251, 152], # 10: Terrain
    [70, 130, 180],  # 11: Sky
    [220, 20, 60],   # 12: Pedestrian
    [0, 0, 142],     # 13: Car (Old 14)
    [0, 0, 70],      # 14: Truck (Old 15)
    [0, 60, 100],    # 15: Bus (Old 16)
    [0, 0, 230],     # 16: Motorcycle (Old 18)
    [119, 11, 32],   # 17: Bicycle (Old 19)
    [110, 190, 160], # 18: Static (Old 20)
    [170, 120, 50],  # 19: Dynamic (Old 21)
    [55, 90, 80],    # 20: Other (Old 22)
    [45, 60, 150],   # 21: Water (Old 23)
    [157, 234, 50],  # 22: RoadLine (Old 24)
    [81, 0, 81],     # 23: Ground (Old 25)
    [150, 100, 100], # 24: Bridge (Old 26)
    [230, 150, 140], # 25: RailTrack (Old 27)
    [180, 165, 180]  # 26: GuardRail (Old 28)
], dtype=np.uint8)

def decode_segmap(image):
    """Decodes new 0-26 tensors into official RGB colors."""
    color_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for class_id in range(NUM_CLASSES):
        color_mask[image == class_id] = TRUE_COLORS[class_id]
    return color_mask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

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
    plt.suptitle("CARLA High-Resolution Semantic Segmentation (ResNet101)", fontsize=18)

    with torch.no_grad(): 
        for i, img_name in enumerate(sample_images):
            rgb_path = os.path.join(rgb_dir, img_name)
            original_img = cv2.imread(rgb_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            mask_path = os.path.join(mask_dir, img_name)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # [MODIFIED] Translate GT disk tags to 27 valid IDs before plotting!
            gt_mask = LABEL_MAPPING[gt_mask]
            
            input_tensor = transform(original_img).unsqueeze(0).to(device) 
            output = model(input_tensor)['out']
            pred_mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

            axes[i, 0].imshow(original_img)
            axes[i, 0].set_title("Original RGB (800x600)")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(decode_segmap(gt_mask))
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(decode_segmap(pred_mask))
            axes[i, 2].set_title("ResNet101 Prediction")
            axes[i, 2].axis('off')

    plt.tight_layout()
    save_path = "inference_results_highres.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Inference complete! Visualizations saved to {save_path}")

if __name__ == "__main__":
    main()