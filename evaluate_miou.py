import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm

from model import get_carla_model

# ==========================================
# Configuration for Evaluation
# ==========================================
TEST_DIR = "./datasets/explicit_map_split/test" # Change to 'val' if you are evaluating on validation set
MODEL_WEIGHTS = "best_carla_model_13classes_weighted.pth" # Update this to your latest 13-class saved model name if different
NUM_CLASSES = 13

# ==========================================
# [NEW] The 13 Super-Classes Dictionary
# ==========================================
CARLA_CLASSES = {
    0: "Unlabeled", 
    1: "Flat Ground", 
    2: "Structures", 
    3: "Pole", 
    4: "TrafficLight", 
    5: "TrafficSign", 
    6: "Vegetation", 
    7: "Sky", 
    8: "Pedestrian", 
    9: "Vehicles", 
    10: "Two-Wheelers", 
    11: "Obstacles", 
    12: "Water"
}

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
# Class 13 (Rider) remains 255

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting mIoU Evaluation on: {device} ===")

    # Initialize model with 13 classes
    model = get_carla_model(num_classes=NUM_CLASSES)
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"Error: {MODEL_WEIGHTS} not found!")
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
    test_images = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    
    print(f"Found {len(test_images)} images in the Test set.")

    total_intersection = np.zeros(NUM_CLASSES)
    total_union = np.zeros(NUM_CLASSES)

    with torch.no_grad():
        for img_name in tqdm(test_images, desc="Evaluating Images"):
            rgb_path = os.path.join(rgb_dir, img_name)
            original_img = cv2.imread(rgb_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            mask_path = os.path.join(mask_dir, img_name)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # [MODIFIED] Instantly remap 29 classes into 13 super-classes
            gt_mask = LABEL_MAPPING[gt_mask]
            
            input_tensor = transform(original_img).unsqueeze(0).to(device)
            output = model(input_tensor)['out']
            pred_mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

            # Filter out ignored pixels (255)
            valid_pixels = gt_mask != 255
            pred_valid = pred_mask[valid_pixels]
            gt_valid = gt_mask[valid_pixels]

            # Calculate Intersection and Union for each valid class
            for cls in range(NUM_CLASSES):
                p = (pred_valid == cls)
                t = (gt_valid == cls)
                
                total_intersection[cls] += np.logical_and(p, t).sum()
                total_union[cls] += np.logical_or(p, t).sum()

    ious = []
    print("\n" + "="*50)
    print(f"{'CLASS NAME':<20} | {'IoU SCORE':<10}")
    print("="*50)
    
    for cls in range(NUM_CLASSES):
        if total_union[cls] == 0:
            continue 
            
        iou = total_intersection[cls] / total_union[cls]
        ious.append(iou)
        class_name = CARLA_CLASSES.get(cls, f"Class_{cls}")
        
        print(f"{class_name:<20} | {iou * 100:.2f}%")

    mIoU = np.mean(ious)
    
    print("="*50)
    print(f"FINAL mIoU SCORE       | {mIoU * 100:.2f}%")
    print("="*50)
    print("Evaluation Complete!")

if __name__ == "__main__":
    main()