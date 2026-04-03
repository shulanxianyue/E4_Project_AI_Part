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
TEST_DIR = "./datasets/explicit_map_split/test"
MODEL_WEIGHTS = "best_carla_model_27classes_weighted.pth"
# [MODIFIED] Set to 27 classes!
NUM_CLASSES = 27

# [MODIFIED] The NEW CARLA Mapping matching the shifted 27 classes
# Rider (13) and Train (17) are removed. The rest shift upwards.
CARLA_CLASSES = {
    0: "Unlabeled", 1: "Roads", 2: "SideWalks", 3: "Building", 4: "Wall",
    5: "Fence", 6: "Pole", 7: "TrafficLight", 8: "TrafficSign", 9: "Vegetation",
    10: "Terrain", 11: "Sky", 12: "Pedestrian", 
    # Notice the shift here:
    13: "Car", 14: "Truck", 15: "Bus", 
    # Train is skipped
    16: "Motorcycle", 17: "Bicycle", 18: "Static", 19: "Dynamic", 
    20: "Other", 21: "Water", 22: "RoadLine", 23: "Ground", 
    24: "Bridge", 25: "RailTrack", 26: "GuardRail"
}

# [NEW] The exact same mapping logic from dataset.py to convert disk labels
LABEL_MAPPING = np.full(29, 255, dtype=np.uint8)
new_id = 0
for old_id in range(29):
    if old_id == 13 or old_id == 17:
        continue
    LABEL_MAPPING[old_id] = new_id
    new_id += 1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting mIoU Evaluation on: {device} ===")

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
            
            # [MODIFIED] Remap the ground truth mask from disk (0-28) to our new (0-26) space
            gt_mask = LABEL_MAPPING[gt_mask]
            
            input_tensor = transform(original_img).unsqueeze(0).to(device)
            output = model(input_tensor)['out']
            pred_mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

            # [MODIFIED] Filter out 255 (the ignored classes)
            valid_pixels = gt_mask != 255
            pred_valid = pred_mask[valid_pixels]
            gt_valid = gt_mask[valid_pixels]

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