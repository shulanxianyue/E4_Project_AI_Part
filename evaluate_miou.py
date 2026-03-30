import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm # Professional progress bar

# Import the model definition
from model import get_carla_model

# ==========================================
# Configuration for Evaluation
# ==========================================
TEST_DIR = "./datasets/split_data_by_map/test"
MODEL_WEIGHTS = "best_carla_model_resnet101_highres.pth"
NUM_CLASSES = 29

# Standard CARLA semantic class mapping for human-readable reporting
CARLA_CLASSES = {
    0: "Unlabeled", 1: "Building", 2: "Fence", 3: "Other", 4: "Pedestrian",
    5: "Pole", 6: "Road line", 7: "Road", 8: "Sidewalk", 9: "Vegetation",
    10: "Vehicles", 11: "Wall", 12: "Traffic sign", 13: "Sky", 14: "Ground",
    15: "Bridge", 16: "Rail track", 17: "GuardRail", 18: "Traffic Light",
    19: "Static", 20: "Dynamic", 21: "Water", 22: "Terrain"
}
# Fill any remaining experimental classes with generic names
for i in range(23, NUM_CLASSES):
    CARLA_CLASSES[i] = f"Class_{i}"

def main():
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting mIoU Evaluation on: {device} ===")

    # 2. Load the best high-res model
    model = get_carla_model(num_classes=NUM_CLASSES)
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"Error: {MODEL_WEIGHTS} not found!")
        return
        
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model = model.to(device)
    model.eval() # CRITICAL for evaluation

    # 3. Define transformations (Normalization only, no resizing)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Prepare dataset paths
    rgb_dir = os.path.join(TEST_DIR, 'rgb')
    mask_dir = os.path.join(TEST_DIR, 'mask')
    test_images = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    
    print(f"Found {len(test_images)} images in the Test set (Town10).")

    # Arrays to accumulate intersection and union over the ENTIRE dataset
    total_intersection = np.zeros(NUM_CLASSES)
    total_union = np.zeros(NUM_CLASSES)

    # 5. Evaluation Loop with Progress Bar
    with torch.no_grad():
        for img_name in tqdm(test_images, desc="Evaluating Images"):
            # Load RGB
            rgb_path = os.path.join(rgb_dir, img_name)
            original_img = cv2.imread(rgb_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # Load Ground Truth Mask
            mask_path = os.path.join(mask_dir, img_name)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Forward pass
            input_tensor = transform(original_img).unsqueeze(0).to(device)
            output = model(input_tensor)['out']
            pred_mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

            # Filter out invalid pixels (e.g., boundaries or ignored labels >= 29)
            valid_pixels = gt_mask < NUM_CLASSES
            pred_valid = pred_mask[valid_pixels]
            gt_valid = gt_mask[valid_pixels]

            # Calculate Intersection and Union for each class in this specific image
            for cls in range(NUM_CLASSES):
                p = (pred_valid == cls)
                t = (gt_valid == cls)
                
                total_intersection[cls] += np.logical_and(p, t).sum()
                total_union[cls] += np.logical_or(p, t).sum()

    # 6. Compute Final IoU per class and Mean IoU (mIoU)
    ious = []
    print("\n" + "="*50)
    print(f"{'CLASS NAME':<20} | {'IoU SCORE':<10}")
    print("="*50)
    
    for cls in range(NUM_CLASSES):
        if total_union[cls] == 0:
            # If a class never appears in the test set, skip it to avoid dividing by zero
            continue 
            
        iou = total_intersection[cls] / total_union[cls]
        ious.append(iou)
        class_name = CARLA_CLASSES.get(cls, f"Class_{cls}")
        
        # Print score for this class (formatted as percentage)
        print(f"{class_name:<20} | {iou * 100:.2f}%")

    # Calculate overall Mean IoU
    mIoU = np.mean(ious)
    
    print("="*50)
    print(f"FINAL mIoU SCORE       | {mIoU * 100:.2f}%")
    print("="*50)
    print("Evaluation Complete!")

if __name__ == "__main__":
    main()