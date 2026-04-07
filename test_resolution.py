import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from model import get_carla_model

# ==========================================
# Configuration for ROS Resolution Test
# ==========================================
TEST_DIR = "./datasets/explicit_map_split/test"
MODEL_WEIGHTS = "best_carla_model_13classes_weighted.pth" 
NUM_CLASSES = 13

# [NEW] Target Resolution from the ROS Camera
TARGET_WIDTH = 1241
TARGET_HEIGHT = 376

# ==========================================
# AD-Focused Palette for 13 Classes
# ==========================================
SUPER_CLASS_COLORS = np.array([
    [0, 0, 0],       # 0: Unlabeled (Black)
    [128, 64, 128],  # 1: Road (Purple)
    [244, 35, 232],  # 2: Sidewalk (Pink)
    [157, 234, 50],  # 3: RoadLine (Bright Green)
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
    """
    Converts a 2D class index array into an RGB image.
    """
    color_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for class_id in range(NUM_CLASSES):
        color_mask[image == class_id] = SUPER_CLASS_COLORS[class_id]
    return color_mask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Testing ROS Camera Resolution ({TARGET_WIDTH}x{TARGET_HEIGHT}) on: {device} ===")

    # 1. Load Model
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

    # 2. Pick a random image from test set
    rgb_dir = os.path.join(TEST_DIR, 'rgb')
    img_name = random.choice([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    
    rgb_path = os.path.join(rgb_dir, img_name)
    original_img = cv2.imread(rgb_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # 3. [CORE LOGIC] Force resize to ROS camera resolution
    print(f"Original image shape: {original_img.shape}")
    resized_img = cv2.resize(original_img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
    print(f"Resized image shape (simulating ROS camera): {resized_img.shape}")

    # 4. Inference
    with torch.no_grad():
        input_tensor = transform(resized_img).unsqueeze(0).to(device) 
        
        # Check model output tensor shape
        output = model(input_tensor)['out']
        print(f"Model output tensor shape: {output.shape}")
        
        pred_mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

    # 5. Visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 6)) # Stacked vertically for ultra-wide images
    plt.suptitle(f"ROS Resolution Inference Test ({TARGET_WIDTH}x{TARGET_HEIGHT})", fontsize=16)

    axes[0].imshow(resized_img)
    axes[0].set_title(f"Simulated ROS Input")
    axes[0].axis('off')

    axes[1].imshow(decode_segmap(pred_mask))
    axes[1].set_title("DeepLabV3 Output")
    axes[1].axis('off')

    plt.tight_layout()
    save_path = "ros_resolution_test.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"\n[SUCCESS] Model handled the resolution perfectly. Visualization saved to: {save_path}")

if __name__ == "__main__":
    main()