import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Your latest complete 29-class mapping table
# The TRUE CARLA mapping for your specific dataset
CARLA_CLASSES = {
    0: "Unlabeled", 1: "Roads", 2: "SideWalks", 3: "Building", 4: "Wall",
    5: "Fence", 6: "Pole", 7: "TrafficLight", 8: "TrafficSign", 9: "Vegetation",
    10: "Terrain", 11: "Sky", 12: "Pedestrian", 13: "Rider", 14: "Car",
    15: "Truck", 16: "Bus", 17: "Train", 18: "Motorcycle", 19: "Bicycle",
    20: "Static", 21: "Dynamic", 22: "Other", 23: "Water", 24: "RoadLine",
    25: "Ground", 26: "Bridge", 27: "RailTrack", 28: "GuardRail"
}

# Path to the root directory of your current dataset 
# (It will automatically find all 'mask' folders inside)
DATASET_DIR = "./datasets/explicit_map_split" 

def main():
    print(f"=== Starting Ground Truth Pixel Analysis on: {DATASET_DIR} ===")
    
    # Collect all mask image paths
    mask_paths = []
    for root, dirs, files in os.walk(DATASET_DIR):
        # Only process files inside folders specifically named 'mask'
        if os.path.basename(root) == 'mask':
            for file in files:
                if file.endswith('.png'):
                    mask_paths.append(os.path.join(root, file))
                    
    if not mask_paths:
        print("Error: No mask images found. Please check the DATASET_DIR path.")
        return
        
    print(f"Found {len(mask_paths)} mask images. Scanning pixels...")

    # Initialize an array of length 256 to count all possible grayscale values (0-255).
    # Using uint64 to prevent integer overflow when dealing with millions of pixels.
    global_pixel_counts = np.zeros(256, dtype=np.int64)

    # Iterate through all images and count pixels 
    # (Using numpy.bincount on flattened arrays is extremely fast)
    for path in tqdm(mask_paths):
        mask_img = Image.open(path)
        mask_array = np.array(mask_img)
        
        # Flatten the 2D array to 1D and count occurrences of each ID
        counts = np.bincount(mask_array.flatten(), minlength=256)
        global_pixel_counts += counts

    print("\n" + "="*70)
    print(f"{'ID':<4} | {'CLASS NAME':<18} | {'PIXEL COUNT':<15} | {'STATUS'}")
    print("="*70)

    total_pixels_in_dataset = np.sum(global_pixel_counts)
    missing_classes = []
    rare_classes = []

    # Only analyze the predefined 29 classes (0-28)
    for class_id in range(29):
        class_name = CARLA_CLASSES.get(class_id, f"Unknown_{class_id}")
        count = global_pixel_counts[class_id]
        
        if count == 0:
            status = "❌ MISSING (0%)"
            missing_classes.append(class_name)
        else:
            percentage = (count / total_pixels_in_dataset) * 100
            # Classes making up less than 0.01% of total pixels are flagged as rare
            if percentage < 0.01: 
                status = f"⚠️ RARE ({percentage:.4f}%)"
                rare_classes.append(class_name)
            else:
                status = f"✅ PRESENT ({percentage:.2f}%)"
                
        print(f"{class_id:<4} | {class_name:<18} | {count:<15} | {status}")

    print("="*70)
    
    # Check for abnormally high pixel values (e.g., masks saved with 255 as an ignore label)
    abnormal_counts = np.sum(global_pixel_counts[29:])
    if abnormal_counts > 0:
        print(f"⚠️ WARNING: Found {abnormal_counts} pixels with IDs >= 29! These are invalid labels or ignore index markers.")

    print("\n📊 --- ANALYSIS SUMMARY ---")
    print(f"Total Missing Classes ({len(missing_classes)}): {missing_classes}")
    print(f"Total Rare Classes (<0.01%) ({len(rare_classes)}): {rare_classes}")
    print("Action Item: You should merge or ignore these Missing and Rare classes in your dataset.py lookup table!")

if __name__ == "__main__":
    main()