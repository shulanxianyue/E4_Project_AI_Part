import os
import shutil
from collections import defaultdict

# ==========================================
# Configuration for Explicit Map Split
# ==========================================
# Source directory (your existing data pool)
RAW_DATA_DIR = "./datasets/split_data_by_map" 
# New target directory for this specific experiment
TARGET_DIR = "./datasets/explicit_map_split"

# Explicitly define which map goes to which split
TRAIN_TOWNS = ['Town01', 'Town03', 'Town06', 'Town07', 'Town10HD'] 
VAL_TOWNS   = ['Town04']
TEST_TOWNS  = ['Town05']

def create_dirs(base_path):
    """Creates train, val, test directories for rgb and mask."""
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_path, split, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(base_path, split, 'mask'), exist_ok=True)

def main():
    print("=== Starting Explicit Map-Based Dataset Resplit ===")
    
    map_dict = defaultdict(list)
    total_found = 0

    # 1. Collect ALL files from the existing folders
    for split_folder in ['train', 'val', 'test']:
        rgb_dir = os.path.join(RAW_DATA_DIR, split_folder, 'rgb')
        mask_dir = os.path.join(RAW_DATA_DIR, split_folder, 'mask')
        
        if not os.path.exists(rgb_dir):
            continue
            
        images = [f for f in os.listdir(rgb_dir) if f.endswith('.png')]
        
        for img_name in images:
            src_rgb = os.path.join(rgb_dir, img_name)
            src_mask = os.path.join(mask_dir, img_name)
            
            # Extract map name (e.g., "Town10HD" or "Town01")
            map_name = img_name.split('_')[0] 
            
            map_dict[map_name].append({
                'filename': img_name,
                'src_rgb': src_rgb,
                'src_mask': src_mask
            })
            total_found += 1

    if total_found == 0:
        print(f"Error: No images found in {RAW_DATA_DIR}.")
        return

    print(f"\nFound {total_found} total images. Current maps in your data:")
    for m, files in map_dict.items():
        print(f"  - {m}: {len(files)} images")

    # 2. Setup target directories
    create_dirs(TARGET_DIR)

    # 3. Distribute files based on strict rules
    print("\nDistributing files based on Explicit Rules...")
    
    stats = {'train': 0, 'val': 0, 'test': 0, 'ignored': 0}

    for map_name, file_dicts in map_dict.items():
        # Determine the target split for this specific map
        if map_name in TRAIN_TOWNS:
            target_split = 'train'
        elif map_name in VAL_TOWNS:
            target_split = 'val'
        elif map_name in TEST_TOWNS:
            target_split = 'test'
        else:
            # If a map is not in any list (e.g., Town02 or Town05), skip it or assign it
            print(f"  [Warning] {map_name} is not in any list. Ignoring its {len(file_dicts)} images.")
            stats['ignored'] += len(file_dicts)
            continue
            
        # Copy the files to their designated split
        for item in file_dicts:
            dst_rgb = os.path.join(TARGET_DIR, target_split, 'rgb', item['filename'])
            dst_mask = os.path.join(TARGET_DIR, target_split, 'mask', item['filename'])
            
            shutil.copy(item['src_rgb'], dst_rgb)
            shutil.copy(item['src_mask'], dst_mask)
            
        stats[target_split] += len(file_dicts)
        print(f"  -> Routed {map_name} completely to [{target_split.upper()}] ({len(file_dicts)} images)")

    print(f"\n=== Split Complete! ===")
    print(f"Summary -> Train: {stats['train']} | Val: {stats['val']} | Test: {stats['test']} | Ignored: {stats['ignored']}")
    print(f"New dataset is ready at: {TARGET_DIR}")

if __name__ == "__main__":
    main()