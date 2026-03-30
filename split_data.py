import os
import shutil

# 1. path
SOURCE_DIR = "./datasets/carla_data"          # original datas
TARGET_DIR = "./datasets/split_data_by_map"   # new

# 2. define the split strategy
SPLIT_MAP = {
    'train': ['Town01', 'Town03', 'Town04', 'Town05', 'Town06'],
    'val': ['Town07'],
    'test': ['Town10HD']
}

def create_directories():
    """Create a standard deep learning dataset structure in the target folder"""
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(TARGET_DIR, split, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(TARGET_DIR, split, 'mask'), exist_ok=True)
    print(f"Created target directories in {TARGET_DIR}")

def main():
    create_directories()
    
    # get all the towns
    available_towns = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    total_processed = 0
    
    for town in available_towns:
        target_split = None
        for split_name, towns_in_split in SPLIT_MAP.items():
            # Fuzzy matching
            if any(t in town for t in towns_in_split):
                target_split = split_name
                break
        
        if target_split is None:
            print(f"  Warning: The folder {town} is not in the partitioning strategy and will be skipped.")
            continue

        print(f"\nProcessing {town} -> assign to [{target_split.upper()}] set...")
        
        rgb_dir = os.path.join(SOURCE_DIR, town, "rgb")
        mask_dir = os.path.join(SOURCE_DIR, town, "mask")
        
        if not os.path.exists(rgb_dir) or not os.path.exists(mask_dir):
            print(f"  Warning: {town} lack of rgb or mask folder. Pass.")
            continue
            
        images = [f for f in os.listdir(rgb_dir) if f.endswith('.png')]
        
        for img_name in images:
            new_name = f"{town}_{img_name}"
            
            # source
            src_rgb = os.path.join(rgb_dir, img_name)
            src_mask = os.path.join(mask_dir, img_name)
            
            # destination
            dst_rgb = os.path.join(TARGET_DIR, target_split, 'rgb', new_name)
            dst_mask = os.path.join(TARGET_DIR, target_split, 'mask', new_name)
            
            # copy
            shutil.copy2(src_rgb, dst_rgb)
            shutil.copy2(src_mask, dst_mask)
            
        print(f"  -> succeeded in copying {len(images)} images to {target_split}")
        total_processed += len(images)

    print(f"\n=== dataset split by map completed！ ===")
    print(f"Total: {total_processed} images")
    print(f"stored in : {TARGET_DIR}")

if __name__ == "__main__":
    main()