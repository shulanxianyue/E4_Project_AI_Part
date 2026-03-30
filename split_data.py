import os
import shutil

# 1. 配置路径
SOURCE_DIR = "./datasets/carla_data"          # 原始数据目录
TARGET_DIR = "./datasets/split_data_by_map"   # 按地图划分后输出的新目录

# 2. 定义划分策略 (你可以根据需要修改这里的列表)
# 确保列表中的名字与你 carla_data 里的文件夹名字能够匹配
SPLIT_MAP = {
    'train': ['Town01', 'Town03', 'Town04', 'Town05', 'Town06'],
    'val': ['Town07'],
    'test': ['Town10HD']
}

def create_directories():
    """在目标文件夹中创建标准的深度学习数据集结构"""
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(TARGET_DIR, split, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(TARGET_DIR, split, 'mask'), exist_ok=True)
    print(f"Created target directories in {TARGET_DIR}")

def main():
    create_directories()
    
    # 获取所有的 Town 文件夹
    available_towns = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    total_processed = 0
    
    for town in available_towns:
        # 确定这个地图属于哪个集合
        target_split = None
        for split_name, towns_in_split in SPLIT_MAP.items():
            # 使用 in 进行模糊匹配 (解决 Town10 和 Town10HD 命名差异)
            if any(t in town for t in towns_in_split):
                target_split = split_name
                break
        
        if target_split is None:
            print(f"  Warning: 文件夹 {town} 不在划分策略中，将被跳过。")
            continue

        print(f"\nProcessing {town} -> 分配至 [{target_split.upper()}] 集...")
        
        rgb_dir = os.path.join(SOURCE_DIR, town, "rgb")
        mask_dir = os.path.join(SOURCE_DIR, town, "mask")
        
        if not os.path.exists(rgb_dir) or not os.path.exists(mask_dir):
            print(f"  Warning: {town} 缺少 rgb 或 mask 文件夹。跳过。")
            continue
            
        images = [f for f in os.listdir(rgb_dir) if f.endswith('.png')]
        
        for img_name in images:
            # 构建带有地图前缀的新文件名，防止如果以后要合并时发生覆盖
            new_name = f"{town}_{img_name}"
            
            # 源文件路径
            src_rgb = os.path.join(rgb_dir, img_name)
            src_mask = os.path.join(mask_dir, img_name)
            
            # 目标文件路径
            dst_rgb = os.path.join(TARGET_DIR, target_split, 'rgb', new_name)
            dst_mask = os.path.join(TARGET_DIR, target_split, 'mask', new_name)
            
            # 复制文件
            shutil.copy2(src_rgb, dst_rgb)
            shutil.copy2(src_mask, dst_mask)
            
        print(f"  -> 成功复制 {len(images)} 张图片到 {target_split}")
        total_processed += len(images)

    print(f"\n=== 数据集按地图划分完成！ ===")
    print(f"总共处理了: {total_processed} 张图片")
    print(f"数据已准备好，存放于: {TARGET_DIR}")

if __name__ == "__main__":
    main()