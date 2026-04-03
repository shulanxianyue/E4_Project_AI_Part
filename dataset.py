import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# [ULTIMATE] Super-Class Merging Table (13 Classes)
# ==========================================
# 0: Unlabeled
# 1: Flat Ground (Road, Sidewalk, Terrain, RoadLine, Ground)
# 2: Structures (Building, Wall, Fence, Bridge, GuardRail, RailTrack)
# 3: Pole
# 4: TrafficLight
# 5: TrafficSign
# 6: Vegetation
# 7: Sky
# 8: Pedestrian
# 9: Vehicles (Car, Truck, Bus, Train)
# 10: Two-Wheelers (Motorcycle, Bicycle)
# 11: Obstacles (Static, Dynamic, Other)
# 12: Water

LABEL_MAPPING = np.full(29, 255, dtype=np.uint8)

# 0: Unlabeled 
LABEL_MAPPING[0] = 0

# 1: Flat Ground 
LABEL_MAPPING[[1, 2, 10, 24, 25]] = 1

# 2: Structures 
LABEL_MAPPING[[3, 4, 5, 26, 27, 28]] = 2

# Core individual classes
LABEL_MAPPING[6] = 3   # Pole
LABEL_MAPPING[7] = 4   # TrafficLight
LABEL_MAPPING[8] = 5   # TrafficSign
LABEL_MAPPING[9] = 6   # Vegetation
LABEL_MAPPING[11] = 7  # Sky
LABEL_MAPPING[12] = 8  # Pedestrian

# 9: Vehicles
LABEL_MAPPING[[14, 15, 16, 17]] = 9

# 10: Two-Wheelers
LABEL_MAPPING[[18, 19]] = 10

# 11: Obstacles
LABEL_MAPPING[[20, 21, 22]] = 11

# 12: Water
LABEL_MAPPING[23] = 12

# 13 (Rider) is left as 255 (Ignored)

class CarlaSegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset for CARLA Semantic Segmentation.
    (Optimized: Using original image resolution, Data Augmentation added)
    """
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        
        self.rgb_dir = os.path.join(root_dir, split, 'rgb')
        self.mask_dir = os.path.join(root_dir, split, 'mask')
        
        self.images = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith('.png')])
        
        # Standard PyTorch normalization for RGB images
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Define Color Jitter for data augmentation (brightness, contrast, saturation, hue)
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        # 1. Load RGB Image
        rgb_path = os.path.join(self.rgb_dir, img_name)
        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Load Mask and apply mapping immediately (0-28 to 0-26)
        mask_path = os.path.join(self.mask_dir, img_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = LABEL_MAPPING[mask]
        
        # Convert numpy arrays to PIL Images for torchvision.functional operations
        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask)
        
        # ==========================================
        # Data Augmentation (TRAINING SET ONLY)
        # Prevents the model from overfitting to static scenes
        # ==========================================
        if self.split == 'train':
            # Augmentation 1: 50% chance of horizontal flip (Image and Mask MUST flip together)
            if random.random() > 0.5:
                image_pil = TF.hflip(image_pil)
                mask_pil = TF.hflip(mask_pil)
            
            # Augmentation 2: Color Jitter (Apply to RGB ONLY, never to the Mask!)
            image_pil = self.color_jitter(image_pil)
        
        # 3. Convert back to PyTorch Tensors
        image = self.img_transform(image_pil) 
        
        # Convert PIL mask back to numpy, then to tensor
        mask_array = np.array(mask_pil)
        mask = torch.from_numpy(mask_array).long()
        
        return image, mask