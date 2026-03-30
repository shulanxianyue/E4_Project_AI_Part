import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CarlaSegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset for CARLA Semantic Segmentation.
    (Optimized: Using original image resolution, no resizing)
    """
    def __init__(self, root_dir, split='train'):
        """
        Args:
            root_dir (str): Root directory of the split dataset.
            split (str): 'train', 'val', or 'test'.
        """
        self.root_dir = root_dir
        self.split = split
        
        # Define paths for RGB and Mask directories
        self.rgb_dir = os.path.join(root_dir, split, 'rgb')
        self.mask_dir = os.path.join(root_dir, split, 'mask')
        
        # Get all image filenames (sorted to ensure matching)
        self.images = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith('.png')])
        
        # Standard PyTorch normalization for RGB images
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        """Returns the total number of samples in this split."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Loads and returns a single sample (RGB image and Mask) at the given index.
        """
        img_name = self.images[idx]
        
        # 1. Load RGB Image
        rgb_path = os.path.join(self.rgb_dir, img_name)
        # cv2 reads as BGR, we need to convert to RGB
        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Load Mask (Semantic Labels)
        mask_path = os.path.join(self.mask_dir, img_name)
        # Read as grayscale because pixel values ARE the class indices (0, 1, 2...)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 3. Convert to PyTorch Tensors (Directly, without resizing)
        # image becomes a FloatTensor [C, H, W]
        image = self.img_transform(image) 
        
        # mask becomes a LongTensor [H, W] (required for CrossEntropyLoss)
        mask = torch.from_numpy(mask).long()
        
        return image, mask

# ==========================================
# Test the Dataset and DataLoader
# ==========================================
if __name__ == "__main__":
    DATA_DIR = "./datasets/split_data_by_map"
    
    # Create the training dataset instance
    train_dataset = CarlaSegmentationDataset(root_dir=DATA_DIR, split='train')
    
    print(f"Total training samples: {len(train_dataset)}")
    
    # Create the DataLoader 
    # Notice: We keep batch_size=4 here just for testing, but in training, 
    # original resolution might require a smaller batch size to avoid OOM.
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # Fetch one batch to verify dimensions
    for images, masks in train_loader:
        print(f"Batch Images shape: {images.shape}") # Should now be [4, 3, 600, 800]
        print(f"Batch Masks shape: {masks.shape}")   # Should now be [4, 600, 800]
        print(f"Data types: Images -> {images.dtype}, Masks -> {masks.dtype}")
        
        unique_classes = torch.unique(masks)
        print(f"Unique class IDs in this batch: {unique_classes.tolist()}")
        break