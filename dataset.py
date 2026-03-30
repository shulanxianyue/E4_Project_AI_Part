import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CarlaSegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset for CARLA Semantic Segmentation.
    """
    def __init__(self, root_dir, split='train', img_size=(400, 300)):
        """
        Args:
            root_dir (str): Root directory of the split dataset.
            split (str): 'train', 'val', or 'test'.
            img_size (tuple): Target image size (width, height) to save VRAM.
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        
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
        
        # 3. Resize images (to save GPU memory)
        # RGB uses standard linear interpolation
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        # CRITICAL: Mask MUST use nearest neighbor to preserve exact class integer values!
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        # 4. Convert to PyTorch Tensors
        # image becomes a FloatTensor [C, H, W]
        image = self.img_transform(image) 
        
        # mask becomes a LongTensor [H, W] (required for CrossEntropyLoss)
        mask = torch.from_numpy(mask).long()
        
        return image, mask

# ==========================================
# Test the Dataset and DataLoader
# ==========================================
if __name__ == "__main__":
    # Define the path to your newly split dataset
    DATA_DIR = "./datasets/split_data_by_map"
    
    # Create the training dataset instance
    # Resizing from 800x600 to 400x300 to prevent Out-Of-Memory errors on laptop GPUs
    train_dataset = CarlaSegmentationDataset(root_dir=DATA_DIR, split='train', img_size=(400, 300))
    
    print(f"Total training samples: {len(train_dataset)}")
    
    # Create the DataLoader (Batching and multiprocessing)
    # batch_size=4 is a safe starting point for laptops
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # Fetch one batch to verify dimensions
    for images, masks in train_loader:
        print(f"Batch Images shape: {images.shape}") # Should be [4, 3, 300, 400]
        print(f"Batch Masks shape: {masks.shape}")   # Should be [4, 300, 400]
        print(f"Data types: Images -> {images.dtype}, Masks -> {masks.dtype}")
        
        # Check unique classes present in this batch's mask
        unique_classes = torch.unique(masks)
        print(f"Unique class IDs in this batch: {unique_classes.tolist()}")
        break # We only want to test the first batch