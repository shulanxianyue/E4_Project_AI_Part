import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

def get_carla_model(num_classes=23):
    """
    Creates a DeepLabV3 model with a ResNet50 backbone, 
    adapted for the specific number of classes in CARLA.
    """
    # 1. Load the pre-trained DeepLabV3 model (pretrained on standard datasets like COCO/Cityscapes)
    # Using 'DEFAULT' loads the most up-to-to-date best weights
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)
    
    # 2. Modify the classifier head
    # The original model outputs 21 classes (COCO dataset). 
    # CARLA typically has up to 23 semantic tags (0 to 22).
    # We need to replace the final convolutional layer to match our num_classes.
    
    # model.classifier[4] is the final output layer of DeepLabV3
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    # DeepLabV3 also has an 'auxiliary classifier' used during training to help gradients flow.
    # We need to change its output layer too.
    if model.aux_classifier is not None:
        in_channels_aux = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
    return model

# ==========================================
# Test the Model architecture
# ==========================================
if __name__ == "__main__":
    # CARLA has 23 standard semantic tags (0: Unlabeled, 1: Building, ..., 4: Pedestrian, 10: Vehicles, etc.)
    NUM_CLASSES = 23 
    
    print("Building the DeepLabV3 model...")
    model = get_carla_model(num_classes=NUM_CLASSES)
    print("Model built successfully!")
    
    # Create a "dummy" batch of images to test if the model processes them correctly
    # Shape: [Batch_size=4, Channels=3, Height=300, Width=400]
    dummy_input = torch.randn(4, 3, 300, 400)
    
    print("\nFeeding dummy data to the model...")
    # Put the model in evaluation mode for testing
    model.eval() 
    with torch.no_grad():
        output = model(dummy_input)
    
    # DeepLabV3 returns a dictionary. The final prediction is stored under the key 'out'
    predictions = output['out']
    
    print(f"Input shape:  {dummy_input.shape}")
    # Expected Output shape: [4, 23, 300, 400] -> [Batch, Classes, Height, Width]
    print(f"Output shape: {predictions.shape}") 
    print("Everything is working perfectly!")