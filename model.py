import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

# [MODIFIED] Default num_classes changed to 13
def get_carla_model(num_classes=13):
    """
    Creates a DeepLabV3 model with a ResNet101 backbone for higher accuracy, 
    adapted for the specific number of classes in CARLA.
    """
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights)
    
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    if model.aux_classifier is not None:
        in_channels_aux = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
    return model

# ==========================================
# Test the Model architecture
# ==========================================
if __name__ == "__main__":
    # [MODIFIED] Updated to 13
    NUM_CLASSES = 13 
    
    print("Building the DeepLabV3-ResNet101 model...")
    model = get_carla_model(num_classes=NUM_CLASSES)
    print("Model built successfully!")
    
    dummy_input = torch.randn(2, 3, 600, 800)
    
    print("\nFeeding dummy high-res data to the model...")
    model.eval() 
    with torch.no_grad():
        output = model(dummy_input)
    
    predictions = output['out']
    
    print(f"Input shape:  {dummy_input.shape}")
    # Expected Output shape: [2, 13, 600, 800] -> [Batch, Classes, Height, Width]
    print(f"Output shape: {predictions.shape}") 
    print("Everything is working perfectly!")