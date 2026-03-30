import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

def get_carla_model(num_classes=29):
    """
    Creates a DeepLabV3 model with a ResNet101 backbone for higher accuracy, 
    adapted for the specific number of classes in CARLA.
    """
    # 1. Load the pre-trained DeepLabV3 model with a deeper ResNet101 backbone
    # Using 'DEFAULT' loads the state-of-the-art pre-trained weights
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights)
    
    # 2. Modify the main classifier head
    # model.classifier[4] is the final convolutional layer for DeepLabV3
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    # 3. Modify the auxiliary classifier head (helps with gradient flow during training)
    if model.aux_classifier is not None:
        in_channels_aux = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
    return model

# ==========================================
# Test the Model architecture
# ==========================================
if __name__ == "__main__":
    # Updated for CARLA 0.9.16 with 29 standard semantic classes
    NUM_CLASSES = 29 
    
    print("Building the DeepLabV3-ResNet101 model...")
    model = get_carla_model(num_classes=NUM_CLASSES)
    print("Model built successfully!")
    
    # Simulate high-resolution input from CARLA dataset
    # Shape: [Batch_size=2, Channels=3, Height=600, Width=800]
    # Note: Batch size is reduced to 2 to prevent Out-Of-Memory errors on high-res images
    dummy_input = torch.randn(2, 3, 600, 800)
    
    print("\nFeeding dummy high-res data to the model...")
    # Put the model in evaluation mode for testing
    model.eval() 
    with torch.no_grad():
        output = model(dummy_input)
    
    # DeepLabV3 returns a dictionary. The final prediction is stored under the key 'out'
    predictions = output['out']
    
    print(f"Input shape:  {dummy_input.shape}")
    # Expected Output shape: [2, 29, 600, 800] -> [Batch, Classes, Height, Width]
    print(f"Output shape: {predictions.shape}") 
    print("Everything is working perfectly!")