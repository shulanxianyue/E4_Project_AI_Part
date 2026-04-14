import os
import cv2
import torch
import numpy as np
from PIL import Image  # [NEW] Imported PIL to match dataset.py behavior

# --- ROS 2 Imports ---
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge

from torchvision import transforms
from model import get_carla_model

# ==========================================
# Configuration 
# ==========================================
MODEL_WEIGHTS = "best_carla_model_13classes_weighted.pth" 
NUM_CLASSES = 13

# Updated to listen to the fake camera for offline testing
INPUT_RGB_TOPIC = "/fake_camera/image_raw" 
# The topic nav2 group will listen to
OUTPUT_MASK_TOPIC = "/perception/semantic_mask"

# ==========================================
# AD-Focused Palette for 13 Classes
# ==========================================
SUPER_CLASS_COLORS = np.array([
    [0, 0, 0],       # 0: Unlabeled (Black)
    [128, 64, 128],  # 1: Road (Purple)
    [244, 35, 232],  # 2: Sidewalk (Pink)
    [157, 234, 50],  # 3: RoadLine (Bright Green)
    [0, 0, 142],     # 4: Vehicles (Dark Blue)
    [220, 20, 60],   # 5: Pedestrian (Crimson)
    [250, 170, 30],  # 6: TrafficLight (Orange)
    [220, 220, 0],   # 7: TrafficSign (Yellow)
    [153, 153, 153], # 8: Pole (Light Grey)
    [70, 70, 70],    # 9: Structures (Dark Grey)
    [107, 142, 35],  # 10: Nature/Terrain (Forest Green)
    [70, 130, 180],  # 11: Sky (Sky Blue)
    [110, 190, 160]  # 12: Obstacles/Misc (Teal)
], dtype=np.uint8)

class SemanticSegmentationNode(Node):
    def __init__(self):
        super().__init__('semantic_segmentation_node')
        
        # 1. Initialize CV Bridge (Translates between ROS Image and OpenCV)
        self.bridge = CvBridge()
        
        # 2. Setup Device & Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Initializing deep learning model on {self.device}...")
        
        self.model = get_carla_model(num_classes=NUM_CLASSES)
        if not os.path.exists(MODEL_WEIGHTS):
            self.get_logger().error(f"Weights {MODEL_WEIGHTS} not found!")
            raise FileNotFoundError
            
        self.model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Standard PyTorch normalization (Same as dataset.py)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 3. Create Subscriber (Listening to the camera stream)
        self.subscription = self.create_subscription(
            ROSImage,
            INPUT_RGB_TOPIC,
            self.image_callback,
            10 # Queue size
        )
        
        # 4. Create Publisher (Sending out the segmented mask)
        self.publisher = self.create_publisher(ROSImage, OUTPUT_MASK_TOPIC, 10)
        
        self.get_logger().info("Semantic Segmentation Node is READY and listening!")

    def decode_segmap(self, image):
        """Colorizes the 2D tensor of class IDs into an RGB visualization."""
        color_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        for class_id in range(NUM_CLASSES):
            color_mask[image == class_id] = SUPER_CLASS_COLORS[class_id]
        return color_mask

    def image_callback(self, msg):
        try:
            # Step A: Convert ROS Image message to OpenCV Image (BGR format)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Step B: Convert OpenCV BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Step C: Convert to PIL Image to strictly match dataset.py pipeline
            pil_image = Image.fromarray(rgb_image)
            
            # Step D: Apply transforms (ToTensor will scale PIL [0,255] to [0.0, 1.0] correctly)
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Step E: Neural Network Inference
            with torch.no_grad():
                output = self.model(input_tensor)['out']
                pred_mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
            
            # Step F: Colorize the output mask for visualization
            colorized_mask = self.decode_segmap(pred_mask)
            
            # Step G: Convert back to ROS Image message and publish
            ros_mask_msg = self.bridge.cv2_to_imgmsg(colorized_mask, encoding="rgb8")
            
            # Crucial: Keep the original timestamp and frame_id so nav2 can synchronize it
            ros_mask_msg.header = msg.header 
            
            self.publisher.publish(ros_mask_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SemanticSegmentationNode()
    try:
        rclpy.spin(node) # Keeps the node running, waiting for images
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()