import os
import cv2
import glob
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge

class FakeCameraNode(Node):
    def __init__(self):
        super().__init__('fake_camera_node')
        
        # 1. 创建一个发布者，发布假摄像头的画面
        self.publisher_ = self.create_publisher(ROSImage, '/fake_camera/image_raw', 10)
        self.bridge = CvBridge()
        
        # 2. 读取你硬盘里已经有的测试图片
        self.image_dir = "./datasets/explicit_map_split/test/rgb"
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))
        self.current_idx = 0
        
        if not self.image_paths:
            self.get_logger().error(f"没有找到图片！请检查路径: {self.image_dir}")
            return
            
        # 3. 设置定时器，0.1秒触发一次（即 10 FPS）
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info(f"Fake Camera 启动！以 10Hz 循环发布 {len(self.image_paths)} 张图片...")

    def timer_callback(self):
        if not self.image_paths:
            return
            
        img_path = self.image_paths[self.current_idx]
        cv_image = cv2.imread(img_path) # OpenCV 默认读取为 BGR 格式
        
        if cv_image is not None:
            # 关键：把 OpenCV 格式转换为 ROS 的 sensor_msgs/Image
            msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            # 加上时间戳和坐标系名称（下游 Nav2 会用到）
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_link"
            
            self.publisher_.publish(msg)
            
        # 循环播放
        self.current_idx = (self.current_idx + 1) % len(self.image_paths)

def main(args=None):
    rclpy.init(args=args)
    node = FakeCameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()