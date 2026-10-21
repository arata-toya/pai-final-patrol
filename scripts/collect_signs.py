from pathlib import Path
from datetime import datetime
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

root = Path(__file__).resolve().parents[1] / "data"
labels = {"g": "normal", "r": "inspect", "n": "background"}
for label in labels.values():
    (root / label).mkdir(parents=True, exist_ok=True)

class Collector(Node):
    def __init__(self):
        super().__init__("sign_collector")
        self.bridge = CvBridge()
        self.frame = None
        self.subscription = self.create_subscription(
            Image, "/patrol/camera", self.receive,
            qos_profile_sensor_data
        )

    def receive(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="bgr8"
        ).copy()

rclpy.init()
node = Collector()
print("画像待機中。Gazeboとカメラのブリッジを起動してください。")
print("画像画面で g:緑、r:赤、n:背景、q:終了")

try:
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.frame is None:
            continue

        cv2.imshow("Sign collector", node.frame)
        key = cv2.waitKey(10) & 0xff

        if key == ord("q"):
            break

        char = chr(key)
        if char not in labels:
            continue

        frame = node.frame.copy()
        print("対象をドラッグで囲み、Enter。取消は c。")
        x, y, w, h = cv2.selectROI(
            "Select region", frame,
            showCrosshair=True, fromCenter=False
        )
        cv2.destroyWindow("Select region")

        if w < 8 or h < 8:
            print("保存しませんでした。範囲を少し大きく選んでください。")
            continue

        crop = frame[y:y+h, x:x+w]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = root / labels[char] / f"{stamp}.png"
        if not cv2.imwrite(str(path), crop):
            raise RuntimeError(f"保存失敗: {path}")
        print("保存しました:", path)
finally:
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()
