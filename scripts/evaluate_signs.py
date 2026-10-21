from pathlib import Path
from datetime import datetime
import csv
import cv2
import torch
from torch import nn
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
checkpoint = torch.load(
    ROOT / "models/sign_classifier.pt",
    map_location="cpu", weights_only=True
)
labels = checkpoint["labels"]
size = checkpoint["image_size"]

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 32, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(32, 3),
)
model.load_state_dict(checkpoint["model_state"])
model.eval()

session = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
output = ROOT / "evaluation" / session
output.mkdir(parents=True, exist_ok=True)
keys = {"g": "normal", "r": "inspect", "n": "background"}
results = []

class Camera(Node):
    def __init__(self):
        super().__init__("evaluate_signs")
        self.frame = None
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, "/patrol/camera", self.receive,
            qos_profile_sensor_data
        )

    def receive(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="bgr8"
        ).copy()

rclpy.init()
node = Camera()
print("g:緑を評価 / r:赤を評価 / n:背景を評価 / q:終了")
print("選ぶ範囲の正しい分類に対応するキーを押してください。")

try:
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.frame is None:
            continue
        cv2.imshow("Evaluate signs", node.frame)
        key = cv2.waitKey(10) & 0xff
        if key == ord("q"):
            break
        char = chr(key)
        if char not in keys:
            continue

        frame = node.frame.copy()
        x, y, w, h = cv2.selectROI(
            "Select test region", frame, True, False
        )
        cv2.destroyWindow("Select test region")
        if w < 8 or h < 8:
            continue

        crop = frame[y:y+h, x:x+w]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size))
        tensor = torch.from_numpy(rgb.copy()).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            scores = model(tensor).softmax(1)[0]
        index = int(scores.argmax())
        predicted = labels[index]
        truth = keys[char]
        correct = predicted == truth
        filename = f"{len(results)+1:03d}_{truth}.png"
        if not cv2.imwrite(str(output / filename), crop):
            raise RuntimeError("評価画像を保存できません")
        results.append([
            filename, truth, predicted, float(scores[index]), correct
        ])
        print(
            f"正解={truth} / AI判定={predicted} / "
            f"{'正解' if correct else '不正解'}",
            flush=True
        )
finally:
    with (output / "results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "truth", "prediction", "softmax_score", "correct"
        ])
        writer.writerows(results)
    print("\n今回の評価結果:")
    for label in labels:
        rows = [row for row in results if row[1] == label]
        ok = sum(row[4] for row in rows)
        print(f"  {label}: {ok}/{len(rows)}")
    print("保存先:", output)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()
