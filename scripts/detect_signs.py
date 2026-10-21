from pathlib import Path
import time
import cv2
import numpy as np
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

class Camera(Node):
    def __init__(self):
        super().__init__("detect_signs")
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

def detect(frame):
    height, width = frame.shape[:2]
    boxes, patches = [], []

    # 大きさを変えた窓で画像を切り出します。
    for window in [32, 64, 96]:
        if window > min(height, width):
            continue
        step = window // 2
        xs = sorted(set(
            list(range(0, width-window+1, step)) + [width-window]
        ))
        ys = sorted(set(
            list(range(0, height-window+1, step)) + [height-window]
        ))
        for y in ys:
            for x in xs:
                patch = frame[y:y+window, x:x+window]
                patch = cv2.resize(patch, (size, size))
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patches.append(patch)
                boxes.append((x, y, window, window))

    if not patches:
        return frame.copy(), []

    scores = []
    with torch.no_grad():
        for start in range(0, len(patches), 32):
            batch = np.stack(patches[start:start+32])
            tensor = torch.from_numpy(batch).float()
            tensor = tensor.permute(0, 3, 1, 2) / 255.0
            scores.append(model(tensor).softmax(1).numpy())
    scores = np.concatenate(scores)
    winners = scores.argmax(axis=1)

    display = frame.copy()
    found = []
    colors = {"normal": (0, 220, 0), "inspect": (0, 0, 255)}
    for label in ["normal", "inspect"]:
        index = labels.index(label)
        candidates = np.where(
            (winners == index) & (scores[:, index] >= 0.98)
        )[0]
        if len(candidates) == 0:
            continue
        best = candidates[np.argmax(scores[candidates, index])]
        x, y, w, h = boxes[best]
        score = float(scores[best, index])
        cv2.rectangle(display, (x, y), (x+w, y+h), colors[label], 2)
        text_y = max(15, y-5)
        cv2.putText(
            display, f"{label} {score:.3f}", (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[label], 1
        )
        found.append(label)

    return display, found

if __name__ == "__main__":
    rclpy.init()
    node = Camera()
    next_update = 0.0
    display = None
    previous = None
    print("自動認識のみ。ロボットへの走行指令は送りません。")
    print("画像ウィンドウで q を押すと終了します。")
    
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if node.frame is None:
                continue
            if time.monotonic() >= next_update:
                display, found = detect(node.frame.copy())
                status = ", ".join(found) if found else "no sign"
                if status != previous:
                    print("認識:", status, flush=True)
                    previous = status
                next_update = time.monotonic() + 0.5
            if display is not None:
                cv2.imshow("AI sign detection", display)
            if cv2.waitKey(10) & 0xff == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
