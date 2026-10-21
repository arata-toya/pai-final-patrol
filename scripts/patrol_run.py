from pathlib import Path
from datetime import datetime
import csv
import math
import time
import cv2
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from detect_signs import Camera, detect

ROOT = Path(__file__).resolve().parents[1]
session = datetime.now().strftime("%Y%m%d_%H%M%S")
output = ROOT / "runs" / session
output.mkdir(parents=True, exist_ok=True)

class Patrol(Camera):
    def __init__(self):
        self.last_image = 0.0
        self.last_odom = 0.0
        self.x = None
        self.speed = float("inf")
        self.turn_speed = float("inf")
        self.odom_seq = 0
        super().__init__()
        self.velocity = self.create_publisher(
            Twist, "/model/vehicle_blue/cmd_vel", 10
        )
        self.odom_subscription = self.create_subscription(
            Odometry, "/model/vehicle_blue/odometry",
            self.receive_odom, 10
        )

    def receive(self, msg):
        super().receive(msg)
        self.last_image = time.monotonic()

    def receive_odom(self, msg):
        self.x = msg.pose.pose.position.x
        velocity = msg.twist.twist
        self.speed = math.hypot(velocity.linear.x, velocity.linear.y)
        self.turn_speed = abs(velocity.angular.z)
        self.odom_seq += 1
        self.last_odom = time.monotonic()

    def command(self, speed=0.0):
        msg = Twist()
        msg.linear.x = speed
        self.velocity.publish(msg)

rclpy.init()
node = Patrol()
state = "READY"
inspected = False
green_seen = False
red_count = 0
pause_until = 0.0
next_detection = 0.0
display = None
found = []
events = []
stopped_samples = 0
last_stop_seq = -1
stop_requested_at = 0.0

def record(event):
    x = node.x if node.x is not None else ""
    events.append([datetime.now().isoformat(), x, event])
    print(f"x={x} : {event}", flush=True)
    with (output / "events.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "odometry_x_m", "event"])
        writer.writerows(events)

print("画面をクリックして s:開始 / q:終了")
print("開始位置は x=0 付近にしてください。")
print("点検区間: x>=5m / 終了位置: x>=6.5m")

cv2.namedWindow("AI patrol", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI patrol", 640, 480)

try:
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.03)
        now = time.monotonic()

        fresh = (
            node.frame is not None and node.x is not None
            and now - node.last_image < 5
            and now - node.last_odom < 5
        )

        if state in ("RUN", "BRAKING", "INSPECT", "EXIT", "FINAL_STOP") and not fresh:
            node.command()
            state = "FAULT"
            record("Sensor timeout: stopped")

        if node.frame is not None and now >= next_detection:
            display, found = detect(node.frame.copy())
            next_detection = time.monotonic() + 0.5

            if state == "RUN":
                if "normal" in found and not green_seen:
                    green_seen = True
                    record("AI: normal sign recognized")

                if node.x >= 5.0 and "inspect" in found:
                    red_count += 1
                else:
                    red_count = 0

                if red_count >= 2:
                    node.command()
                    state = "BRAKING"
                    stopped_samples = 0
                    last_stop_seq = node.odom_seq
                    stop_requested_at = time.monotonic()
                    record("AI: inspect sign recognized; braking")

        if state in ("RUN", "EXIT") and node.x >= 6.5:
            node.command()
            state = "FINAL_STOP"
            stopped_samples = 0
            last_stop_seq = node.odom_seq
            stop_requested_at = time.monotonic()
            record("End position reached; braking")

        if state in ("BRAKING", "FINAL_STOP"):
            node.command()

            # 同じ位置情報を重複して数えず、新しい受信ごとに確認。
            if node.odom_seq != last_stop_seq:
                last_stop_seq = node.odom_seq
                stationary = (
                    node.speed < 0.01 and node.turn_speed < 0.02
                )
                stopped_samples = stopped_samples + 1 if stationary else 0

            if stopped_samples >= 2:
                if state == "BRAKING":
                    if not cv2.imwrite(
                        str(output / "inspection.png"), node.frame
                    ):
                        raise RuntimeError("点検画像の保存に失敗")
                    inspected = True
                    state = "INSPECT"
                    pause_until = time.monotonic() + 3
                    record("Stop confirmed; image saved; inspection pause")
                else:
                    state = "DONE"
                    record(
                        "Stop confirmed: finished after inspection"
                        if inspected else
                        "Stop confirmed: finished WITHOUT inspection"
                    )
            elif time.monotonic() - stop_requested_at > 20:
                state = "FAULT"
                record("Stop confirmation timeout; keep stop command")

        if state == "INSPECT" and time.monotonic() >= pause_until:
            state = "EXIT"
            record("Inspection pause finished; resume")

        speed = 0.2 if state in ("RUN", "EXIT") else 0.0
        node.command(speed)

        if display is not None:
            view = display.copy()
            position = f"{node.x:.2f}" if node.x is not None else "--"
            cv2.putText(
                view, f"{state}  x={position}m", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
            cv2.putText(
                view, "s: start   q: quit", (5, view.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
            )
            cv2.imshow("AI patrol", view)

        key = cv2.waitKey(10) & 0xff
        if key == ord("q"):
            break
        if key == ord("s") and state == "READY":
            if not fresh:
                print("カメラ画像と走行位置の受信を確認してください。")
            elif abs(node.x) > 0.5:
                print("Gazeboを起動し直し、開始位置に戻してください。")
            elif node.velocity.get_subscription_count() == 0:
                print("速度指令のブリッジを確認してください。")
            else:
                state = "RUN"
                record("Patrol started")
finally:
    for _ in range(10):
        node.command()
        rclpy.spin_once(node, timeout_sec=0.1)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()
    print("停止指令を送りました。記録先:", output)
