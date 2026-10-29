#!/usr/bin/env python3
"""発展版のROS 2ノード。コースの正解標識座標は読み込まない。"""
from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from vision import WallVision, annotate, normalize_gray
from control import PatrolControl, Telemetry

ROOT=Path(__file__).resolve().parents[1]


class PatrolNode(Node):
    def __init__(self,args):
        super().__init__('pai_wall_patrol')
        self.args=args
        self.vision=WallVision()
        self.control=PatrolControl(stop_m=args.stop_m,end_x=args.end_x,capture_only=args.capture_only)
        self.origin=time.monotonic()
        self.frame=None
        self.frame_time=-100.0
        self.frame_seq=0
        self.processed_seq=-1
        self.odom_time=-100.0
        self.telemetry=Telemetry(0,float('inf'),float('inf'),0)
        self.bridge=CvBridge()
        self.detections=[]
        self.last_command=0.0
        self.saved_inspections=0
        self.written_events=0
        self.last_snapshot=0.0
        self.started=False
        self.latest_crack=None
        self.latest_crack_time=None
        self.report_items=[]
        self.latest_candidate=None
        self.latest_candidate_time=None
        self.diagnostic_saved=False
        self.last_infer_ms=0.0
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.output=ROOT/'runs_v3'/(f'{stamp}_{args.scenario}_'+('capture' if args.capture_only else 'cnn'))
        self.output.mkdir(parents=True)
        import hashlib
        model_path=ROOT/'models/wall_classifier.pt'
        self.runtime_model=dict(sha256=hashlib.sha256(model_path.read_bytes()).hexdigest()
                                if model_path.exists() else 'unknown')
        card_path=ROOT/'models/camera_model_card.json'
        if card_path.exists():
            card=json.loads(card_path.read_text())
            if card.get('sha256')==self.runtime_model['sha256']:
                self.runtime_model.update(card)
        (self.output/'runtime_model.json').write_text(json.dumps(self.runtime_model,indent=2))
        qos=QoSProfile(history=HistoryPolicy.KEEP_LAST,depth=1,
                       reliability=ReliabilityPolicy.BEST_EFFORT)
        self.image_sub=self.create_subscription(Image,'/wall/camera',self.on_image,qos)
        self.odom_sub=self.create_subscription(Odometry,'/model/vehicle_blue/odometry',self.on_odom,10)
        self.pub=self.create_publisher(Twist,'/model/vehicle_blue/cmd_vel',10)
        self.trace_file=(self.output/'trace.csv').open('w',newline='')
        self.perception_file=(self.output/'perception.jsonl').open('w',encoding='utf-8')
        self.trace=csv.writer(self.trace_file)
        self.trace.writerow(['elapsed_s','frame_seq','state','odom_x_m','speed_mps',
                             'command_mps','red_range_m','inference_ms'])
        if not args.headless:
            cv2.namedWindow('AI wall patrol V3',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('AI wall patrol V3',960,640)
        print('AI wall patrol V3: s=start / q=stop and quit',flush=True)
        print(f'scenario={args.scenario}, mode={args.mode}, image stop gap={args.stop_m}m',flush=True)
        print('ひび割れの停止判断に既知の異常位置は使いません。',flush=True)
        print('CNN model:',self.runtime_model.get('version','unidentified'),flush=True)
        print('保存先:',self.output,flush=True)
        if args.capture_only:
            print('CAPTURE ONLY: 共通枠で停止して撮影します。AIによるひび割れ検出の成功には数えません。',flush=True)

    def elapsed(self):
        return time.monotonic()-self.origin

    def on_image(self,msg):
        self.frame=self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8').copy()
        self.frame_time=self.elapsed()
        self.frame_seq+=1

    def on_odom(self,msg):
        v=msg.twist.twist
        self.telemetry=Telemetry(msg.pose.pose.position.x,
                                 float(np.hypot(v.linear.x,v.linear.y)),
                                 abs(v.angular.z),self.telemetry.sequence+1)
        self.odom_time=self.elapsed()

    def command(self,speed):
        msg=Twist();msg.linear.x=float(speed)
        self.pub.publish(msg)
        self.last_command=float(speed)

    def flush_events(self):
        for event in self.control.events[self.written_events:]:
            print(f"{event['event']} x={event['x_m']:.3f}m",flush=True)
        if len(self.control.events)!=self.written_events:
            (self.output/'events.json').write_text(json.dumps(
                self.control.events,ensure_ascii=False,indent=2))
            self.written_events=len(self.control.events)

    def start_if_ready(self):
        now=self.elapsed()
        fresh=now-self.frame_time<2 and now-self.odom_time<2
        if not fresh:
            print('画像・位置情報を待っています。',flush=True);return
        if self.pub.get_subscription_count()==0:
            print('速度指令のブリッジを確認してください。',flush=True);return
        if self.control.start(now,self.telemetry):
            self.started=True
        else:
            print('開始位置 x=0 付近へ戻してください。',flush=True)

    def step(self):
        now=self.elapsed()
        fresh=now-self.frame_time<2 and now-self.odom_time<2
        new_detections=None
        if self.frame is not None and self.frame_seq!=self.processed_seq:
            before=time.monotonic()
            self.detections=self.vision.detect(self.frame)
            self.last_infer_ms=(time.monotonic()-before)*1000
            self.perception_file.write(json.dumps(dict(elapsed_s=now,frame=self.frame_seq,
                detections=[dict(label=d.label,score=d.score,bbox=d.bbox,forward_m=d.forward_m,
                                 probabilities=d.probabilities) for d in self.detections]),ensure_ascii=False)+'\n')
            self.perception_file.flush()
            candidates=[d for d in self.detections if not any(
                abs(self.telemetry.x+self.control.camera_forward+d.forward_m-v)<1.0
                for v in self.control.visited_world_x)]
            if candidates:
                self.latest_candidate=min(candidates,key=lambda d:d.forward_m)
                self.latest_candidate_time=now
            cracks=[d for d in self.detections if d.label=='crack' and not any(
                abs(self.telemetry.x+self.control.camera_forward+d.forward_m-v)<1.0
                for v in self.control.visited_world_x)]
            if cracks:
                self.latest_crack=min(cracks,key=lambda d:d.forward_m)
                self.latest_crack_time=now
            self.processed_seq=self.frame_seq
            new_detections=self.detections
        speed=self.control.tick(now,self.telemetry,new_detections,
                                self.processed_seq,fresh=fresh)
        self.command(speed)
        if len(self.control.inspections)>self.saved_inspections:
            item=self.control.inspections[-1]
            stem=f"inspection_{item['index']:02d}"
            if not cv2.imwrite(str(self.output/f'{stem}.png'),self.frame):
                raise RuntimeError('点検画像を保存できません')
            cv2.imwrite(str(self.output/f'{stem}_annotated.png'),annotate(self.frame,self.detections))
            (self.output/f'{stem}.json').write_text(json.dumps(item,indent=2))
            detection=self.latest_candidate if self.args.capture_only else self.latest_crack
            detail=dict(**item, classification=(detection.label if detection is not None else 'unknown') if self.args.capture_only else 'crack',
                        capture_only=self.args.capture_only,model_sha256=self.runtime_model['sha256'],
                        confidence=None if detection is None else detection.score,
                        probabilities=None if detection is None else detection.probabilities,
                        crop_age_s=None if detection is None else now-(self.latest_candidate_time if self.args.capture_only else self.latest_crack_time),
                        image=stem+'.png',annotated_image=stem+'_annotated.png',
                        note='Synthetic wall experiment; not a structural diagnosis')
            if detection is not None:
                cv2.imwrite(str(self.output/f'{stem}_crop.png'),detection.crop)
                prepared=normalize_gray(cv2.resize(detection.crop,(96,96)))
                cv2.imwrite(str(self.output/f'{stem}_input.png'),prepared)
            (self.output/f'{stem}.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2))
            self.report_items.append(detail)
            self.write_report()
            self.saved_inspections=len(self.control.inspections)
        if self.control.state in ('FAULT','REVIEW_BRAKING','REVIEW_REQUIRED') and not self.diagnostic_saved:
            if self.frame is not None:
                cv2.imwrite(str(self.output/'diagnostic_camera.png'),self.frame)
                cv2.imwrite(str(self.output/'diagnostic_annotated.png'),annotate(self.frame,self.detections))
            if self.latest_candidate is not None:
                cv2.imwrite(str(self.output/'diagnostic_crop.png'),self.latest_candidate.crop)
                (self.output/'diagnostic.json').write_text(json.dumps(dict(
                    label=self.latest_candidate.label,score=self.latest_candidate.score,
                    probabilities=self.latest_candidate.probabilities,
                    age_s=now-self.latest_candidate_time,state=self.control.state),indent=2))
            self.diagnostic_saved=True
        if new_detections is not None:
            self.trace.writerow([round(now,4),self.frame_seq,self.control.state,
                                 self.telemetry.x,self.telemetry.speed,self.last_command,
                                 self.control.range_m,round(self.last_infer_ms,3)])
            self.trace_file.flush()
        self.flush_events()
        if self.args.autostart and not self.started and self.control.state=='READY' and fresh:
            self.start_if_ready()
        if not self.args.headless:
            view=self.make_hud()
            cv2.imshow('AI wall patrol V3',view)
            key=cv2.waitKey(1)&0xff
            if key==ord('s'):self.start_if_ready()
            if key in (ord('q'),27):return False
        return not (self.args.autostart and self.control.state in ('DONE','FAULT','REVIEW_REQUIRED'))

    def make_hud(self):
        hud=np.full((640,960,3),(36,27,17),dtype=np.uint8)
        def line(s,x,y,size=.65,color=(235,240,245)):
            cv2.putText(hud,str(s),(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
        line('AI WALL PATROL V3',20,35,.9,(190,220,90))
        line(f'{self.args.scenario} | '+('CAPTURE ONLY' if self.args.capture_only else 'CNN'),420,35,.65)
        line(f'STATE: {self.control.state}',20,68,.7)
        if self.frame is not None:
            view=annotate(self.frame,self.detections)
            hud[90:570,:640]=cv2.resize(view,(640,480))
        else:
            line('Waiting for camera...',25,250)
        line('ROI capture' if self.args.capture_only else 'Crack inspection',660,105,.65,(190,220,90))
        line(f'x: {self.telemetry.x:.2f} m',660,145)
        line(f'speed: {self.telemetry.speed:.3f}',660,183)
        line(f'command: {self.last_command:.2f}',660,221)
        gap='--' if self.control.range_m is None else f'{self.control.range_m:.2f} m'
        line(f'image gap: {gap}',660,273)
        line(f'stop gap: {self.args.stop_m:.2f} m',660,311)
        line(f'inspections: {len(self.control.inspections)}',660,349)
        line(f'infer: {self.last_infer_ms:.1f} ms',660,387)
        if self.latest_crack is not None:
            line(f'Last crack: {self.latest_crack.score:.3f}',660,425,.56,(80,160,255))
            crop=cv2.cvtColor(cv2.resize(self.latest_crack.crop,(180,120)),cv2.COLOR_GRAY2BGR)
            hud[448:568,660:840]=crop
        line('s: start    q: quit',660,578,.58)
        line('Common frame ROI -> grayscale CNN: normal / stain / crack',20,605,.6,(190,220,90))
        line('Latest: '+(self.control.events[-1]['event'] if self.control.events else 'READY'),
             20,630,.55)
        return hud

    def write_report(self):
        import html
        cards=[]
        for item in self.report_items:
            score='--' if item['confidence'] is None else f"{item['confidence']:.3f}"
            cards.append(f"<article><h2>{'撮影対象（判定未確定）' if self.args.capture_only else 'ひび割れ候補'} {item['index']}</h2>"
                         f"<p>停止位置: x={item['stop_x_m']:.3f} m / CNN出力: {score}</p>"
                         f"<p>推定異常位置 x: {item['estimated_target_x_m']:.3f} m</p>"
                         f"<img src='{item['annotated_image']}'><p>CNNへ入力した領域（グレースケール）</p>"
                         f"<img class='crop' src='inspection_{item['index']:02d}_crop.png'>"
                         f"<p>実際のCNN入力（96×96、明るさ補正後）</p>"
                         f"<img class='crop' src='inspection_{item['index']:02d}_input.png'></article>")
        page=("<!doctype html><meta charset='utf-8'><title>V3 wall inspection report</title>"
              "<style>body{font:17px sans-serif;max-width:1000px;margin:40px auto;color:#163b49}"
              "article{background:#eef4f4;padding:24px;margin:24px 0;border-radius:14px}"
              "img{max-width:100%}.crop{width:250px}h1{color:#087f83}</style>"
              f"<h1>{'撮影データ（認識調整用）' if self.args.capture_only else '壁面巡視 点検レポート'} / {html.escape(self.args.scenario)}</h1>"
              f"<p>終了状態: {html.escape(self.control.state)} / 点検回数: {len(self.report_items)}</p>"
              "<p>合成画像を使ったシミュレーション実証。位置はオドメトリと画像からの推定値。</p>"
              +(''.join(cards) or '<p>ひび割れの点検記録はありません。</p>'))
        (self.output/'inspection_report.html').write_text(page,encoding='utf-8')
        (self.output/'inspection_report.json').write_text(json.dumps(self.report_items,ensure_ascii=False,indent=2))

    def finish(self):
        for _ in range(10):
            self.command(0.0)
            rclpy.spin_once(self,timeout_sec=.05)
        self.flush_events()
        self.trace_file.close()
        self.perception_file.close()
        summary=dict(version='v3.0.2',scenario=self.args.scenario,mode='capture' if self.args.capture_only else 'cnn',
                     capture_only=self.args.capture_only,model_sha256=self.runtime_model['sha256'],
                     final_state=self.control.state,
                     normal_seen=self.control.normal_seen,
                     stain_seen=self.control.stain_seen,
                     inspections=self.control.inspections,
                     stop_m=self.args.stop_m,end_x=self.args.end_x,
                     last_x_m=self.telemetry.x,
                     elapsed_s=self.elapsed(),events=self.control.events)
        (self.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        self.write_report()
        cv2.destroyAllWindows()
        print('最終状態:',self.control.state,'/ 撮影回数:' if self.args.capture_only else '/ 点検回数:',len(self.control.inspections),flush=True)
        print('記録先:',self.output,flush=True)
        if self.args.capture_only:
            import tarfile
            archive=ROOT/'capture_review_latest.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                tar.add(self.output,arcname=self.output.name)
            print('撮影データ一式:',archive,flush=True)
        elif self.control.state in ('FAULT','REVIEW_REQUIRED'):
            import tarfile
            archive=ROOT/'run_review_latest.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                tar.add(self.output,arcname=self.output.name)
            print('確認用記録一式:',archive,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scenario',default='CASE_A')
    parser.add_argument('--mode',choices=['cnn'],default='cnn')
    parser.add_argument('--stop-m',type=float,default=2.8)
    parser.add_argument('--end-x',type=float,default=12.3)
    parser.add_argument('--capture-only',action='store_true')
    parser.add_argument('--autostart',action='store_true')
    parser.add_argument('--headless',action='store_true')
    args=parser.parse_args()
    if args.stop_m<=0 or args.end_x<=0:parser.error('distances must be positive')
    if args.headless and not args.autostart:parser.error('--headless requires --autostart')
    rclpy.init()
    node=None
    try:
        node=PatrolNode(args)
        while rclpy.ok():
            rclpy.spin_once(node,timeout_sec=.04)
            if not node.step():break
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.finish()
            node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':main()
