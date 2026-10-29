"""画像由来の距離で停止する制御コア。コース内のひび割れ位置を参照しない。"""
from dataclasses import dataclass
import math
from statistics import median


@dataclass
class Telemetry:
    x: float
    speed: float
    turn_speed: float
    sequence: int


class PatrolControl:
    ACTIVE = {"RUN", "BRAKING", "INSPECT", "EXIT", "FINAL_STOP", "REVIEW_BRAKING"}

    def __init__(self, stop_m=2.8, end_x=12.3, capture_only=False):
        self.capture_only=capture_only
        self.uncertain_count=0
        self.state = "READY"
        self.stop_m, self.end_x = stop_m, end_x
        self.camera_forward = 1.3-0.151427
        self.events = []
        self.inspections = []
        self.visited_world_x = []
        self.normal_seen = False
        self.stain_seen = False
        self.target_world = None
        self.last_target_time = None
        self.crack_history = []
        self.crack_count = 0
        self.last_frame = -1
        self.last_stop_seq = -1
        self.stopped_samples = 0
        self.state_entered = 0
        self.pause_until = 0
        self.range_m = None
        self.started_at = None

    def event(self, now, t, name, **kwargs):
        self.events.append(dict(elapsed_s=now, x_m=t.x, event=name, **kwargs))

    def start(self, now, t):
        if self.state != "READY":
            return False
        if abs(t.x) > 0.5:
            return False
        self.started_at = now
        self.state = "RUN"
        self.state_entered = now
        self.event(now,t,"START")
        return True

    def fault(self, now, t, reason):
        if self.state != "FAULT":
            self.state="FAULT"
            self.event(now,t,"FAULT",reason=reason)

    def begin_braking(self, now, t, final=False):
        self.state = "FINAL_STOP" if final else "BRAKING"
        self.state_entered=now
        self.last_stop_seq=t.sequence
        self.stopped_samples=0
        self.event(now,t,"END_BRAKING" if final else ("ROI_BRAKING" if self.capture_only else "AI_BRAKING"),
                   image_range_m=self.range_m)

    def tick(self, now, t, detections=None, frame_id=None, fresh=True):
        if self.state in self.ACTIVE and not fresh:
            self.fault(now,t,"image_or_odometry_timeout")
        if self.state in self.ACTIVE and self.started_at is not None:
            if now-self.started_at > 180:
                self.fault(now,t,"run_timeout")

        if detections is not None and frame_id != self.last_frame:
            self.last_frame=frame_id
            if self.state in ("RUN","EXIT"):
                if any(d.label=="normal" for d in detections) and not self.normal_seen:
                    self.normal_seen=True
                    self.event(now,t,"NORMAL_RECOGNIZED")
                if any(d.label=="stain" for d in detections) and not self.stain_seen:
                    self.stain_seen=True
                    self.event(now,t,"STAIN_RECOGNIZED")
                if not self.capture_only:
                    near_uncertain=any(d.label=='uncertain' and d.forward_m is not None
                        and d.forward_m<=3.2 and not any(
                            abs(t.x+self.camera_forward+d.forward_m-v)<1.0
                            for v in self.visited_world_x) for d in detections)
                    self.uncertain_count=self.uncertain_count+1 if near_uncertain else 0
                    if self.uncertain_count>=3:
                        self.state='REVIEW_BRAKING'
                        self.state_entered=now
                        self.last_stop_seq=t.sequence
                        self.stopped_samples=0
                        self.event(now,t,'UNCERTAIN_BRAKING',reason='near_uncertain_candidate')
                cracks=[]
                for d in detections:
                    if (d.label!="crack" and not self.capture_only) or d.forward_m is None:
                        continue
                    world_x=t.x+self.camera_forward+d.forward_m
                    if any(abs(world_x-v)<1.0 for v in self.visited_world_x):
                        continue
                    cracks.append((d.forward_m,world_x))
                if cracks:
                    gap,world_x=min(cracks)
                    if self.target_world is None or abs(world_x-self.target_world)>0.8:
                        self.crack_history=[]
                        self.crack_count=0
                    self.crack_history=(self.crack_history+[gap])[-3:]
                    self.range_m=median(self.crack_history)
                    self.target_world=world_x
                    self.last_target_time=now
                    self.crack_count=self.crack_count+1 if self.range_m<=self.stop_m else 0
                    if self.crack_count>=3 and self.state in ("RUN","EXIT"):
                        self.begin_braking(now,t)
                else:
                    self.crack_count=0

        if self.state in ("RUN","EXIT"):
            if t.x>=self.end_x:
                self.begin_braking(now,t,final=True)
            elif self.target_world is not None and self.last_target_time is not None:
                remaining=self.target_world-t.x-self.camera_forward
                if remaining<3.0 and now-self.last_target_time>1.5:
                    self.fault(now,t,"near_target_lost")

        if self.state in ("BRAKING","FINAL_STOP","REVIEW_BRAKING"):
            if t.sequence!=self.last_stop_seq:
                self.last_stop_seq=t.sequence
                stopped=(math.isfinite(t.speed) and math.isfinite(t.turn_speed)
                         and t.speed<0.01 and t.turn_speed<0.02)
                self.stopped_samples=self.stopped_samples+1 if stopped else 0
            if self.stopped_samples>=2:
                if self.state=='REVIEW_BRAKING':
                    self.state='REVIEW_REQUIRED'
                    self.event(now,t,'REVIEW_REQUIRED',reason='uncertain_candidate_stop_confirmed')
                elif self.state=="BRAKING":
                    item=dict(index=len(self.inspections)+1,stop_x_m=t.x,
                              estimated_target_x_m=self.target_world,
                              last_image_range_m=self.range_m)
                    self.inspections.append(item)
                    self.visited_world_x.append(self.target_world)
                    self.event(now,t,"CAPTURE_CONFIRMED" if self.capture_only else "STOP_CONFIRMED",**item)
                    self.state="INSPECT"
                    self.pause_until=now+3
                else:
                    self.state="DONE"
                    self.event(now,t,"DONE",inspection_count=len(self.inspections))
            elif now-self.state_entered>20:
                self.fault(now,t,"stop_confirmation_timeout")

        if self.state=="INSPECT" and now>=self.pause_until:
            self.event(now,t,"RESUME")
            self.state="EXIT"
            self.target_world=None
            self.range_m=None
            self.crack_history=[]
            self.crack_count=0
            self.last_target_time=None

        if self.state not in ("RUN","EXIT"):
            return 0.0
        if self.target_world is None:
            return 0.28
        gap=self.target_world-t.x-self.camera_forward
        if gap<2.5:
            return 0.08
        if gap<3.5:
            return 0.14
        return 0.28
