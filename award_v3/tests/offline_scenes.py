"""幾何投影した合成カメラと簡易移動モデル。Gazebo実走行とは別の確認。"""
from pathlib import Path
import sys,json,math
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cv2,numpy as np
from build_worlds import SCENES
from vision import WallVision,annotate
from control import PatrolControl,Telemetry
ROOT=Path(__file__).resolve().parents[1]

def render(x,panels):
    frame=np.full((480,640,3),190,dtype='uint8');frame[250:]=155
    focal=640/(2*math.tan(.6));c,s=math.cos(.45),math.sin(.45)
    for label,world_x in sorted(panels,key=lambda p:p[1],reverse=True):
        dx=world_x-x-(1.3-.151427)
        points=[];depths=[]
        # Texture TL,TR,BR,BL on a plane facing -X.
        for offset,z in [(-.8,2.0),(.8,2.0),(.8,.4),(-.8,.4)]:
            ddx=dx+offset;dy=1.45;depth=c*ddx+s*dy;left=-s*ddx+c*dy
            if depth<=.2:break
            points.append((319.5-focal*left/depth,239.5-focal*(z-.95)/depth));depths.append(depth)
        if len(points)!=4:continue
        points=np.float32(points)
        if points[:,0].max()<0 or points[:,0].min()>640:continue
        texture=cv2.imread(str(ROOT/'assets/textures'/f'{label}.png'))
        transform=cv2.getPerspectiveTransform(np.float32([[0,0],[511,0],[511,511],[0,511]]),points)
        warped=cv2.warpPerspective(texture,transform,(640,480))
        mask=cv2.warpPerspective(np.full((512,512),255,dtype='uint8'),transform,(640,480))
        frame[mask>128]=warped[mask>128]
    return frame

def main():
    vision=WallVision();results=[]
    for case,panels in SCENES.items():
        control=PatrolControl();x=0.;speed=0.;seq=0;control.start(0,Telemetry(0,0,0,0))
        seen=set();positions=[]
        for step in range(1800):
            now=step*.1;frame=render(x,panels);detections=vision.detect(frame)
            seen.update(d.label for d in detections)
            if step==0:cv2.imwrite(str(ROOT/'checks'/f'{case}_projected_start.png'),annotate(frame,detections))
            before=len(control.inspections)
            command=control.tick(now,Telemetry(x,speed,0,seq),detections,seq)
            if len(control.inspections)>before:
                cv2.imwrite(str(ROOT/'checks'/f'{case}_projected_stop_{len(control.inspections)}.png'),annotate(frame,detections))
                positions.append(x)
            if control.state in ('DONE','FAULT','REVIEW_REQUIRED'):break
            speed+=max(-.1,min(.1,command-speed));x+=speed*.1;seq+=1
        expected=sum(label.startswith('crack') for label,_ in panels)
        result=dict(case=case,state=control.state,expected=expected,inspections=len(control.inspections),
                    stop_positions=positions,observed_classes=sorted(seen),events=control.events)
        truth=[px for label,px in panels if label.startswith('crack')]
        gaps=[target-stop-(1.3-.151427) for target,stop in zip(truth,positions)]
        result['true_forward_gaps_m']=gaps
        result['pass']=control.state=='DONE' and len(control.inspections)==expected and all(abs(g-2.8)<.6 for g in gaps)
        results.append(result);print(case,control.state,len(control.inspections),'/',expected, sorted(seen),flush=True)
    (ROOT/'checks/projected_scene_checks.json').write_text(json.dumps(dict(
        scope='synthetic projection + simple 1D motion; NOT Gazebo',results=results),indent=2))
    if not all(r['pass'] for r in results):raise SystemExit('offline scenario failed')

if __name__=='__main__':main()
