"""共通枠で領域を求め、中の無彩色パターンだけをCNNへ入力する。"""
from pathlib import Path
from dataclasses import dataclass
import math
import cv2,numpy as np,torch
from network import make_model,LABELS
from synthetic import normalize_gray
ROOT=Path(__file__).resolve().parents[1]

@dataclass
class Detection:
    label:str
    score:float
    bbox:tuple
    forward_m:float|None
    quad:object=None
    crop:object=None
    probabilities:object=None

def ordered_quad(points):
    points=np.asarray(points,dtype='float32').reshape(4,2)
    ordered=np.empty((4,2),dtype='float32');s=points.sum(1);d=np.diff(points,axis=1).ravel()
    ordered[0]=points[np.argmin(s)];ordered[2]=points[np.argmax(s)]
    ordered[1]=points[np.argmin(d)];ordered[3]=points[np.argmax(d)]
    return ordered

def forward_distance(quad,image_width=640):
    q=ordered_quad(quad);f=image_width/(2*math.tan(1.2/2))
    h=(np.linalg.norm(q[3]-q[0])+np.linalg.norm(q[2]-q[1]))/2
    depth=f*1.4/h
    u=float(q[:,0].mean());left=((image_width-1)/2-u)*depth/f
    return math.cos(.45)*depth-math.sin(.45)*left

class WallVision:
    def __init__(self,threshold=.85):
        torch.set_num_threads(2);self.threshold=threshold
        saved=torch.load(ROOT/'models/wall_classifier.pt',map_location='cpu',weights_only=True)
        if saved['labels']!=LABELS:raise ValueError('model labels do not match')
        self.model=make_model();self.model.load_state_dict(saved['model_state']);self.model.eval()

    def detect(self,frame):
        h,w=frame.shape[:2];hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,(85,85,45),(115,255,255))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,hierarchy=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        candidates=[]
        for index,contour in enumerate(contours):
            # 枠が遠方パネルと接触しても、枠内の穴を個別に取り出す。
            if hierarchy is None or hierarchy[0][index][3]<0:continue
            x,y,bw,bh=cv2.boundingRect(contour)
            if bw<22 or bh<35 or bw*bh<1200 or not .22<bw/bh<1.6:continue
            if x<=1 or y<=1 or x+bw>=w-1 or y+bh>=h-1:continue
            approx=cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True)
            if len(approx)!=4 or not cv2.isContourConvex(approx):continue
            q=ordered_quad(approx);dest=np.float32([[0,0],[159,0],[159,159],[0,159]])
            transform=cv2.getPerspectiveTransform(q,dest)
            rectified=cv2.warpPerspective(frame,transform,(160,160))
            crop=cv2.cvtColor(rectified[3:157,3:157],cv2.COLOR_BGR2GRAY)
            distance=forward_distance(q,w)
            if not .4<distance<18:continue
            candidates.append(((x,y,bw,bh),q,crop,distance))
        candidates=sorted(candidates,key=lambda z:z[0][2]*z[0][3],reverse=True)[:8]
        if not candidates:return []
        batch=np.stack([normalize_gray(cv2.resize(c[2],(96,96))) for c in candidates])
        with torch.inference_mode():
            probs=self.model(torch.tensor(batch[:,None]/255,dtype=torch.float32)).softmax(1).numpy()
        results=[]
        for (box,q,crop,distance),p in zip(candidates,probs):
            index=int(p.argmax());score=float(p[index]);label=LABELS[index] if score>=self.threshold else 'uncertain'
            results.append(Detection(label,score,box,distance,q,crop,p.tolist()))
        return results

def annotate(frame,detections):
    out=frame.copy()
    for item in detections:
        color={'crack':(40,60,255),'stain':(0,185,245),'normal':(70,220,70),'uncertain':(180,180,180)}[item.label]
        if item.quad is not None:cv2.polylines(out,[item.quad.astype('int32')],True,color,2)
        x,y,_,_=item.bbox
        cv2.putText(out,f'{item.label} {item.score:.2f}  {item.forward_m:.2f}m',
                    (x,max(16,y-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
    return out
