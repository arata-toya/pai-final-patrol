#!/usr/bin/env python3
"""V3の既存合成学習画像だけを使用。Gazebo撮影画像は学習・選択に使用しない。"""
from pathlib import Path
import argparse,json,csv,hashlib
import cv2,numpy as np,torch
from network import make_model,LABELS
from synthetic import normalize_gray
ROOT=Path(__file__).resolve().parents[1]

def camera_view(image,rng,augment=True):
    if not augment:return image.copy()
    # 投影による横方向の画素減少、再拡大、描画のぼけを模擬する。
    width=int(rng.integers(32,97));height=int(rng.integers(60,113))
    reduced=cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA)
    out=cv2.resize(reduced,(96,96),interpolation=cv2.INTER_LINEAR)
    if rng.random()<.85:
        sigma=float(rng.uniform(.2,1.3));out=cv2.GaussianBlur(out,(5,5),sigma)
    out=np.clip(out.astype(float)+rng.normal(0,float(rng.uniform(0,1)),out.shape),0,255).astype('uint8')
    if rng.random()<.5:out=cv2.flip(out,1)
    return normalize_gray(out)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--initial-model',type=Path,default=ROOT/'models/wall_classifier_before_camera_fix.pt')
    args=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(84141)
    manifest=json.loads((args.source_root/'data/manifest.json').read_text());data={}
    for split in ['train','validation']:
        rows=[r for r in manifest if r['split']==split]
        data[split]=([cv2.imread(str(args.source_root/r['path']),0) for r in rows],
                     np.array([LABELS.index(r['label']) for r in rows]))
    train,y=data['train'];validation,yv=data['validation']
    rng=np.random.default_rng(84141)
    val_views=np.stack(validation+[camera_view(im,np.random.default_rng(600+i)) for i,im in enumerate(validation)]
                       +[camera_view(im,np.random.default_rng(1600+i)) for i,im in enumerate(validation)])
    xv=torch.tensor(val_views[:,None]/255,dtype=torch.float32);yv=torch.tensor(np.tile(yv,3))
    model=make_model();saved=torch.load(args.initial_model,map_location='cpu',weights_only=True)
    model.load_state_dict(saved['model_state']);optimizer=torch.optim.Adam(model.parameters(),lr=.00015)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=12,gamma=.5)
    best=-1;log=[]
    for epoch in range(1,37):
        # 各元画像の無加工1枚と、異なる劣化2枚を使用する。
        augmented=np.stack(train+[camera_view(im,rng) for im in train]+[camera_view(im,rng) for im in train])
        x=torch.tensor(augmented[:,None]/255,dtype=torch.float32);target=torch.tensor(np.tile(y,3))
        order=torch.randperm(len(target));loss_sum=0;model.train()
        for start in range(0,len(order),48):
            idx=order[start:start+48];logits=model(x[idx]);loss=torch.nn.functional.cross_entropy(logits,target[idx])
            optimizer.zero_grad();loss.backward();optimizer.step();loss_sum+=float(loss)*len(idx)
        model.eval()
        with torch.inference_mode():prediction=model(xv).argmax(1);accuracy=float((prediction==yv).float().mean())
        log.append(dict(epoch=epoch,loss=loss_sum/len(target),validation_paired_accuracy=accuracy))
        if accuracy>best:
            best=accuracy
            torch.save(dict(model_state=model.state_dict(),labels=LABELS,image_size=96,channels=1,
                            seed=84141,selected_epoch=epoch,version='v3.0.2-camera-augmentation'),ROOT/'models/wall_classifier.pt')
        scheduler.step()
        if epoch%6==0 or epoch==1:print(epoch,round(loss_sum/len(target),4),'validation paired',round(accuracy,4),flush=True)
    with (ROOT/'models/camera_training_log.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(log[0]));w.writeheader();w.writerows(log)
    print('selected validation accuracy',best,flush=True)

if __name__=='__main__':main()
