#!/usr/bin/env python3
"""学習・選択用validation・最終testの画像seedを分離する。CPU実行。"""
from pathlib import Path
import json,csv,hashlib
import cv2,numpy as np,torch
from torch import nn
from synthetic import LABELS,concrete,training_view,panel_texture
from network import make_model
ROOT=Path(__file__).resolve().parents[1]

def main():
    torch.set_num_threads(2);torch.manual_seed(84140)
    arrays={};manifest=[]
    for split,count,offset in [('train',240,10000),('validation',60,100000),('test',60,300000)]:
        images=[];targets=[]
        for index,label in enumerate(LABELS):
            for i in range(count):
                seed=offset+index*1000+i
                raw=concrete(label,seed)
                image=training_view(raw,seed+500000)
                dest=ROOT/'data'/split/label/f'{seed}.png';dest.parent.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(dest),image)
                manifest.append(dict(split=split,label=label,seed=seed,path=str(dest.relative_to(ROOT)),
                                     sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
                images.append(image);targets.append(index)
        arrays[split]=(torch.tensor(np.stack(images)[:,None]/255,dtype=torch.float32),torch.tensor(targets))
    model=make_model();opt=torch.optim.Adam(model.parameters(),lr=.0007)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=15,gamma=.4)
    best=-1;log=[]
    for epoch in range(1,51):
        model.train();x,y=arrays['train'];order=torch.randperm(len(y));loss_sum=0
        for start in range(0,len(y),32):
            idx=order[start:start+32];z=x[idx]
            # 明るさを各エポックで独立に変える。色相は入力自体にない。
            gain=.85+.3*torch.rand(len(z),1,1,1);z=(z*gain).clamp(0,1)
            loss=nn.functional.cross_entropy(model(z),y[idx]);opt.zero_grad();loss.backward();opt.step()
            loss_sum+=float(loss)*len(idx)
        model.eval()
        with torch.inference_mode():
            xv,yv=arrays['validation']; acc=float((model(xv).argmax(1)==yv).float().mean())
        log.append(dict(epoch=epoch,loss=loss_sum/len(y),validation_accuracy=acc))
        if acc>best:
            best=acc
            torch.save(dict(model_state=model.state_dict(),labels=LABELS,image_size=96,
                            channels=1,seed=84140,selected_epoch=epoch),ROOT/'models/wall_classifier.pt')
        scheduler.step()
        if epoch%5==0 or epoch==1:print(epoch,round(loss_sum/len(y),4),'validation',round(acc,4),flush=True)
    checkpoint=torch.load(ROOT/'models/wall_classifier.pt',weights_only=True);model.load_state_dict(checkpoint['model_state']);model.eval()
    with torch.inference_mode():
        xt,yt=arrays['test'];pred=model(xt).argmax(1)
    matrix=np.zeros((3,3),dtype=int)
    for a,b in zip(yt.tolist(),pred.tolist()):matrix[a,b]+=1
    result=dict(scope='held-out synthetic crops, NOT Gazebo or real damage',labels=LABELS,
                selected_epoch=checkpoint['selected_epoch'],train_counts={s:240 for s in LABELS},
                validation_counts={s:60 for s in LABELS},test_counts={s:60 for s in LABELS},
                confusion_matrix=matrix.tolist(),correct=int((pred==yt).sum()),total=len(yt),
                accuracy=float((pred==yt).float().mean()))
    (ROOT/'checks/classification_test.json').write_text(json.dumps(result,indent=2))
    (ROOT/'data/manifest.json').write_text(json.dumps(manifest,indent=2))
    with (ROOT/'models/training_log.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(log[0]));w.writeheader();w.writerows(log)
    # Gazebo用の画像はtrain/validation/testのいずれとも異なるseedを使用。
    for label,seed in [('normal',900010),('stain',901010),('crack',902010),('crack_alt',902020)]:
        actual='crack' if label=='crack_alt' else label
        cv2.imwrite(str(ROOT/'assets/textures'/f'{label}.png'),panel_texture(concrete(actual,seed)))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
