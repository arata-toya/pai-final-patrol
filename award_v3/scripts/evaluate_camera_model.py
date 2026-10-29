from pathlib import Path
import argparse,json
import cv2,numpy as np,torch
from network import make_model,LABELS
from synthetic import concrete,training_view,normalize_gray
from train_camera_robust import camera_view
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--captures',type=Path);args=parser.parse_args();torch.set_num_threads(2)
    models={}
    for tag,file in [('before','wall_classifier_before_camera_fix.pt'),('after','wall_classifier.pt')]:
        m=make_model();m.load_state_dict(torch.load(ROOT/'models'/file,map_location='cpu',weights_only=True)['model_state']);m.eval();models[tag]=m
    output=ROOT/'checks/camera_model';output.mkdir(parents=True,exist_ok=True)
    def evaluate(images,truth,name,scope):
        truth=np.asarray(truth);result={}
        for tag,m in models.items():
            scores=[]
            for start in range(0,len(images),64):
                x=np.stack([normalize_gray(cv2.resize(im,(96,96))) for im in images[start:start+64]])[:,None]/255
                with torch.inference_mode():scores.append(m(torch.tensor(x,dtype=torch.float32)).softmax(1).numpy())
            scores=np.concatenate(scores);pred=scores.argmax(1);matrix=np.zeros((3,3),dtype=int);accepted=scores.max(1)>=.85
            for y,p in zip(truth,pred):matrix[y,p]+=1
            result[tag]=dict(correct=int((pred==truth).sum()),total=len(truth),confusion_matrix=matrix.tolist(),
                accepted=int(accepted.sum()),accepted_correct=int(((pred==truth)&accepted).sum()),
                crack_TP=int(((truth==2)&(pred==2)&accepted).sum()),crack_FP=int(((truth!=2)&(pred==2)&accepted).sum()))
            if len(images)==3:result[tag]['scores']=scores.tolist()
        (output/(name+'.json')).write_text(json.dumps(dict(scope=scope,labels=LABELS,threshold=.85,results=result),indent=2))
        print(name,result,flush=True)
    if args.captures:
        evaluate([cv2.imread(str(args.captures/f'inspection_{i:02d}_crop.png'),0) for i in [1,2,3]],
                 [0,1,2],'gazebo_development_3','Three known CASE_A capture crops for development verification, not independent test or autonomous task success')
    rows=[r for r in json.loads((args.source_root/'data/manifest.json').read_text()) if r['split']=='test']
    evaluate([cv2.imread(str(args.source_root/r['path']),0) for r in rows],
             [LABELS.index(r['label']) for r in rows],'existing_test_180','Existing synthetic hold-out crops')
    images=[];truth=[];degraded=[]
    for labelindex,label in enumerate(LABELS):
        for i in range(200):
            seed=800000+labelindex*1000+i
            im=training_view(concrete(label,seed),seed+500000)
            images.append(im);truth.append(labelindex)
            degraded.append(camera_view(im,np.random.default_rng(400000+seed)))
    evaluate(images,truth,'fresh_native_600','Fresh synthetic seed 800000+, no training or validation overlap')
    evaluate(degraded,truth,'fresh_degraded_600','Paired degraded versions of the same 600 fresh synthetic images; not 600 additional independent images')

if __name__=='__main__':main()
