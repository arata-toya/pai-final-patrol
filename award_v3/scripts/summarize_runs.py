"""シーン正解を使うのは走行後の評価だけ。制御ノードはこのファイルを読まない。"""
from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parents[1]

def main():
    scenes=json.loads((ROOT/'scenarios.json').read_text());rows=[]
    for p in sorted((ROOT/'runs_v3').glob('*/summary.json')):
        d=json.loads(p.read_text())
        if d.get('capture_only') or d.get('mode')=='capture':continue
        truth=[a['x_m'] for a in scenes[d['scenario']]['panels'] if a['class_name']=='crack']
        expected=len(truth);extra=0;gaps=[]
        for item in d['inspections']:
            if not truth:extra+=1;continue
            target=min(truth,key=lambda t:abs(t-item['estimated_target_x_m']))
            if abs(target-item['estimated_target_x_m'])>1.0:extra+=1;continue
            truth.remove(target);gaps.append(target-item['stop_x_m']-(1.3-.151427))
        success=d['final_state']=='DONE' and not truth and extra==0 and all(abs(x-d['stop_m'])<=.6 for x in gaps)
        rows.append(dict(run=p.parent.name,scenario=d['scenario'],model_sha256=d.get('model_sha256','not_recorded'),state=d['final_state'],expected=expected,
                         inspections=len(d['inspections']),missed=len(truth),extra=extra,
                         forward_gaps_m=';'.join(f'{g:.4f}' for g in gaps),success=success))
    if not rows:
        print('Gazebo実走行記録はまだありません。');return
    out=ROOT/'evaluation_v3';out.mkdir(exist_ok=True)
    with (out/'runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(sum(x['success'] for x in rows),'/',len(rows),'success')
    for model in sorted({r['model_sha256'] for r in rows}):
        selected=[r for r in rows if r['model_sha256']==model]
        print('Model',model[:12],sum(r['success'] for r in selected),'/',len(selected),'success')
    print('Saved:',out/'runs.csv')

if __name__=='__main__':main()
