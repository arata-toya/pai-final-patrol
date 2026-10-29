"""再現可能な合成コンクリート画像。実施設の損傷画像ではない。"""
from pathlib import Path
import cv2
import numpy as np

LABELS = ['normal', 'stain', 'crack']


def concrete(label, seed, size=448):
    rng = np.random.default_rng(seed)
    base = rng.uniform(155, 195)
    coarse = cv2.resize(rng.normal(0, 6, (18, 18)).astype('float32'),
                        (size, size), interpolation=cv2.INTER_CUBIC)
    image = base + coarse + rng.normal(0, 3, (size, size))
    # 全クラスで同じ背景・粒子・ゆるい陰影を使用。
    image += np.linspace(-rng.uniform(0, 15), rng.uniform(0, 15), size)[None, :]
    image = np.clip(image, 0, 255).astype('uint8')
    for _ in range(65):
        x, y = rng.integers(0, size, 2)
        cv2.circle(image, (int(x), int(y)), int(rng.integers(1, 4)),
                   int(base-rng.uniform(8, 28)), -1)
    if label == 'crack':
        # 細く不規則な主亀裂と分岐。毎画像で経路・太さ・方向を変更。
        canvas = np.zeros_like(image)
        pts=[]; x=float(rng.uniform(.25, .75)*size)
        for y in np.linspace(size*.08, size*.92, 19):
            x=np.clip(x+rng.normal(0, size*.035), size*.12, size*.88)
            pts.append((int(x), int(y)))
        cv2.polylines(canvas,[np.array(pts)],False,255,int(rng.integers(3,7)),cv2.LINE_AA)
        for index in rng.choice(np.arange(4,16),size=int(rng.integers(2,5)),replace=False):
            xx,yy=pts[index]; direction=rng.choice([-1,1])
            branch=[(xx,yy)]
            for j in range(1,5):
                branch.append((int(xx+direction*j*size*.025+rng.normal(0,4)),
                               int(yy+j*size*.024+rng.normal(0,4))))
            cv2.polylines(canvas,[np.array(branch)],False,255,2,cv2.LINE_AA)
        angle=float(rng.uniform(-75,75))
        canvas=cv2.warpAffine(canvas,cv2.getRotationMatrix2D((size/2,size/2),angle,1),(size,size))
        shadow=cv2.GaussianBlur(canvas,(7,7),1.8)/255
        image=np.clip(image.astype(float)-shadow*24-canvas/255*rng.uniform(65,110),0,255).astype('uint8')
    elif label == 'stain':
        # 面状の汚れ。亀裂と同じ無彩色で、暗さだけをラベルにしない。
        mask=np.zeros_like(image)
        for _ in range(int(rng.integers(2,6))):
            center=tuple(map(int,rng.integers(int(size*.15),int(size*.85),2)))
            axes=(int(rng.uniform(.035,.12)*size),int(rng.uniform(.04,.18)*size))
            cv2.ellipse(mask,center,axes,float(rng.uniform(0,180)),0,360,255,-1)
        mask=cv2.GaussianBlur(mask,(31,31),rng.uniform(4,10))/255
        image=np.clip(image.astype(float)-mask*rng.uniform(20,80),0,255).astype('uint8')
    elif label != 'normal':
        raise ValueError(label)
    return image


def panel_texture(gray):
    # 全クラス共通の青緑枠：対象範囲と既知寸法の手掛かり。分類には使わない。
    canvas=np.full((512,512,3),(170,115,25),dtype='uint8')  # BGR cyan-blue
    canvas[32:480,32:480]=cv2.cvtColor(cv2.resize(gray,(448,448)),cv2.COLOR_GRAY2BGR)
    return canvas


def normalize_gray(image):
    gain=float(np.clip(165.0/(float(image.mean())+1e-6),.6,2.5))
    return np.clip(image.astype(float)*gain,0,255).astype('uint8')


def training_view(image, seed):
    rng=np.random.default_rng(seed)
    image=cv2.resize(image,(112,112))
    shift=int(rng.integers(0,9)); image=image[shift:shift+96,shift:shift+96]
    image=cv2.resize(image,(96,96))
    image=np.clip(image.astype(float)*rng.uniform(.65,1.15)+rng.uniform(-12,12),0,255).astype('uint8')
    if rng.random()<.5:image=cv2.GaussianBlur(image,(3,3),float(rng.uniform(.2,.9)))
    if rng.random()<.5:image=cv2.flip(image,1)
    return normalize_gray(image)
