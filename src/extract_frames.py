"""Sample evenly-spaced IR frames from each clip and write them as JPEGs, so the whole
corpus fits in a small Kaggle dataset instead of shipping 6 GB of mp4.

HARn clips are ~2 s single-action  -> 8 frames.
HAU  clips are ~12 s multi-action  -> 20 frames (needed for `sequence` / `multi`).
"""
import cv2, os, sys, json, numpy as np, pandas as pd, collections, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import load

N_HARN, N_HAU = 8, 20
QUALITY, MAXW = 82, 640

def sample(video, n):
    c=cv2.VideoCapture(video); tot=int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    if tot<=0: c.release(); return []
    want={int(round(x)) for x in np.linspace(0, tot-1, min(n,tot))}
    out=[]; i=0
    while True:
        ok,f=c.read()
        if not ok: break
        if i in want:
            if f.shape[1]>MAXW:
                s=MAXW/f.shape[1]; f=cv2.resize(f,(MAXW,int(f.shape[0]*s)))
            out.append(f)
        i+=1
    c.release(); return out

def clip_key(path, source):
    p=str(path)
    if p.startswith('large_model_track_test'): return p.split('/')[1]
    return p.strip('/').replace('/','__')

def clip_dir(path, source, root):
    p=str(path)
    if p.startswith('large_model_track_test'): return os.path.join(root, os.path.dirname(os.path.dirname(p)))
    return os.path.join(root, p)   # training path is the clip directory

def run(rows, outdir, root, tag):
    os.makedirs(outdir, exist_ok=True); man=[]
    for i,(path,source) in enumerate(rows):
        key=clip_key(path,source); n=N_HARN if source=='HARn' else N_HAU
        vid=os.path.join(clip_dir(path,source,root),'IR','IR.mp4')
        if not os.path.exists(vid):
            print("  MISSING", vid); continue
        fr=sample(vid,n)
        d=os.path.join(outdir,key); os.makedirs(d,exist_ok=True)
        for j,f in enumerate(fr):
            cv2.imwrite(os.path.join(d,f"{j:02d}.jpg"),f,[cv2.IMWRITE_JPEG_QUALITY,QUALITY])
        man.append({'path':path,'source':source,'key':key,'n_frames':len(fr)})
        if (i+1)%50==0: print(f"  {tag} {i+1}/{len(rows)}")
    pd.DataFrame(man).to_csv(os.path.join(outdir,'_manifest.csv'),index=False)
    mb=sum(os.path.getsize(os.path.join(dp,f)) for dp,_,fs in os.walk(outdir) for f in fs)/1e6
    print(f"  {tag}: {len(man)} clips, {mb:.0f} MB -> {outdir}")

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('which',choices=['test','train'])
    ap.add_argument('--limit',type=int,default=0); ap.add_argument('--source',default=''); a=ap.parse_args()
    tr,te=load()
    if a.which=='test':
        rows=te[['path','source']].drop_duplicates().values.tolist()
        run(rows,'frames/test','data','test')
    else:
        t=tr[tr.source==a.source] if a.source else tr
        rows=t[['path','source']].drop_duplicates().values.tolist()
        if a.limit: rows=rows[:a.limit]
        run(rows,'frames/train','data','train')
