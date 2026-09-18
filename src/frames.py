import cv2, numpy as np, os, math
def read_frames(path, n=12):
    c=cv2.VideoCapture(path); tot=int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    if tot<=0: c.release(); return []
    idx=np.linspace(0, tot-1, min(n,tot)).astype(int); out=[]; want=set(idx.tolist())
    i=0
    while True:
        ok,f=c.read()
        if not ok: break
        if i in want: out.append((i,f))
        i+=1
    c.release(); return out
def montage(frames, cols=4, cell=(320,240), label=True):
    if not frames: return None
    rows=math.ceil(len(frames)/cols)
    W,H=cell; canvas=np.zeros((rows*H, cols*W, 3), np.uint8)
    for k,(fi,f) in enumerate(frames):
        r,c=divmod(k,cols); img=cv2.resize(f,(W,H))
        if img.ndim==2: img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        if label:
            cv2.putText(img,f"t={fi}",(6,22),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        canvas[r*H:(r+1)*H, c*W:(c+1)*W]=img
    return canvas
def clip_montage(clip_dir, modality, n=12, cols=4, cell=(320,240)):
    p=os.path.join(clip_dir, modality, modality+'.mp4')
    return montage(read_frames(p,n), cols, cell)
