# ThreeDPW_loader.py  – 19 Apr 2025
# 3DPW ➜ COCO‑17 clip loader with frame‑by‑frame actor re‑ID.

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

_JMAP = [0,16,15,-1,17,5,2,6,3,7,4,12,9,13,10,14,11]
_NUM  = 17


class ThreeDPWSequenceDataset(Dataset):
    def __init__(self,
                 seq_dir: str | Path,
                 img_root: str | Path,
                 *,
                 input_len: int = 2,
                 output_len: int = 2,
                 frame_gap: int = 1,
                 image_size: Tuple[int,int]=(192,256),
                 heatmap_size: Tuple[int,int]=(48,64),
                 sigma: int = 2,
                 normalize: bool = True):
        self.seq_dir, self.img_root = Path(seq_dir), Path(img_root)
        self.in_len, self.out_len, self.step = input_len, output_len, frame_gap
        self.img_sz = np.array(image_size,np.int32);  self.hm_sz = np.array(heatmap_size,np.int32)
        self.sigma, self.norm = sigma, normalize
        self.clips: List[Dict] = [];  self._index()

    # ------------------------------------------------------------------
    def _index(self):
        for pkl in sorted(self.seq_dir.glob("*.pkl")):
            d = pickle.load(pkl.open("rb"), encoding="latin1")
            seq   = d["sequence"]
            frames= np.asarray(d["img_frame_ids"], int)          # (T,)
            poses = np.asarray(d["poses2d"], dtype=np.float32)   # (A,T,*,*)
            if poses.ndim==3: poses = poses[:,None,...].transpose(1,0,2,3)
            if poses.shape[-2:]==(3,18): poses = poses.transpose(0,1,3,2)

            A,T   = poses.shape[:2]
            clip_len = (self.in_len+self.out_len-1)*self.step+1
            for t0 in range(0, T-clip_len+1):
                idx = np.arange(t0, t0+clip_len, self.step)
                self.clips.append(dict(seq=seq, fids=frames[idx], poses=poses[:,idx])) # (A,clip_len,18,3)

    def __len__(self): return len(self.clips)

    # ------------------------------------------------------------------
    @staticmethod
    def _third(a,b): d=a-b; return b+np.array([-d[1], d[0]], np.float32)

    def _affine_mat(self,c,s):
        if not isinstance(s,np.ndarray): s=np.array([s,s],np.float32)
        sw,sh=s*200.; ow,oh=self.img_sz
        src=np.zeros((3,2),np.float32); dst=np.zeros_like(src)
        src_dir=np.array([0,-0.5*sh],np.float32); dst_dir=np.array([0,-0.5*oh],np.float32)
        src[0]=c; src[1]=c+src_dir; src[2]=self._third(src[0],src[1])
        dst[0]=[0.5*ow,0.5*oh]; dst[1]=dst[0]+dst_dir; dst[2]=self._third(dst[0],dst[1])
        return cv2.getAffineTransform(src.astype(np.float32), dst.astype(np.float32))

    @staticmethod
    def _affine_pt(pt,M): return np.dot(M,[pt[0],pt[1],1.])[:2]

    def _heatmaps(self, joints, vis):
        tgt=np.zeros((_NUM,self.hm_sz[1],self.hm_sz[0]),np.float32)
        tw =np.ones ((_NUM,1),np.float32)
        stride=self.img_sz/self.hm_sz; tmp=self.sigma*3
        for j in range(_NUM):
            if vis[j]==0: tw[j]=0; continue
            mu=(joints[j]/stride+0.5).astype(int)
            if not(0<=mu[0]<self.hm_sz[0] and 0<=mu[1]<self.hm_sz[1]): tw[j]=0; continue
            ul=mu-tmp; br=mu+tmp+1; size=2*tmp+1
            g=np.exp(-(((np.arange(size)-tmp)**2)[:,None]+((np.arange(size)-tmp)[None])**2)/(2*self.sigma**2))
            g_x=max(0,-ul[0]),min(br[0],self.hm_sz[0])-ul[0]; g_y=max(0,-ul[1]),min(br[1],self.hm_sz[1])-ul[1]
            img_x=max(0,ul[0]),min(br[0],self.hm_sz[0]); img_y=max(0,ul[1]),min(br[1],self.hm_sz[1])
            tgt[j,img_y[0]:img_y[1],img_x[0]:img_x[1]]=g[g_y[0]:g_y[1],g_x[0]:g_x[1]]
        return tgt,tw

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        clip = self.clips[idx]
        A,L  = clip["poses"].shape[:2]       # actors, clip length
        kp_all=torch.zeros(L,_NUM,3)
        imgs,hms,wts=[],[],[]

        prev_torso=None
        for t in range(L):
            # ---------- pick actor for this frame -----------------
            best_a = 0
            if prev_torso is None:
                # first frame → choose most visible
                visibility = [(clip["poses"][a,t,:,2]>0).sum() for a in range(A)]
                best_a = int(np.argmax(visibility))
            else:
                # choose closest torso centre to previous
                centres=[]
                for a in range(A):
                    kp18=clip["poses"][a,t]
                    torso= kp18[[5,2,12,9],:2]   # shoulders & hips
                    vis  = kp18[[5,2,12,9],2]>0
                    centres.append(torso[vis].mean(axis=0) if vis.any() else np.array([1e6,1e6]))
                dists=[np.linalg.norm(c-prev_torso) for c in centres]
                best_a=int(np.argmin(dists))
            kp18=clip["poses"][best_a,t]

            # ---------- map joints --------------------------------
            kp=np.zeros((_NUM,3),np.float32); vis=np.zeros(_NUM,np.float32)
            for cj,src in enumerate(_JMAP):
                if src>=0: kp[cj]=kp18[src]; vis[cj]=kp18[src,2]

            # ---------- bbox & affine -----------------------------
            v=vis>0
            x1=y1=0.; x2=y2=1.
            if v.any():
                x1,y1=kp[v,0].min(),kp[v,1].min(); x2,y2=kp[v,0].max(),kp[v,1].max()
            w,h=x2-x1,y2-y1
            x1-=0.1*w; y1-=0.15*h; x2+=0.1*w; y2+=0.05*h
            ctr=np.array([(x1+x2)/2,(y1+y2)/2],np.float32)
            scl=np.array([x2-x1,y2-y1],np.float32)/200.
            M=self._affine_mat(ctr,scl)

            # ---------- crop image --------------------------------
            fid=clip["fids"][t]
            rgb=cv2.cvtColor(cv2.imread(str(self.img_root/clip["seq"]/f"image_{int(fid):05d}.jpg")),cv2.COLOR_BGR2RGB)
            crop=cv2.warpAffine(rgb,M,tuple(self.img_sz.astype(int))).astype(np.float32)/255.
            if self.norm: crop=(crop-[0.485,0.456,0.406])/[0.229,0.224,0.225]
            imgs.append(torch.from_numpy(crop.transpose(2,0,1)))

            # ---------- transform joints --------------------------
            for j in range(_NUM):
                if vis[j]>0: kp[j,:2]=self._affine_pt(kp[j,:2],M)
            kp_all[t]=torch.from_numpy(kp)

            hm,wt=self._heatmaps(kp[:,:2],vis)
            hms.append(torch.from_numpy(hm)); wts.append(torch.from_numpy(wt))

            # update torso centre for next frame
            torso_pts=kp[[5,2,11,12],:2]; vis_torso=vis[[5,2,11,12]]>0
            prev_torso = torso_pts[vis_torso].mean(axis=0) if vis_torso.any() else prev_torso

        tin=self.in_len
        return {
            "images_in": torch.stack(imgs[:tin]),   "kps_in":  kp_all[:tin],
            "hm_in":     torch.stack(hms[:tin]),    "w_in":    torch.stack(wts[:tin]),
            "images_out":torch.stack(imgs[tin:]),   "kps_out": kp_all[tin:],
            "hm_out":    torch.stack(hms[tin:]),    "w_out":   torch.stack(wts[tin:]),
            "meta":{"seq":clip["seq"], "frame_ids":clip["fids"].tolist()}
        }
