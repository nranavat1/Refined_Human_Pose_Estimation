#!/usr/bin/env python3
"""
Write PNGs to /tmp showing
  • red  = raw 2‑D detections   (poses2d)
  • blue = SMPL joints projected with camera extrinsics/intrinsics

Example
-------
python quick_debug_3dpw_proj.py \\
        --pkl      ~/3DPW/sequenceFiles/test/downtown_cafe_00.pkl \\
        --imgroot  ~/3DPW/imageFiles/imageFiles \\
        --actor    1    --start 100 --nframes 40
"""

import argparse, pickle, pathlib, cv2, numpy as np

# ---------------------------------------------------------------- helpers
def load_poses2d(seq):
    """Return poses2d as numpy array (A,T,18,3)."""
    p = np.asarray(seq["poses2d"], np.float32)
    if p.ndim == 3:                      # (T,*,*) → add actor dim
        p = p[:, None, ...].transpose(1, 0, 2, 3)
    if p.shape[-2:] == (3, 18):          # (A,T,3,18) → (A,T,18,3)
        p = p.transpose(0, 1, 3, 2)
    return p

def reshapeK(k):
    """Return K as 3×3 matrix for any vector/matrix input."""
    k = np.asarray(k, np.float32).ravel()
    if k.size == 9:                 # flattened 3×3
        return k.reshape(3, 3)
    if k.size == 4:                 # fx, fy, cx, cy
        fx, fy, cx, cy = k
    elif k.size == 3:               # fx, cx, cy   (assume fy = fx)
        fx, cx, cy = k
        fy = fx
    else:
        raise ValueError("Unsupported intrinsics length:", k.size)
    return np.array([[fx, 0,  cx],
                     [0,  fy, cy],
                     [0,   0,  1]], np.float32)

def project_smpl24(seq):
    """Return projected SMPL joints as (A,T,24,3) with (u,v,1)."""
    JP  = seq["jointPositions"]                     # list[A] of (T,24*3)
    CP  = np.asarray(seq["cam_poses"], np.float32)  # (T,4,4)
    Kin = seq["cam_intrinsics"]                     # single K or per‑frame

    A = len(JP)
    T = JP[0].shape[0]
    out = np.zeros((A, T, 24, 3), np.float32)

    # Decide if K varies per frame
    per_frame = (
        isinstance(Kin, (list, tuple)) and len(Kin) == T
    ) or (
        isinstance(Kin, np.ndarray) and Kin.ndim == 2 and Kin.shape[0] == T
    )

    for t in range(T):
        K = reshapeK(Kin[t] if per_frame else Kin)
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

        for a in range(A):
            Xw = JP[a][t].reshape(24, 3)                        # world
            homog = np.hstack([Xw, np.ones((24,1), np.float32)])# (24,4)
            Xc = (CP[t] @ homog.T).T                           # camera
            u = fx * Xc[:,0] / Xc[:,2] + cx
            v = fy * Xc[:,1] / Xc[:,2] + cy
            out[a, t, :, :2] = np.stack([u, v], 1)
            out[a, t, :,  2] = 1.0                             # vis flag
    return out

# ---------------------------------------------------------------- script
arg = argparse.ArgumentParser()
arg.add_argument("--pkl",      required=True)
arg.add_argument("--imgroot",  required=True)
arg.add_argument("--actor",    type=int, default=0)
arg.add_argument("--start",    type=int, default=0)
arg.add_argument("--nframes",  type=int, default=100)
opt = arg.parse_args()

seq   = pickle.load(open(opt.pkl, "rb"), encoding="latin1")
p2d   = load_poses2d(seq)
proj  = project_smpl24(seq)

A, T  = p2d.shape[:2]
aid   = min(opt.actor, A-1)
t0    = opt.start
N     = min(opt.nframes, T - t0)

root  = pathlib.Path(opt.imgroot) / seq["sequence"]
RED, BLUE = (0,0,255), (255,0,0)

for k in range(N):
    t   = t0 + k
    fid = seq["img_frame_ids"][t]
    img = cv2.imread(str(root / f"image_{fid:05d}.jpg"))

    # ---- red detections ----
    det = p2d[aid, t]
    if det.shape[0] == 3: det = det.T
    for x, y, v in det:
        if v > 0:
            cv2.circle(img, (int(x), int(y)), 4, RED, -1)

    # ---- blue SMPL joints ---
    for u, v, _ in proj[aid, t]:
        cv2.circle(img, (int(u), int(v)), 3, BLUE, -1)

    out = f"/tmp/3dpw_cmp_{fid:05d}.png"
    cv2.imwrite(out, img)
    print("wrote", out)

print("\nOpen the PNGs in /tmp to compare red (poses2d) vs blue (SMPL projection).")
