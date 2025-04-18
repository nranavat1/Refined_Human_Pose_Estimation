"""
Write raw 3DPW JPEGs with every actor’s 2‑D key‑points burned into the
pixels.  Open the PNGs and zoom—you will *see* the dots.

Usage (adjust paths):
python quick_debug_3dpw.py \
    --pkl      ~/3DPW/sequenceFiles/test/downtown_cafe_00.pkl \
    --imgroot  ~/3DPW/imageFiles/imageFiles \
    --start    100   --nframes 10
"""

import argparse, pickle, pathlib, cv2, numpy as np

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument("--pkl",     required=True)
ap.add_argument("--imgroot", required=True)
ap.add_argument("--start",   type=int, default=0)
ap.add_argument("--nframes", type=int, default=100)
args = ap.parse_args()

# ---------- load sequence ----------
seq = pickle.load(open(args.pkl, "rb"), encoding="latin1")
frames = seq["img_frame_ids"]
poses  = np.asarray(seq["poses2d"], dtype=np.float32)

# normalise pose shape → (A,T,18,3)
if poses.ndim == 3: poses = poses[:, None, ...].transpose(1,0,2,3)
if poses.shape[-2:] == (3,18): poses = poses.transpose(0,1,3,2)
A, T = poses.shape[:2]

root = pathlib.Path(args.imgroot) / seq["sequence"]

# ---------- colours per actor ----------
COL = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255)]

for k in range(args.nframes):
    t   = args.start + k
    if t >= T: break
    fid = frames[t]
    img = cv2.imread(str(root / f"image_{fid:05d}.jpg"))

    for aid in range(A):
        kp18 = poses[aid, t]
        if kp18.shape[0] == 3: kp18 = kp18.T
        for x, y, v in kp18:
            if v > 0:
                cv2.circle(img, (int(x), int(y)), 4, COL[aid % len(COL)], -1)

    out = f"/tmp/3dpw_debug_{t:04d}.png"
    cv2.imwrite(out, img)
    print("wrote", out)

print("\nOpen any of the /tmp/3dpw_debug_*.png files to see dots.")
