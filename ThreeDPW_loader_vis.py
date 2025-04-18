# ThreeDPW_loader_vis.py  – clip + heat‑map viewer

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ThreeDPW_loader import ThreeDPWSequenceDataset


def _denorm(t):
    im = t.numpy().transpose(1,2,0)
    return np.clip(im*[0.229,0.224,0.225] + [0.485,0.456,0.406], 0, 1)


def _view(img, kp, hm, ttl):
    n, cols = hm.shape[0], 6
    rows = int(np.ceil(n/cols))
    fig = plt.figure(figsize=(5+3*cols,3*rows))
    gs  = gridspec.GridSpec(rows, cols+1, wspace=0.05, hspace=0.05, figure=fig)

    ax0 = fig.add_subplot(gs[:,0]); ax0.imshow(img); ax0.set_title(ttl); ax0.axis("off")
    for x,y,v in kp:
        if v>0: ax0.plot(x,y,"ro",ms=3)

    for j in range(n):
        r,c = divmod(j,cols)
        ax = fig.add_subplot(gs[r,c+1]); ax.imshow(hm[j],cmap="hot"); ax.axis("off"); ax.set_title(f"J{j}",fontsize=6)
    plt.tight_layout(); plt.show()


def draw_clip(ds, idx=0):
    it = ds[idx]
    for i in range(it["images_in"].shape[0]):
        _view(_denorm(it["images_in"][i]), it["kps_in"][i].numpy(), it["hm_in"][i].numpy(), f"Input {i}")
    for i in range(it["images_out"].shape[0]):
        _view(_denorm(it["images_out"][i]), it["kps_out"][i].numpy(), it["hm_out"][i].numpy(), f"Output {i}")


if __name__ == "__main__":
    ds = ThreeDPWSequenceDataset(
        seq_dir   = "/home/klingjac/DeepRob/final_project/3DPW/sequenceFiles/sequenceFiles/test",
        img_root  = "/home/klingjac/DeepRob/final_project/3DPW/imageFiles/imageFiles",
        input_len = 2, output_len = 2, frame_gap = 10,
        image_size= (192,256)
    )
    draw_clip(ds, 0)
