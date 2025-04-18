
import pickle, pathlib, numpy as np, cv2, matplotlib.pyplot as plt

pkl = pathlib.Path("/home/klingjac/DeepRob/final_project/3DPW/sequenceFiles/sequenceFiles/test/downtown_cafe_00.pkl")
seq = pickle.load(pkl.open("rb"), encoding="latin1")

img_root = pathlib.Path("/home/klingjac/DeepRob/final_project/3DPW/imageFiles/imageFiles")
fid0     = seq["img_frame_ids"][0]
img      = cv2.cvtColor(
             cv2.imread(str(img_root/seq["sequence"]/f"image_{fid0:05d}.jpg")),
             cv2.COLOR_BGR2RGB)

plt.imshow(img); plt.axis("off")
print(len(seq["img_frame_ids"]))
print(seq["poses2d"])
assert len(seq["img_frame_ids"]) == len(seq["poses2d"])

for aid, kp18 in enumerate(seq["poses2d"]):           # iterate actors
    kp = np.asarray(kp18[0])                          # frame 0
    if kp.shape[0] == 3: kp = kp.T
    vis = kp[:,2] > 0
    if vis.any():
        x, y = kp[vis,0].mean(), kp[vis,1].mean()
        plt.text(x, y, f"A{aid}", color="yellow", fontsize=12)
plt.show()
