from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords,labels,ax,marker_size=375):
    pos_points=coords[labels==1]
    neg_points =coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
def show_box(box,ax):
    x0, y0=box[0], box[1]
    w,h=box[2]-box[0], box[3]-box[1]
    ax.add_patch(plt.Rectangle((x0,y0),w,h,edgecolor='green',facecolor=(0,0,0,0),lw=2))

sam_checkpoint = r"X:\SAMed_fortest\checkpoints\sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
image_dir = r"X:\SAMed_fortest\RDD2022_CN_000227.jpg"
image = cv2.imread(image_dir)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


sam, _ = sam_model_registry[model_type](image_size=512, num_classes=1, checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

predictor.set_image(image)
box = np.array([1,221,207,406])
dot = np.array([[62, 295],[161,315]])
# dot = np.array([[17.96,390.98],[170.1,245.1],[82.3,332.9],[62, 295],[161,315]])
# label = np.array([1,1,1,0,0])
label = np.array([0,0])
mask, score, logist = predictor.predict(
    point_coords=dot,
    point_labels=label,
    box=box[None, :],
    multimask_output=False,
)

print(score)
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(image)
# show_box(box, plt.gca())
show_mask(mask[1], plt.gca())
show_points(dot, label, plt.gca())
ax.axis('on')
plt.show()

# cv2.imshow("mask", mask[1,:].astype(np.float64))
# cv2.waitKey(0)
# cv2.destroyAllWindows()