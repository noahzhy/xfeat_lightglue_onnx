"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Minimal example of how to use XFeat.
"""

import os, random, glob

import cv2
import tqdm
import torch
import numpy as np
from PIL import Image

# from modules.xfeat import XFeat
from modules.xfeat import XFeat

os.environ['CUDA_VISIBLE_DEVICES'] = '' #Force CPU, comment for GPU

xfeat = XFeat(top_k=2048, detection_threshold=0.05)

#Random input
x = torch.randn(1,3,480,640)

# #Simple inference with batch = 1
# output = xfeat.detectAndCompute(x, top_k = 2048)[0]
# print("----------------")
# print("keypoints: ", output['keypoints'].shape)
# print("descriptors: ", output['descriptors'].shape)
# print("scores: ", output['scores'].shape)
# print("----------------\n")


def draw_points(img, points, size=2, color=(255, 0, 0), thickness=-1):
    for p in points:
        cv2.circle(img, tuple((int(p[0]), int(p[1]))), size, color, thickness)
    return img

# set seed to 0
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

img_path = random.choice(glob.glob(r'D:\projects\xfeat_lightglue_onnx\assets\s*.*g'))
# resize to H:480, W:640 via cv2, at last to tensor
img = cv2.imread(img_path)
resize_img = img = cv2.resize(img, (360, 640))
img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()

# to numpy
img_np = img.numpy()

# compared img and img_np
diff = np.abs(img_np - img[0].numpy())
print(f'Diff: {diff.sum()}')

# output = xfeat.detectAndComputeDense(img, top_k=2048, multiscale=False)
# kpts = output['keypoints'][0]

output = xfeat.detectAndCompute(img)
kpts = output['keypoints']
scores = output['scores']
# print(f'Output shape: {output[0].shape}')
# print(f'scale: {scale}')
# keep only keypoints with score > 0.25
kpts = kpts[scores > 0.1]

img = draw_points(resize_img, kpts)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
