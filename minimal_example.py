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

from modules.xfeat import XFeat

os.environ['CUDA_VISIBLE_DEVICES'] = '' #Force CPU, comment for GPU

xfeat = XFeat()

#Random input
x = torch.randn(1,3,480,640)

#Simple inference with batch = 1
output = xfeat.detectAndCompute(x, top_k = 2048)[0]
print("----------------")
print("keypoints: ", output['keypoints'].shape)
print("descriptors: ", output['descriptors'].shape)
print("scores: ", output['scores'].shape)
print("----------------\n")


def draw_points(img, points, size=2, color=(255, 0, 0), thickness=-1):
    for p in points:
        cv2.circle(img, tuple((int(p[0]), int(p[1]))), size, color, thickness)
    return img




img_path = random.choice(glob.glob(r'D:\projects\xfeat_lightglue_onnx\assets\s*.*g'))
# resize to 640x360
img = Image.open(img_path).convert('RGB').resize((640, 360))
# transpose to CHW
w, h = img.size
img = np.array(img).astype(np.float32)
img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
img = torch.Tensor(img)
print(img.shape)

output = xfeat.detectAndCompute(img, top_k = 2048)

# print(f'Output shape: {output[0].shape}')
kpts = output[0]['keypoints']
scale = output[0]['scores']
print(f'scale: {scale}')
# to visualize keypoints
img = cv2.imread(img_path)
img = draw_points(img, kpts)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

quit()

x = torch.randn(1,3,480,640)
# Stress test
for i in tqdm.tqdm(range(100), desc="Stress test on VGA resolution"):
	output = xfeat.detectAndCompute(x, top_k = 2048)

# Batched mode
x = torch.randn(4,3,480,640)
outputs = xfeat.detectAndCompute(x, top_k = 2048)
print("# detected features on each batch item:", [len(o['keypoints']) for o in outputs])

# Match two images with sparse features
x1 = torch.randn(1,3,480,640)
x2 = torch.randn(1,3,480,640)
mkpts_0, mkpts_1 = xfeat.match_xfeat(x1, x2)

# Match two images with semi-dense approach -- batched mode with batch size 4
x1 = torch.randn(4,3,480,640)
x2 = torch.randn(4,3,480,640)
matches_list = xfeat.match_xfeat_star(x1, x2)
print(matches_list[0].shape)
