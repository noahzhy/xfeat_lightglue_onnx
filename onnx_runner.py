import os, glob, sys, random
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from matplotlib import pyplot as plt

# set seed
np.random.seed(0)
random.seed(0)


# wrapper functions to execute time of functions
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        # count time (ms)
        exec_time = (perf_counter() - start_time) * 1000
        print(f"Execution time: {exec_time:.2f} ms")
        return result
    return wrapper


def resize_img(img, height=640, width=360, keep_aspect_ratio=False):
    if keep_aspect_ratio:
        h, w = img.shape[:2]
        if w > h:
            new_w = width
            new_h = int(h * new_w / w)
        else:
            new_h = height
            new_w = int(w * new_h / h)
        img = cv2.resize(img, (new_w, new_h))
        return img
    return cv2.resize(img, (width, height))


def draw_points(img, points, size=2, color=(255, 0, 0), thickness=-1):
    for p in points:
        cv2.circle(img, tuple((int(p[0]), int(p[1]))), size, color, thickness)
    return img


def draw_matches(img0, img1, kpts0, kpts1, matches, scores, threshold=0.1, show_lines=True):
    # Convert images to RGB
    img0, img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB), cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    # hstack images
    vis = np.hstack((img0, img1))
    h0, w0 = img0.shape[:2]

    # Sort matches by score
    sorted_idx = np.argsort(scores)
    matches = matches[sorted_idx]
    scores = scores[sorted_idx]

    # Filter matches by score threshold
    valid_matches = matches[scores > threshold]
    valid_scores = scores[scores > threshold]  # Corresponding valid scores
    norm_score = (valid_scores - valid_scores.min()) / (valid_scores.max() - valid_scores.min())

    pts = []
    # draw lines first
    for m, score in zip(valid_matches, norm_score):
        pt0 = tuple(map(int, kpts0[m[0]]))
        pt1 = tuple(map(int, kpts1[m[1]]))
        color = plt.cm.jet(score)
        dot_size = score * 2.5 + 0.5
        pts.append((pt0, pt1, dot_size, color))
        if show_lines:
            plt.plot((pt0[0], pt1[0] + w0), (pt0[1], pt1[1]), color=color, linewidth=0.1)

    for pt0, pt1, dot_size, color in pts:
        plt.plot(pt0[0], pt0[1], 'o', color=color, markersize=dot_size)
        plt.plot(pt1[0] + w0, pt1[1], 'o', color=color, markersize=dot_size)

    # Display and save the final image
    plt.imshow(vis)
    plt.axis('off')
    plt.savefig('matches.png', bbox_inches='tight', pad_inches=0, dpi=300)
    return valid_matches, valid_scores


def warp_image(img, kpts0, kpts1):
    try:
        # Find homography
        H, _ = cv2.findHomography(kpts0, kpts1, cv2.RANSAC, 5.0)
        if H is None:
            return img
        # Warp image
        img_warped = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
        return img_warped, H
    except Exception as e:
        print(f"Error warping image: {e}")
        return img, H


def show_img(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def normalize_kpts(kpts, im_height, im_width):
    if len(kpts.shape) == 3:
        kpts = kpts.squeeze(0)
    kpts = kpts.copy()
    print('im_height:', im_height, 'im_width:', im_width)
    kpts[:, 0] = kpts[:, 0] / im_width
    kpts[:, 1] = kpts[:, 1] / im_height
    return kpts


class OrtRun:
    def __init__(self, model_path, force_cpu=True, *args, **kwargs):
        self.provider = ['CPUExecutionProvider'] if force_cpu else ort.get_available_providers()
        options = ort.SessionOptions()
        options.intra_op_num_threads = os.cpu_count()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, options, providers=self.provider)
        self.input_name = self.session.get_inputs()
        self.output_name = self.session.get_outputs()

    def preprocess(self, data):
        return data

    def postprocess(self, data):
        return data

    @timeit
    def infer(self, data):
        return self.session.run(None, data)

    def run(self, data):
        data = self.preprocess(data)
        inputs = {}
        for i, inp in enumerate(self.input_name):
            inputs[inp.name] = data[i]

        res = self.infer(inputs)
        return self.postprocess(res)


class FeatExtractor(OrtRun):
    def __init__(self, model_path, force_cpu=True, **kwargs):
        super().__init__(model_path, force_cpu, **kwargs)

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)) / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return [img]

    def postprocess(self, data):
        kpt0, desc0, score0 = data
        # kpt0 = kpt0[score0 > 0.1]
        # desc0 = desc0[score0 > 0.1]
        return kpt0, desc0


class Matcher(OrtRun):
    def __init__(self, model_path, force_cpu=True):
        super().__init__(model_path, force_cpu)

    def preprocess(self, data):
        # add batch dim
        data = [np.expand_dims(d, axis=0) for d in data]
        return data

    def postprocess(self, data):
        matches, scores = data
        if len(matches) == 0:
            return np.array([]), np.array([])
        return matches, scores


if __name__ == '__main__':
    extractor = FeatExtractor('onnx/xfeat_1024_640x360.onnx')
    matcher = Matcher('onnx/lighterglue_L6.onnx')

    im1 = 'assets/hard/image_6.jpg'
    im2 = 'assets/hard/image_7.jpg'
    # im1 = 'assets/001.jpg'
    # im2 = 'assets/002.jpg'
    # im1 = 'assets/003.png'
    # im2 = 'assets/004.png'
    # im1 = 'assets/ref.png'
    # im2 = 'assets/tgt.png'
    im1 = cv2.imread(im1)
    im2 = cv2.imread(im2)

    h, w = 640, 360
    im1 = resize_img(im1, height=h, width=w)
    im2 = resize_img(im2, height=h, width=w)
    print(im1.shape, im2.shape)

    start_time = perf_counter()
    kpt1, desc1 = extractor.run(im1)
    kpt2, desc2 = extractor.run(im2)

    norm_kpt1 = normalize_kpts(kpt1, im_height=im1.shape[0], im_width=im1.shape[1])
    norm_kpt2 = normalize_kpts(kpt2, im_height=im2.shape[0], im_width=im2.shape[1])
    matches, scores = matcher.run((norm_kpt1, norm_kpt2, desc1, desc2))
    print(f'Shape of matches: {matches.shape}, scores: {scores.shape}')
    print(f'Inference time: {perf_counter() - start_time:.2f} s')

    if len(matches) > 0:
        # vis = draw_points(im1, kpt1, size=2)
        # show_img(vis)
        matches, scores = draw_matches(
            im1, im2,
            kpt1, kpt2,
            matches, scores,
            threshold=0.5,
            show_lines=True
        )
        print(f"Found {len(matches)} matches above threshold")
    else:
        print("No matches found")
