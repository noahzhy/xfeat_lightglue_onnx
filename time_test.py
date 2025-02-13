import os, glob, sys, random
from time import perf_counter

import cv2
import torch
import numpy as np
import onnxruntime as ort
from PIL import Image

# set seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# onnx inference
def inference_onnx_model(model_path, img_path, target_size=(480, 640)):
    img = cv2.imread(img_path)
    resize_img = img = cv2.resize(img, target_size)
    img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()

    # to numpy
    img = img.numpy()
    # img = np.transpose(img, (2, 0, 1))
    # img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    # output_name = session.get_outputs()[0].name
    # Run the model
    output = session.run(None, {input_name: img})
    for i in output:
        print(i.shape)
    return output, resize_img


def test_onnx_model_speed(model_path, input_shape, warm_up=20, test=200, force_cpu=True):
    # Set ONNX Runtime options for better performance
    options = ort.SessionOptions()
    options.intra_op_num_threads = os.cpu_count()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # ort devices check
    print(f'Available devices: {ort.get_device()}')

    # Create session with optimized options
    provider_options = ['CPUExecutionProvider'] if force_cpu else ort.get_available_providers()
    # check providers
    print(f'Available providers: {provider_options}')
    session = ort.InferenceSession(model_path, options, providers=provider_options)
    
    # Pre-allocate input data array
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_data = np.random.random(input_shape).astype(np.float32)

    # Warm up phase
    for _ in range(warm_up):
        session.run([output_name], {input_name: input_data})

    # Test phase with timing
    times = []
    for _ in range(test):
        start = perf_counter()
        session.run([output_name], {input_name: input_data})
        times.append(perf_counter() - start)

    # Calculate statistics
    # sort the times and remove the first 10% and last 10% of the times
    times = np.sort(times)[int(test * 0.2):int(test * 0.8)]
    average_time = np.mean(times)
    print(f'Average time: {average_time * 1000:.2f} ms')
    print(f'Min time: {min(times)*1000:.2f} ms')
    print(f'Max time: {max(times)*1000:.2f} ms')


def draw_points(img, points, size=2, color=(255, 0, 0), thickness=-1):
    for p in points:
        cv2.circle(img, tuple((int(p[0]), int(p[1]))), size, color, thickness)
    return img


if __name__ == '__main__':
    # model_path = 'onnx/xfeat_dense.onnx'
    model_path = 'weights/xfeat.onnx'
    model_path = 'weights/xfeat_dense.onnx'
    # print md5 in windows
    # print(f'Model MD5: {os.popen(f'CertUtil -hashfile {model_path} MD5').read().splitlines()[1]}')
    # model_path = 'onnx/superpoint.onnx'

    test_onnx_model_speed(model_path, (1, 3, 640, 360))
    quit()

    img_path = random.choice(glob.glob('assets/s*.*g'))
    # resize to 640x360
    output, resize_img = inference_onnx_model(model_path, img_path, target_size=(360, 640))
    # print(f'Output shape: {output[0].shape}')

    kpts = output[0]
    scores = output[-1]
    print(f'scores: {scores}')
    print(f'Keypoints shape: {kpts.shape}')
    # filter keypoints
    kpts = kpts[scores > 0.1]
    # to visualize keypoints
    img = draw_points(resize_img, kpts)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
