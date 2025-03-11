import os, glob, sys, random
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# set seed
np.random.seed(0)
random.seed(0)


def inference_onnx_model(model_path, img_path, target_size=(480, 640)):
    img = cv2.imread(img_path)
    resize_img = cv2.resize(img, target_size)
    img_rgb = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
    # Prepare input: convert to NCHW format
    img_input = np.transpose(img_rgb, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
    # Run inference
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_input})
    return output, resize_img


def test_onnx_model_speed(model_path, inputs, warm_up=20, test=200, force_cpu=True):
    # Set ONNX Runtime options for better performance
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.intra_op_num_threads = os.cpu_count()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # ort devices check
    print(f'Available devices: {ort.get_device()}')

    # Create session with optimized options
    provider_options = ['CPUExecutionProvider'] if force_cpu else ort.get_available_providers()
    # check providers
    print(f'Available providers: {provider_options}')
    session = ort.InferenceSession(model_path, options, providers=provider_options)

    # run one inference to get the output shape
    session.run(None, inputs)
    quit()

    # # Pre-allocate input data array
    # input_name = session.get_inputs()
    # output_name = session.get_outputs()

    # Warm up phase
    for _ in range(warm_up):
        session.run(None, inputs)

    # Test phase with timing
    times = []
    for _ in range(test):
        start = perf_counter()
        session.run(None, inputs)
        times.append(perf_counter() - start)

    # Calculate statistics
    # sort the times and remove the first 10% and last 10% of the times
    times = np.sort(times)[int(test * 0.3):int(test * 0.7)]
    average_time = np.mean(times)
    print(f'Average time: {average_time * 1000:.2f} ms')
    print(f'Min time: {min(times)*1000:.2f} ms')
    print(f'Max time: {max(times)*1000:.2f} ms')


def draw_points(img, points, size=2, color=(255, 0, 0), thickness=-1):
    for p in points:
        cv2.circle(img, tuple((int(p[0]), int(p[1]))), size, color, thickness)
    return img


if __name__ == '__main__':
    model_path = 'onnx/xfeat_2048_640x360.onnx'
    # model_path = 'onnx/xfeat_1024.onnx'
    # model_path = 'onnx/xfeat_2048.onnx'
    input_shapes = {
        'images': np.random.random((1, 3, 640, 360)).astype(np.float32)
    }

    model_path = 'onnx/lighterglue_L3.onnx'
    n_kpts = 1024
    kpt0 = np.random.random((1, n_kpts, 2)).astype(np.float32)
    desc0 = np.random.random((1, n_kpts, 64)).astype(np.float32)
    kpt1 = np.random.permutation(kpt0)
    desc1 = np.random.permutation(desc0)
    input_shapes = {
        'kpts0': kpt0,
        'kpts1': kpt0,
        'desc0': desc0,
        'desc1': desc0
    }

    test_onnx_model_speed(model_path, input_shapes)
    quit()

    img_path = random.choice(glob.glob('assets/0*.*g'))
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
