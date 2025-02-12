dependencies = ['torch']
from modules.xfeat import XFeat as _XFeat
import torch

from modules.lighterglue import LighterGlue as _LighterGlue


def XFeat(pretrained=True, top_k=4096, detection_threshold=0.05):
    """
    XFeat model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    weights = None
    if pretrained:
        weights = torch.hub.load_state_dict_from_url("https://github.com/verlab/accelerated_features/raw/main/weights/xfeat.pt")
    
    model = _XFeat(weights, top_k=top_k, detection_threshold=detection_threshold)
    return model


# export the model as an ONNX file
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = XFeat(pretrained=True, top_k=2048).eval().to(device)
    # dynamic_axes = {'input': {2: 'height', 3: 'width'}}
    # dummy_input = torch.randn(1, 3, 640, 360).to(device)
    # torch.onnx.export(
    #     model,
    #     dummy_input,
    #     "xfeat.onnx",
    #     input_names=["input"],
    #     output_names=["output"],
    #     dynamic_axes=dynamic_axes,
    #     opset_version=20,
    #     verbose=True,
    # )
        # result = self.net({
        #     'image0': {
        #         'keypoints': data['keypoints0'],
        #         'descriptors': data['descriptors0'],
        #         'image_size': data['image_size0']},
        #     'image1': {
        #         'keypoints': data['keypoints1'],
        #         'descriptors': data['descriptors1'],
        #         'image_size': data['image_size1']
        #     }})
    model = _LighterGlue().eval().to(device)
    output = model({
        'keypoints0': torch.randn(1, 2048, 2).to(device),
        'keypoints1': torch.randn(1, 2048, 2).to(device),
        'descriptors0': torch.randn(1, 2048, 64).to(device),
        'descriptors1': torch.randn(1, 2048, 64).to(device),
        'image_size0': torch.tensor([640, 480]).to(device),
        'image_size1': torch.tensor([640, 480]).to(device),
    })

    print(output['matches'].shape)
    torch.onnx.export(
        model,
        (
            {
                'keypoints0': torch.randn(1, 2048, 2).to(device),
                'keypoints1': torch.randn(1, 2048, 2).to(device),
                'descriptors0': torch.randn(1, 2048, 64).to(device),
                'descriptors1': torch.randn(1, 2048, 64).to(device),
                'image_size0': torch.tensor([640, 480]).to(device),
                'image_size1': torch.tensor([640, 480]).to(device),
            },
            0.1
        ),
        "lighterglue.onnx",
        input_names=["input", "min_conf"],
        output_names=["output"],
        opset_version=20,
        verbose=True,
    )
