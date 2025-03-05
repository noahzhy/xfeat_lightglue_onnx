
# from kornia.feature.lightglue import LightGlue
from modules.lightglue import LightGlue
from torch import nn
import numpy as np
import torch
import os


class LighterGlue(nn.Module):
    """
        Lighter version of LightGlue :)
    """

    default_conf_xfeat = {
        "name": "xfeat",  # just for interfacing
        # input descriptor dimension (autoselected from weights)
        "input_dim": 64,
        "descriptor_dim": 96,
        "add_scale_ori": False,
        "add_laf": False,  # for KeyNetAffNetHardNet
        "scale_coef": 1.0,  # to compensate for the SIFT scale bigger than KeyNet
        "n_layers": 6,
        "num_heads": 1,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        # "width_confidence": 0.95,  # point pruning, disable with -1
        "width_confidence": -1,  # disabled because onnx is not supported dynamic control flow
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
    }

    def __init__(self,
        n_layers=6,
        weights=os.path.abspath(os.path.dirname(__file__)) + '/../weights/xfeat-lighterglue.pt'
    ):
        super().__init__()
        self.default_conf_xfeat['n_layers'] = n_layers
        LightGlue.default_conf = self.default_conf_xfeat
        self.net = LightGlue(None)
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if os.path.exists(weights):
            state_dict = torch.load(weights, map_location=self.dev)
        else:
            state_dict = torch.hub.load_state_dict_from_url(
                "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat-lighterglue.pt")

        # rename old state dict entries
        for i in range(self.net.conf.n_layers):
            pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            state_dict = {k.replace('matcher.', ''): v for k, v in state_dict.items()}

        self.net.load_state_dict(state_dict, strict=False)
        self.net.to(self.dev)

    @torch.inference_mode()
    def forward(self, kpt0, kpt1, desc0, desc1):
        result = self.net(kpt0, kpt1, desc0, desc1)
        return result


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LighterGlue(n_layers=3).eval()
    # n_kpts = 1024
    n_kpts = 512
    kpt0 = torch.randn(1, n_kpts, 2).to(device)
    kpt1 = torch.randn(1, n_kpts, 2).to(device)
    desc0 = torch.randn(1, n_kpts, 64).to(device)
    desc1 = torch.randn(1, n_kpts, 64).to(device)

    # force equal
    kpt0, desc0 = kpt1, desc1

    outputs = model(kpt0, kpt1, desc0, desc1)
    matchers = outputs[0]
    scores = outputs[1]
    # matchers = matchers[scores > 0.1]
    print(f'scores: {scores.shape}')
    print(f'matchers: {matchers.shape}') # shape: (n, 2), n for randn should be 0-5

    # calculate flops and params
    from thop import profile
    flops, params = profile(model, inputs=(kpt0, kpt1, desc0, desc1))
    # GFLOPS and M Params
    print(f'GFLOPS: {flops / 1e9:.3f}')
    print(f'M Params: {params / 1e6:.3f}')
