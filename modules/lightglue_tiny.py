from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

torch.backends.cudnn.deterministic = True


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(
    kpts: torch.Tensor, size: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, head_dim: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, head_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-1)
        # emb.shape == (2, 1, N, 32, 1)
        emb = torch.cat((emb, emb), dim=-1)
        # emb.shape == (2, 1, N, 32, 2)
        emb = emb.reshape(2, 1, 1, -1, self.head_dim)
        return emb


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super(TokenConfidence, self).__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """get confidence tokens"""
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q, k, v) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v)


class SelfBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch = 1
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention()
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        qkv: torch.Tensor = self.Wqkv(x)
        qkv = qkv.reshape(self.batch, -1, self.num_heads, self.head_dim, 3)
        qkv = qkv.transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = self.apply_cached_rotary_emb(encoding, q)
        k = self.apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v)
        # context.shape == (1, 4, N, 64)
        context = context.transpose(1, 2)
        context = context.reshape(self.batch, -1, self.embed_dim)
        message = self.out_proj(context)
        return x + self.ffn(torch.cat((x, message), -1))

    def rotate_half(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(self.batch, self.num_heads, -1, self.head_dim // 2, 2)
        t = torch.stack((-t[..., 1], t[..., 0]), dim=-1)
        t = t.reshape(self.batch, self.num_heads, -1, self.head_dim)
        return t

    def apply_cached_rotary_emb(
        self, freqs: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return (t * freqs[0]) + (self.rotate_half(t) * freqs[1])


class CrossBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch = 1
        self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.inner_attn = Attention() # Q, K, V dot product
        self.to_out = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor]:
        qk0, qk1 = map(self.to_qk, (x0, x1))
        v0, v1 = map(self.to_v, (x0, x1))
        qk0, qk1, v0, v1 = map(
            lambda t: t.reshape(
                self.batch, -1, self.num_heads, self.head_dim
            ).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )

        m0 = self.inner_attn(qk0, qk1, v1)
        m1 = self.inner_attn(qk1, qk0, v0)

        m0, m1 = map(
            lambda t: t.transpose(1, 2).reshape(self.batch, -1, self.embed_dim),
            (m0, m1),
        )
        m0, m1 = map(self.to_out, (m0, m1))
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.self_attn = SelfBlock(embed_dim, num_heads)
        self.cross_attn = CrossBlock(embed_dim, num_heads)

    def forward(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        encoding0: torch.Tensor,
        encoding1: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        desc0 = self.self_attn(desc0, encoding0)
        desc1 = self.self_attn(desc1, encoding1)
        return self.cross_attn(desc0, desc1)


def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim, 1)
    scores = scores0 + scores1 + certainties
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super(MatchAssignment, self).__init__()
        self.dim = dim
        self.scale = dim**0.25
        self.final_proj = nn.Linear(dim, dim, bias=True)
        self.matchability = nn.Linear(dim, 1, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> torch.Tensor:
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = map(self.final_proj, (desc0, desc1))
        mdesc0, mdesc1 = map(lambda t: t / self.scale, (mdesc0, mdesc1))
        sim = mdesc0 @ mdesc1.transpose(1, 2)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


# def filter_matches(scores: torch.Tensor, th: float):
#     """obtain matches from a log assignment matrix [BxMxN]"""
#     max0 = torch.topk(scores, k=1, dim=2, sorted=False)  # scores.max(2)
#     max1 = torch.topk(scores, k=1, dim=1, sorted=False)  # scores.max(1)
#     m0, m1 = max0.indices[:, :, 0], max1.indices[:, 0, :]
#     indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
#     # indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
#     mutual0 = indices0 == m1.gather(1, m0)
#     # mutual1 = indices1 == m0.gather(1, m1)
#     max0_exp = max0.values[:, :, 0].exp()
#     zero = max0_exp.new_tensor(0)
#     mscores0 = torch.where(mutual0, max0_exp, zero)
#     # mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
#     valid0 = mscores0 > th
#     # valid1 = mutual1 & valid0.gather(1, m1)
#     # m0 = torch.where(valid0, m0, -1)
#     # m1 = torch.where(valid1, m1, -1)
#     # return m0, m1, mscores0, mscores1

#     m_indices_0 = indices0[valid0]
#     m_indices_1 = m0[0][m_indices_0]

#     matches = torch.stack([m_indices_0, m_indices_1], -1)
#     mscores = mscores0[0][m_indices_0]
#     return matches, mscores

def filter_matches(scores: torch.Tensor):
    """obtain matches from a log assignment matrix [BxMxN]"""
    max0 = torch.topk(scores, k=1, dim=2, sorted=False)  # scores.max(2)
    max1 = torch.topk(scores, k=1, dim=1, sorted=False)  # scores.max(1)
    m0, m1 = max0.indices[:, :, 0], max1.indices[:, 0, :]
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]

    mutual0 = indices0 == m1.gather(1, m0)
    max0_exp = max0.values[:, :, 0].exp()

    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)

    m_indices_0 = indices0[mutual0]
    m_indices_1 = m0[0][m_indices_0]

    matches = torch.stack([m_indices_0, m_indices_1], -1)
    mscores = mscores0[0][m_indices_0]
    return matches, mscores


class LightGlue(nn.Module):
    default_conf = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 256,
        "n_layers": 9,
        "num_heads": 4,
        "filter_threshold": 0.1,  # match threshold
        "depth_confidence": -1,  # -1 is no early stopping, recommend: 0.95
        "width_confidence": -1,  # -1 is no point pruning, recommend: 0.99
        "weights": None,
    }

    # lighterglue L3
    _l3_conf_xfeat = {
        "name": "xfeat",
        "n_layers": 3,
        "num_heads": 1,
        "input_dim": 64,
        "descriptor_dim": 96,
        "add_scale_ori": False,
        "add_laf": False,  # for KeyNetAffNetHardNet
        "scale_coef": 1.0,  # to compensate for the SIFT scale bigger than KeyNet
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        # "width_confidence": 0.95,  # point pruning, disable with -1
        "width_confidence": -1,  # disabled because onnx is not supported dynamic control flow
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
    }

    version = "v0.1_arxiv"
    url = "https://github.com/cvg/LightGlue/releases/download/{}/{}_lightglue.pth"

    features = {
        "superpoint": ("superpoint_lightglue", 256),
        "disk": ("disk_lightglue", 128),
        "xfeat": ("xfeat_lightglue", 64),
    }

    def __init__(self, features="superpoint", **conf) -> None:
        super().__init__()
        self.conf = {**self.default_conf, **conf}
        if features is not None:
            assert features in self.features
            self.conf["weights"], self.conf["input_dim"] = self.features[features]
        self.conf = conf = SimpleNamespace(**self.conf)

        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(2, head_dim)

        # 1, 3, 96 for l3 lighterglue with xfeat
        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim

        self.transformers = nn.ModuleList([TransformerLayer(d, h) for _ in range(n)])

        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])

        self.token_confidence = nn.ModuleList(
            [TokenConfidence(d) for _ in range(n - 1)]
        )
        self.register_buffer(
            "confidence_thresholds",
            torch.Tensor([self.confidence_threshold(i) for i in range(n)]),
        )

        state_dict = None
        if features is not None:
            fname = f"{conf.weights}_{self.version}.pth".replace(".", "-")
            state_dict = torch.hub.load_state_dict_from_url(
                self.url.format(self.version, features), file_name=fname
            )
        elif conf.weights is not None:
            path = Path(__file__).parent
            path = path / "weights/{}.pth".format(self.conf.weights)
            state_dict = torch.load(str(path), map_location="cpu")

        if state_dict is not None:
            # rename old state dict entries
            for i in range(n):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)

        print("Loaded LightGlue model")

    def forward(
        self,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
    ):
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape

        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)

        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        for i in range(self.conf.n_layers):
            # self+cross attention
            desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1)
            if i == self.conf.n_layers - 1:
                continue  # no early stopping or adaptive width at last layer

        scores = self.log_assignment[i](desc0, desc1)
        matches, mscores = filter_matches(scores)
        return matches, mscores

    def confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.conf.n_layers)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(
        self,
        confidences: Optional[torch.Tensor],
        scores: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > (1 - self.conf.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def check_if_stop(
        self,
        confidences0: torch.Tensor,
        confidences1: torch.Tensor,
        layer_index: int,
        num_points: int,
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.conf.depth_confidence
