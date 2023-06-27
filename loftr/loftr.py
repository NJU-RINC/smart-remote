from typing import Any, Dict, Optional, Union

import torch

from kornia.core import Module, Tensor
from kornia.geometry import resize
from kornia.utils.helpers import map_location_to_cpu

from .backbone import build_backbone
from .loftr_module import LocalFeatureTransformer
from .utils.coarse_matching import CoarseMatching
from .utils.position_encoding import PositionEncodingSine

urls: Dict[str, str] = {}
urls["outdoor"] = "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt"
urls["indoor_new"] = "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor_ds_new.ckpt"
urls["indoor"] = "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor.ckpt"

# Comments: the config below is the one corresponding to the pretrained models
# Some do not change there anything, unless you want to retrain it.

default_cfg = {
    'backbone_type': 'ResNetFPN',
    'resolution': (8, 2),
    'fine_window_size': 5,
    'fine_concat_coarse_feat': True,
    'resnetfpn': {'initial_dim': 128, 'block_dims': [128, 196, 256]},
    'coarse': {
        'd_model': 256,
        'd_ffn': 256,
        'nhead': 8,
        'layer_names': ['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross'],
        'attention': 'linear',
        'temp_bug_fix': False,
    },
    'match_coarse': {
        'thr': 0.1,
        'border_rm': 2,
        'match_type': 'dual_softmax',
        'dsmax_temperature': 0.1,
        'skh_iters': 3,
        'skh_init_bin_score': 1.0,
        'skh_prefilter': True,
        'train_coarse_percent': 0.4,
        'train_pad_num_gt_min': 200,
    },
    'fine': {'d_model': 128, 'd_ffn': 128, 'nhead': 8, 'layer_names': ['self', 'cross'], 'attention': 'linear'},
}


class LoFTR(Module):
    r"""Module, which finds correspondences between two images.

    This is based on the original code from paper "LoFTR: Detector-Free Local
    Feature Matching with Transformers". See :cite:`LoFTR2021` for more details.

    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.

    Args:
        config: Dict with initiliazation parameters. Do not pass it, unless you know what you are doing`.
        pretrained: Download and set pretrained weights to the model. Options: 'outdoor', 'indoor'.
                    'outdoor' is trained on the MegaDepth dataset and 'indoor'
                    on the ScanNet.

    Returns:
        Dictionary with image correspondences and confidence scores.

    Example:
        >>> img1 = torch.rand(1, 1, 320, 200)
        >>> img2 = torch.rand(1, 1, 128, 128)
        >>> input = {"image0": img1, "image1": img2}
        >>> loftr = LoFTR('outdoor')
        >>> out = loftr(input)
    """

    def __init__(self, pretrained: Optional[str] = 'outdoor', config: Dict[str, Any] = default_cfg):
        super().__init__()
        # Misc
        self.config = config
        if pretrained == 'indoor_new':
            self.config['coarse']['temp_bug_fix'] = True
        # Modules
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'], temp_bug_fix=config['coarse']['temp_bug_fix']
        )
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'], config['coarse']['d_model'])
        # self.fine_preprocess = FinePreprocess(config)
        # self.loftr_fine = LocalFeatureTransformer(config["fine"])
        # self.fine_matching = FineMatching()
        self.pretrained = pretrained
        if pretrained is not None:
            if pretrained not in urls.keys():
                raise ValueError(f"pretrained should be None or one of {urls.keys()}")

            pretrained_dict = torch.hub.load_state_dict_from_url(urls[pretrained], map_location=map_location_to_cpu)
            self.load_state_dict(pretrained_dict['state_dict'])
        self.eval()

    def forward(self, image0, image1) -> Dict[str, Tensor]:
        """
        Args:
            data: dictionary containing the input data in the following format:

        Keyword Args:
            image0: left image with shape :math:`(N, 1, H1, W1)`.
            image1: right image with shape :math:`(N, 1, H2, W2)`.
            mask0 (optional): left image mask. '0' indicates a padded position :math:`(N, H1, W1)`.
            mask1 (optional): right image mask. '0' indicates a padded position :math:`(N, H2, W2)`.

        Returns:
            - ``keypoints0``, matching keypoints from image0 :math:`(NC, 2)`.
            - ``keypoints1``, matching keypoints from image1 :math:`(NC, 2)`.
            - ``confidence``, confidence score [0, 1] :math:`(NC)`.
            - ``batch_indexes``, batch indexes for the keypoints and lafs :math:`(NC)`.
        """
        # 1. Local Feature CNN
        _data: Dict[str, Union[Tensor, int, torch.Size]] = {
            'bs': image0.size(0),
            'hw0_i': image0.shape[2:],
            'hw1_i': image1.shape[2:],
        }

        feats_c = self.backbone(torch.cat([image0, image1], dim=0))
        feat_c0, feat_c1 = feats_c.split(_data['bs'])


        _data.update(
            {
                'hw0_c': feat_c0.shape[2:],
                'hw1_c': feat_c1.shape[2:],
            }
        )

        feat_c0 = torch.flatten(self.pos_encoding(feat_c0), 2, 3).permute(0, 2, 1)
        feat_c1 = torch.flatten(self.pos_encoding(feat_c1), 2, 3).permute(0, 2, 1)

        mask_c0 = mask_c1 = None  # mask is useful in training
      
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, _data)

        # rename_keys: Dict[str, str] = {
        #     "mkpts0_c": 'keypoints0',
        #     "mkpts1_c": 'keypoints1',
        #     "mconf": 'confidence',
        #     # "b_ids": 'batch_indexes',
        # }
        # out: Dict[str, Tensor] = {}
        # for k, v in rename_keys.items():
        #     _d = _data[k]
        #     if isinstance(_d, Tensor):
        #         out[v] = _d
        #     else:
        #         raise TypeError(f'Expected Tensor for item `{k}`. Gotcha {type(_d)}')
        return _data["mkpts0_c"]

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, strict=False, *args, **kwargs)
