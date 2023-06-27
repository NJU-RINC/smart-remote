from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.core import Module, Tensor

INF = 1e9


def mask_border(m: Tensor, b: int, v: Union[Tensor, float, int, bool]) -> None:
    """Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(
    m: Tensor, bd: int, v: Union[Tensor, float, int, bool], p_m0: Tensor, p_m1: Tensor
) -> None:
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd :] = v
        m[b_idx, :, w0 - bd :] = v
        m[b_idx, :, :, h1 - bd :] = v
        m[b_idx, :, :, :, w1 - bd :] = v


def compute_max_candidates(p_m0: Tensor, p_m1: Tensor) -> Tensor:
    """Compute the max candidates of all pairs within a batch.

    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


class CoarseMatching(Module):
    def __init__(self, config: Dict[str, Any], d_size) -> None:
        super().__init__()
        self.config = config
        # general config
        self.thr = config['thr']
        self.border_rm = config['border_rm']
        self.temperature = config['dsmax_temperature']  
        self.d_size = d_size

    def forward(
        self,
        feat_c0: Tensor,
        feat_c1: Tensor,
        data: Dict[str, Tensor],
    ) -> None:
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / self.d_size ** 0.5, [feat_c0, feat_c1])

        # sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
        sim_matrix_orig = torch.matmul(feat_c0, feat_c1.permute((0, 2, 1)))
        sim_matrix = sim_matrix_orig / self.temperature
        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        data.update({'conf_matrix': conf_matrix})

        # predict coarse matches from conf_matrix
        data.update(**self.get_coarse_match(conf_matrix, data))

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix: Tensor, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1],
        }

        # 1. confidence thresholding
        mask : torch.Tensor = conf_matrix > self.thr
        N = conf_matrix.shape[0]
        mask = mask.reshape(N, axes_lengths['h0c'], axes_lengths['w0c'], axes_lengths['h1c'], axes_lengths['w1c'])
        # mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
        #                 **axes_lengths)
        if 'mask0' not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False, data['mask0'], data['mask1'])
        mask = mask.reshape(N, axes_lengths['h0c'] * axes_lengths['w0c'], axes_lengths['h1c'] * axes_lengths['w1c'])
        # rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
        #                 **axes_lengths)

        # 2. mutual nearest
        mask = (
            mask
            * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0])
            * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])
        )

        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
        mask_v, all_j_ids = mask.float().max(dim=2)
        b_ids, i_ids = torch.where(mask_v >= 1.0)
        j_ids = all_j_ids[b_ids, i_ids]

        
        # b_ids, i_ids, j_ids = torch.tensor_split(torch.nonzero(conf_matrix > self.thr), 3, dim=1)
        mconf = conf_matrix[b_ids, i_ids, j_ids]

       
        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # 4. Update with matches in original image resolution
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        # scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        # scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack([i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]], dim=1) * scale
        # mkpts1_c = torch.stack([j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]], dim=1) * scale

        # These matches is the current prediction (for visualization)
        coarse_matches.update(
            {
                # 'gt_mask': mconf == 0,
                # 'm_bids': b_ids[mconf != 0],  # mconf == 0 => gt matches
                'mkpts0_c': mkpts0_c[mconf != 0].to(dtype=conf_matrix.dtype),
                # 'mkpts1_c': mkpts1_c[mconf != 0].to(dtype=conf_matrix.dtype),
                # 'mconf': mconf[mconf != 0],
            }
        )

        return coarse_matches
