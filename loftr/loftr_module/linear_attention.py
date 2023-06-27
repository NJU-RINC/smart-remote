"""Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-
transformers/blob/master/fast_transformers/attention/linear_attention.py."""

from typing import Optional

import torch
from torch.nn import Dropout, Module


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):
    def __init__(self,  nheads, dim, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps
        self.nheads = nheads
        self.dim = dim

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # v_length = values.size(1)
        # values = values / v_length  # prevent fp16 overflow
        # KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        # Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        # queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow

        # Remove einsums to satisfy TensorRT
        k = K.view(-1, self.dim).unsqueeze(2)
        v = values.view(-1, self.dim).unsqueeze(1)
        kv = torch.bmm(k, v)
        kv = kv.reshape(-1, v_length, self.nheads, self.dim, self.dim)
        kv = kv.sum(dim=1)
        # assert(torch.allclose(KV_t, kv, atol=1e-05))
        KV = kv

        # Z_t = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        k = K.sum(dim=1).unsqueeze(1)
        z = (Q * k).sum(dim=-1)
        z = 1 / (z + self.eps)
        # assert(torch.allclose(Z_t, z, atol=1e-05))
        Z = z

        # queried_values_t = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length
        q = Q.unsqueeze(3)
        kv = KV.unsqueeze(1)
        qkv = torch.matmul(q, kv)
        qkv = qkv.squeeze(3)
        qv = qkv * Z.unsqueeze(3)
        qv *= v_length
        # assert(torch.allclose(queried_values_t, qv, atol=1e-05))
        queried_values = qv

        return queried_values.contiguous()
