import numpy as np
import torch
import torch.nn as nn

from network.dwt import DWT_3D
from network.nn import linear, timestep_embedding, UEncoder, UDecoder


class DWTDenoiserNet(nn.Module):
    def __init__(
        self,
        diff_channels,
        cond_channels,
        base_channels,
        out_channels,
        channel_mult,
        num_res_blocks=2,
        attn_res=()
    ):
        super().__init__()

        self.base_channels = base_channels
        time_embed_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            linear(base_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.dwt_layer1 = DWT_3D('haar')
        self.dwt_layer2 = DWT_3D('haar')
        self.dwt_layer3 = DWT_3D('haar')
        self.cond_en = UEncoder(cond_channels, base_channels, num_res_blocks, channel_mult, attn_res)
        self.diff_en = UEncoder(diff_channels, base_channels, num_res_blocks, channel_mult, attn_res)
        self.decoder = UDecoder(diff_channels, base_channels, out_channels, num_res_blocks, channel_mult, attn_res)

    @staticmethod
    def energy_fusion(sub_bands):
        weights, sum_weights = [], 0
        for band in sub_bands:
            weights.append((band ** 2).sum())
            sum_weights += weights[-1]
        weights = [torch.sqrt(1 - (i / sum_weights) ** 2) for i in weights]

        rst = torch.zeros_like(sub_bands[0])
        for s, w in zip(sub_bands, weights):
            rst += s * w
        return rst

    def forward(self, x, timesteps, **kwargs):
        cond = kwargs.get('cond', None)
        emb = self.time_embed(timestep_embedding(timesteps, self.base_channels))

        c_h, c_hs = self.cond_en(cond, emb)
        d_h, d_hs = self.diff_en(x, emb)

        h = c_h + d_h
        hs = [i + j for i, j in zip(c_hs, d_hs)]
        hs[9] += self.energy_fusion(self.dwt_layer1(d_hs[6]))
        hs[6] += self.energy_fusion(self.dwt_layer2(d_hs[4]))
        hs[3] += self.energy_fusion(self.dwt_layer3(d_hs[1]))

        return self.decoder(h, hs, emb)


if __name__ == '__main__':
    ch_mult = [1, 2, 2, 4, 4]
    atn_res = [8, 16]
    device = torch.device('cpu')

    _x = torch.randn([4, 1, 64, 64, 48], device=device)
    _c = torch.randn([4, 3, 64, 64, 48], device=device)
    _t = torch.ones([4,], device=device) * 500

    net = DWTDenoiserNet(1, 3, 64, 1, ch_mult, 2, attn_res=atn_res).to(device)
    o = net(_x, _t, _c)
    print(o.shape)
