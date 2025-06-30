from .base_denoiser import BaseDenoiserNet
from .dwt_denoiser import DWTDenoiserNet
from .base_unet import BaseUNet



def get_net(net_name, **kwargs):
    ch_mult = [1, 2, 2, 4, 4]
    atn_res = [8, 16]
    num_res_blocks = 2
    out_channels = kwargs.get('out_channels', 1)
    base_channels = kwargs.get('base_channels', 64)

    if net_name == 'BaseDenoiserNet':
        in_channels = kwargs.get('in_channels', 1)
        cond_channels = kwargs.get('cond_channels', 3)
        return BaseDenoiserNet(in_channels, cond_channels, base_channels, out_channels, ch_mult, num_res_blocks, attn_res=atn_res)

    if net_name == 'DWTDenoiserNet':
        in_channels = kwargs.get('in_channels', 1)
        cond_channels = kwargs.get('cond_channels', 3)
        return DWTDenoiserNet(in_channels, cond_channels, base_channels, out_channels, ch_mult, num_res_blocks, attn_res=atn_res)

    elif net_name == 'BaseUNet':
        in_channels = kwargs.get('cond_channels', 3)
        return BaseUNet(in_channels, out_channels)

    else:
        raise NotImplementedError(f'Unsupported network {net_name}.')
