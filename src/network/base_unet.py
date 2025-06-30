from monai.networks.nets import BasicUNet
from torch.nn import Module


class BaseUNet(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = BasicUNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, **_):
        return self.net(x)
