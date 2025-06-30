import torch
from torch import nn
from monai.losses.ssim_loss import SSIMLoss


class SSIM_L1(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()

        self.L1_loss = nn.L1Loss(reduction=reduction)
        self.ssim_loss = SSIMLoss(spatial_dims=3)

    def forward(self, x, y, mask=None):
        if mask is None:
            mask = torch.ones_like(y, device=y.device)
            mask[y == 0] = 0
        l1 = self.L1_loss(x[mask != 0], y[mask != 0])

        a = torch.zeros_like(x)
        b = torch.zeros_like(y)
        for i in range(x.shape[0]):
            x_min, x_max = x[i].min(), x[i].max()
            y_min, y_max = y[i].min(), y[i].max()
            if x_min != x_max:
                a[i] = (x[i] - x_min) / (x_max - x_min)
            if y_min != y_max:
                b[i] = (y[i] - y_min) / (y_max - y_min)
        a[mask == 0] = 0
        b[mask == 0] = 0
        ss = self.ssim_loss(a, b)

        return l1 + ss
