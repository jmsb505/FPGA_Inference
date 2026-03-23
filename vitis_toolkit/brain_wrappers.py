# toolkit/brain_wrappers.py
import torch
import torch.nn as nn

from .brain_models import SliceRegistrationNet


class SliceWrapper(nn.Module):
    """
    Wraps SliceRegistrationNet to accept a single tensor of shape
    (B, 2, H, W) instead of (src, tgt) separately.
    """

    def __init__(self):
        super().__init__()
        self.net = SliceRegistrationNet()

    def forward(self, x):
        # x: (B, 2, H, W)
        src = x[:, 0:1, ...]
        tgt = x[:, 1:2, ...]
        moved, flow = self.net(src, tgt)
        return moved, flow
