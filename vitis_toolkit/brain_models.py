# toolkit/brain_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer2D(nn.Module):
    def forward(self, src, flow):
        b, _, h, w = src.shape
        device = src.device

        # Build grid without torch.meshgrid to keep NNDCT happy
        yy = torch.arange(h, device=device).view(h, 1).expand(h, w)
        xx = torch.arange(w, device=device).view(1, w).expand(h, w)
        grid = torch.stack((xx, yy), dim=0).float()  # (2, h, w)
        grid = grid.unsqueeze(0).expand(b, 2, h, w)  # (b, 2, h, w)

        pts = grid + flow
        x = 2.0 * (pts[:, 0] / (w - 1.0)) - 1.0
        y = 2.0 * (pts[:, 1] / (h - 1.0)) - 1.0
        sample_grid = torch.stack((x, y), dim=-1)
        return F.grid_sample(
            src,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )




class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.conv(x))


class SliceRegistrationNet(nn.Module):
    def __init__(self, enc_feats=(32, 64, 64), final_feats=(64, 32)):
        super().__init__()
        self.encoders = nn.ModuleList()
        in_channels = 2
        for nf in enc_feats:
            self.encoders.append(ConvBlock2D(in_channels, nf))
            in_channels = nf

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock2D(enc_feats[-1], enc_feats[-1])

        self.decoders = nn.ModuleList()
        decoder_in = enc_feats[-1]
        for skip_ch in reversed(enc_feats):
            self.decoders.append(ConvBlock2D(decoder_in + skip_ch, skip_ch))
            decoder_in = skip_ch

        self.final_conv0 = ConvBlock2D(decoder_in, final_feats[0])
        self.final_conv1 = ConvBlock2D(final_feats[0], final_feats[1])
        self.flow = nn.Conv2d(final_feats[1], 2, kernel_size=3, padding=1)
        self.transformer = SpatialTransformer2D()
        self.apply(self._init_weights)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for skip, dec in zip(reversed(skips), self.decoders):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        x = self.final_conv0(x)
        x = self.final_conv1(x)
        flow = self.flow(x)
        moved = self.transformer(src, flow)
        return moved, flow
