from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two sequential (Conv2d -> BatchNorm2d -> ReLU) operations.
    Standard building block for U-Net encoder and decoder.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool2d then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv with skip connection concatenation."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 if necessary to match x2 spatial size
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution mapping latent features to class logits."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FloodUNet(nn.Module):
    """
    Lightweight Semantic Segmentation U-Net for Flood and Water Inundation Mapping.
    
    Architecture:
    - Input: Multi-temporal SAR or Optical imagery (default 4 channels: pre_vv, pre_vh, post_vv, post_vh)
    - Encoder: 4 stages with residual double convolutions and downsampling [32, 64, 128, 256]
    - Bottleneck: 512 channels
    - Decoder: 4 stages with transposed convolutions and skip connections
    - Output: 1 channel binary flood probability logits
    
    Properties:
    - Lightweight (~1.9M parameters) for efficient local CPU/GPU execution.
    - Preserves boundary details via skip connections essential for mountain river channels.
    - Automatic spatial padding/cropping for non-power-of-2 input dimensions.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        features: Optional[List[int]] = None,
        bilinear: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        feats = features or [32, 64, 128, 256]
        f0, f1, f2, f3 = feats[0], feats[1], feats[2], feats[3]
        f_bottleneck = f3 * 2

        self.inc = DoubleConv(in_channels, f0)
        self.down1 = Down(f0, f1)
        self.down2 = Down(f1, f2)
        self.down3 = Down(f2, f3)
        factor = 2 if bilinear else 1
        self.down4 = Down(f3, f_bottleneck // factor)

        self.up1 = Up(f_bottleneck, f3 // factor, bilinear)
        self.up2 = Up(f3, f2 // factor, bilinear)
        self.up3 = Up(f2, f1 // factor, bilinear)
        self.up4 = Up(f1, f0, bilinear)
        self.outc = OutConv(f0, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with auto-padding to ensure spatial dimensions are divisible by 16.
        Args:
            x: Input tensor of shape (B, in_channels, H, W)
        Returns:
            Logits of shape (B, out_channels, H, W)
        """
        orig_h, orig_w = x.shape[2], x.shape[3]
        pad_h = (16 - (orig_h % 16)) % 16
        pad_w = (16 - (orig_w % 16)) % 16

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        if pad_h > 0 or pad_w > 0:
            logits = logits[:, :, :orig_h, :orig_w]

        return logits

    def count_parameters(self) -> int:
        """Return total trainable parameters count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_mask(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference helper returning both sigmoid probability map and thresholded binary mask.
        Returns: (probability_map, binary_mask)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            mask = (probs >= threshold).to(torch.uint8)
        return probs, mask
