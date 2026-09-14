import torch
import torch.nn as nn
import torch.nn.functional as F

try:
  from basicsr.archs.mambairv2_arch import MambaIRv2
except ImportError:
  # In case MambaIR is not in python path or installed
  MambaIRv2 = None


class SimpleGate(nn.Module):
  """Feature channel-halving multiplicative gating unit."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=1)
    return x1 * x2


class LayerNorm2d(nn.Module):
  """2D spatial LayerNorm supporting channels-first tensors (B, C, H, W)."""

  def __init__(self, channels: int, eps: float = 1e-3):
    super().__init__()
    self.norm = nn.LayerNorm(channels, eps=eps)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.permute(0, 2, 3, 1).contiguous()
    x = self.norm(x)
    x = x.permute(0, 3, 1, 2).contiguous()
    return x


class SimplifiedChannelAttention(nn.Module):
  """Simplified channel attention mechanism via global pooling and 1x1 conv."""

  def __init__(self, channels: int):
    super().__init__()
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
  """Nonlinear Activation Free Block (NAFNet block)."""

  def __init__(self, dim: int, dw_expand: int = 2, ffn_expand: int = 2):
    super().__init__()
    dw_channels = dim * dw_expand
    ffn_channels = dim * ffn_expand

    self.norm1 = LayerNorm2d(dim)
    self.conv1 = nn.Conv2d(dim, dw_channels, kernel_size=1, padding=0)
    self.conv2 = nn.Conv2d(
        dw_channels,
        dw_channels,
        kernel_size=3,
        padding=1,
        groups=dw_channels,
    )
    self.sg = SimpleGate()
    self.sca = SimplifiedChannelAttention(dw_channels // 2)
    self.conv3 = nn.Conv2d(dw_channels // 2, dim, kernel_size=1, padding=0)

    self.norm2 = LayerNorm2d(dim)
    self.conv4 = nn.Conv2d(dim, ffn_channels, kernel_size=1, padding=0)
    self.conv5 = nn.Conv2d(ffn_channels // 2, dim, kernel_size=1, padding=0)

    self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
    self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    identity = x
    y = self.norm1(x)
    y = self.conv1(y)
    y = self.conv2(y)
    y = self.sg(y)
    y = self.sca(y)
    y = self.conv3(y)
    x = identity + y * self.beta

    identity = x
    y = self.norm2(x)
    y = self.conv4(y)
    y = self.sg(y)
    y = self.conv5(y)
    x = identity + y * self.gamma

    return x


class NAFStack(nn.Module):
  """Sequential stack of NAFBlocks."""

  def __init__(self, dim: int, num_blocks: int):
    super().__init__()
    self.blocks = nn.Sequential(*[NAFBlock(dim) for _ in range(num_blocks)])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.blocks(x)


class UpBlock(nn.Module):
  """Bilinear upsampling decoder stage with skip-connection concatenation."""

  def __init__(
      self,
      in_channels: int,
      skip_channels: int,
      out_channels: int,
      num_blocks: int = 1,
  ):
    super().__init__()
    self.up_conv = nn.Conv2d(
        in_channels, out_channels, kernel_size=3, padding=1
    )
    self.fuse = nn.Conv2d(
        out_channels + skip_channels,
        out_channels,
        kernel_size=1,
        padding=0,
    )
    self.refine = NAFStack(out_channels, num_blocks)

  def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    x = F.interpolate(
        x,
        size=skip.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    x = self.up_conv(x)
    x = torch.cat([x, skip], dim=1)
    x = self.fuse(x)
    x = self.refine(x)
    return x


class CustomSOTAHybrid(nn.Module):
  """Hybrid U-Net featuring NAFNet stages and a MambaIRv2 bottleneck."""

  def __init__(
      self,
      in_channels: int = 1,
      out_channels: int = 1,
      base_dim: int = 32,
      crop_size: int = 256,
      use_mamba: bool = True,
  ):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.base_dim = base_dim
    self.crop_size = crop_size
    self.use_mamba = use_mamba

    c1 = base_dim
    c2 = base_dim * 2
    c3 = base_dim * 4
    c4 = base_dim * 8

    # Input projection
    self.embed = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

    # Encoder stages
    self.enc1 = NAFStack(c1, num_blocks=2)
    self.down1 = nn.Conv2d(c1, c2, kernel_size=2, stride=2)
    self.enc2 = NAFStack(c2, num_blocks=2)
    self.down2 = nn.Conv2d(c2, c3, kernel_size=2, stride=2)
    self.enc3 = NAFStack(c3, num_blocks=3)
    self.down3 = nn.Conv2d(c3, c4, kernel_size=2, stride=2)

    # Bottleneck stage
    bottleneck_img_size = crop_size // 8

    if use_mamba:
      if MambaIRv2 is None:
        raise ImportError(
            "basicsr.archs.mambairv2_arch.MambaIRv2 not found. "
            "Clone MambaIR repository into path or install dependencies."
        )
      self.bottleneck = MambaIRv2(
          img_size=bottleneck_img_size,
          patch_size=1,
          in_chans=c4,
          embed_dim=c4,
          depths=(4, 4),
          mlp_ratio=2.0,
          upscale=1,
          img_range=1.0,
          upsampler="",
          resi_connection="1conv",
      )
    else:
      self.bottleneck = NAFStack(c4, num_blocks=6)

    self.bottleneck_refine = NAFStack(c4, num_blocks=2)

    # Decoder stages
    self.up1 = UpBlock(
        in_channels=c4,
        skip_channels=c3,
        out_channels=c3,
        num_blocks=2,
    )
    self.up2 = UpBlock(
        in_channels=c3,
        skip_channels=c2,
        out_channels=c2,
        num_blocks=2,
    )
    self.up3 = UpBlock(
        in_channels=c2,
        skip_channels=c1,
        out_channels=c1,
        num_blocks=2,
    )

    # Global residual projection
    self.output_projection = nn.Conv2d(
        c1,
        out_channels,
        kernel_size=3,
        padding=1,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Use first channel (or center slice in 2.5D) as the reference for residual addition
    noisy_reference = (
        x[:, self.in_channels // 2 : self.in_channels // 2 + 1, :, :]
        if self.in_channels > 1
        else x[:, 0:1, :, :]
    )

    f0 = self.embed(x)
    skip1 = self.enc1(f0)
    f1 = self.down1(skip1)
    skip2 = self.enc2(f1)
    f2 = self.down2(skip2)
    skip3 = self.enc3(f2)
    f3 = self.down3(skip3)

    bot = self.bottleneck(f3)
    if isinstance(bot, (tuple, list)):
      bot = bot[0]

    if bot.shape[-2:] != f3.shape[-2:]:
      bot = F.interpolate(
          bot,
          size=f3.shape[-2:],
          mode="bilinear",
          align_corners=False,
      )

    bot = self.bottleneck_refine(bot)

    d1 = self.up1(bot, skip3)
    d2 = self.up2(d1, skip2)
    d3 = self.up3(d2, skip1)

    predicted_residual = self.output_projection(d3)
    denoised = noisy_reference + predicted_residual

    return denoised