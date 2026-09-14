from typing import Tuple
import torch
import torch.nn.functional as F


def pad_to_multiple(
    x: torch.Tensor, multiple: int = 16
) -> Tuple[torch.Tensor, Tuple[int, int]]:
  """Pads spatial dimensions (H, W) using reflection to be divisible by a factor.

  Returns the padded tensor and original (H, W) dimensions.
  """
  h, w = x.shape[-2:]
  pad_h = (multiple - h % multiple) % multiple
  pad_w = (multiple - w % multiple) % multiple

  if pad_h == 0 and pad_w == 0:
    return x, (h, w)

  x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
  return x_padded, (h, w)


def crop_to_original(
    x: torch.Tensor, original_hw: Tuple[int, int]
) -> torch.Tensor:
  """Crops padded output tensor back to its original (H, W) resolution."""
  h, w = original_hw
  return x[..., :h, :w]