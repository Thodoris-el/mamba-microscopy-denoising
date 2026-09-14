import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierMSEEdgeLoss(nn.Module):
  """Composite loss function for fluorescence microscopy denoising.

  Combines Charbonnier loss (smooth L1 proxy), Mean Squared Error (MSE),
  and a first-order spatial finite-difference edge penalty.
  """

  def __init__(
      self,
      eps: float = 1e-3,
      charbonnier_weight: float = 0.70,
      mse_weight: float = 0.25,
      edge_weight: float = 0.01,
  ):
    super().__init__()
    self.eps = eps
    self.charbonnier_weight = charbonnier_weight
    self.mse_weight = mse_weight
    self.edge_weight = edge_weight

  def forward(
      self, pred: torch.Tensor, target: torch.Tensor
  ) -> torch.Tensor:
    # 1. Charbonnier penalty
    diff = pred - target
    charbonnier_loss = torch.mean(
        torch.sqrt(diff * diff + self.eps * self.eps)
    )

    # 2. MSE penalty
    mse_loss = F.mse_loss(pred, target)

    # 3. Spatial gradient / Edge penalty via first-order forward differences
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    edge_loss_x = F.l1_loss(pred_dx, target_dx)
    edge_loss_y = F.l1_loss(pred_dy, target_dy)
    edge_loss = 0.5 * (edge_loss_x + edge_loss_y)

    # Total weighted composite loss
    loss = (
        self.charbonnier_weight * charbonnier_loss
        + self.mse_weight * mse_loss
        + self.edge_weight * edge_loss
    )

    return loss