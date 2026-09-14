# tests/test_pipeline.py
import torch
from src.losses.composite_loss import CharbonnierMSEEdgeLoss
from src.models.care_unet import CARE_UNet
from src.models.hybrid_unet import CustomSOTAHybrid
from src.utils.evaluation import predict_tta
from src.utils.padding import crop_to_original, pad_to_multiple


def test_2d_hybrid_forward_backward():
  print("Testing 2D Hybrid Forward & Backward...")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Mock batch: (B=2, C=1, H=256, W=256)
  x = torch.randn(2, 1, 256, 256, device=device)
  target = torch.randn(2, 1, 256, 256, device=device)

  model = CustomSOTAHybrid(
      in_channels=1, out_channels=1, base_dim=32, crop_size=256, use_mamba=True
  ).to(device)
  criterion = CharbonnierMSEEdgeLoss().to(device)

  out = model(x)
  assert out.shape == target.shape, (
      f"Shape mismatch: expected {target.shape}, got {out.shape}"
  )

  loss = criterion(out, target)
  loss.backward()
  print(f"  Passed! Loss: {loss.item():.4f}")


def test_25d_hybrid_forward_backward():
  print("Testing 2.5D Hybrid (5 input channels)...")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Mock 2.5D input: 5 slices, predicting 1 target slice
  x = torch.randn(2, 5, 256, 256, device=device)
  target = torch.randn(2, 1, 256, 256, device=device)

  model = CustomSOTAHybrid(
      in_channels=5, out_channels=1, base_dim=32, crop_size=256, use_mamba=True
  ).to(device)
  criterion = CharbonnierMSEEdgeLoss().to(device)

  out = model(x)
  assert out.shape == target.shape, (
      f"Shape mismatch: expected {target.shape}, got {out.shape}"
  )

  loss = criterion(out, target)
  loss.backward()
  print(f"  Passed! Loss: {loss.item():.4f}")


def test_care_baseline():
  print("Testing CARE U-Net Baseline...")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  x = torch.randn(2, 1, 256, 256, device=device)
  model = CARE_UNet(in_channels=1, out_channels=1, base_channels=32).to(device)
  out = model(x)
  assert out.shape == x.shape
  print("  Passed!")


def test_padding_and_tta():
  print("Testing Padding and TTA...")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Arbitrary non-divisible-by-16 resolution
  x = torch.randn(1, 1, 250, 250, device=device)
  model = CustomSOTAHybrid(
      in_channels=1, out_channels=1, base_dim=32, crop_size=256, use_mamba=True
  ).to(device)

  padded_x, hw = pad_to_multiple(x, multiple=16)
  assert (
      padded_x.shape[-2] % 16 == 0 and padded_x.shape[-1] % 16 == 0
  ), "Padding failed"

  pred = predict_tta(model, padded_x, tta_mode="x4")
  pred = crop_to_original(pred, hw)
  assert pred.shape == x.shape, f"Crop failed: {pred.shape} vs {x.shape}"
  print("  Passed!")


if __name__ == "__main__":
  test_2d_hybrid_forward_backward()
  test_25d_hybrid_forward_backward()
  test_care_baseline()
  test_padding_and_tta()
  print("\n All core components functioning as expected!")