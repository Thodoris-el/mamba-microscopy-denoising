from typing import Any, Dict, Optional
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .padding import crop_to_original, pad_to_multiple

try:
  from careamics.metrics import SampleSIPSNR
except ImportError:
  SampleSIPSNR = None


def model_forward(
    model: torch.nn.Module, x: torch.Tensor, output_mode: str = "clean"
) -> torch.Tensor:
  """Executes forward pass and handles tuple unwrapping and residual additions."""
  y = model(x)
  if isinstance(y, (tuple, list)):
    y = y[0]
  if output_mode == "residual":
    y = x + y
  return y


def predict_tta(
    model: torch.nn.Module,
    x: torch.Tensor,
    output_mode: str = "clean",
    tta_mode: Optional[str] = "x8",
) -> torch.Tensor:
  """Runs Test-Time Augmentation (x4 rotations or x8 dihedral group)."""
  if tta_mode is None or tta_mode == "none":
    return model_forward(model, x, output_mode=output_mode)

  dims = [2, 3]
  preds = []

  # 4 orthogonal rotations
  for k in range(4):
    x_aug = torch.rot90(x, k=k, dims=dims)
    y_aug = model_forward(model, x_aug, output_mode=output_mode)
    y = torch.rot90(y_aug, k=-k, dims=dims)
    preds.append(y)

  # 4 horizontal flips combined with rotations
  if tta_mode == "x8":
    for k in range(4):
      x_aug = torch.rot90(x, k=k, dims=dims)
      x_aug = torch.flip(x_aug, dims=[3])
      y_aug = model_forward(model, x_aug, output_mode=output_mode)
      y = torch.flip(y_aug, dims=[3])
      y = torch.rot90(y, k=-k, dims=dims)
      preds.append(y)

  return torch.stack(preds, dim=0).mean(dim=0)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_mode: str = "clean",
    tta_mode: str = "x8",
    data_range: float = 1.0,
    clamp: bool = True,
    pad_multiple: int = 16,
    use_sipsnr: bool = True,
    use_amp: bool = False,
) -> Dict[str, float]:
  """Evaluates 2D planar datasets (Nuclei, FMD) computing PSNR, SSIM, and SI-PSNR."""
  model.eval()
  total_base_psnr = 0.0
  total_base_ssim = 0.0
  total_model_psnr = 0.0
  total_model_ssim = 0.0
  num_samples = 0

  if use_sipsnr and SampleSIPSNR is not None:
    base_sipsnr_metric = SampleSIPSNR(
        n_channels=1, use_scale_invariance=True
    ).to(device)
    model_sipsnr_metric = SampleSIPSNR(
        n_channels=1, use_scale_invariance=True
    ).to(device)
  else:
    base_sipsnr_metric = None
    model_sipsnr_metric = None

  with torch.inference_mode():
    loop = tqdm(dataloader, desc="Evaluating 2D validation set")
    for batch in loop:
      noisy_imgs = batch["noisy"].to(device, non_blocking=True).float()
      gt_imgs = batch["gt"].to(device, non_blocking=True).float()

      noisy_input, original_hw = (
          pad_to_multiple(noisy_imgs, multiple=pad_multiple)
          if pad_multiple
          else (noisy_imgs, noisy_imgs.shape[-2:])
      )

      with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        preds = predict_tta(
            model=model, x=noisy_input, output_mode=output_mode, tta_mode=tta_mode
        )
        preds = crop_to_original(preds, original_hw).float()

        if clamp:
          preds = torch.clamp(preds, 0.0, 1.0)
          noisy_eval = torch.clamp(noisy_imgs, 0.0, 1.0)
          gt_eval = torch.clamp(gt_imgs, 0.0, 1.0)
        else:
          noisy_eval, gt_eval = noisy_imgs, gt_imgs

        if base_sipsnr_metric is not None:
          base_sipsnr_metric.update(noisy_eval, gt_eval)
          model_sipsnr_metric.update(preds, gt_eval)

        preds_np = preds.detach().cpu().numpy()[:, 0]
        noisy_np = noisy_eval.detach().cpu().numpy()[:, 0]
        gt_np = gt_eval.detach().cpu().numpy()[:, 0]

        for pred, raw, gt in zip(preds_np, noisy_np, gt_np):
          total_base_psnr += peak_signal_noise_ratio(
              gt, raw, data_range=data_range
          )
          total_base_ssim += structural_similarity(
              gt, raw, data_range=data_range
          )
          total_model_psnr += peak_signal_noise_ratio(
              gt, pred, data_range=data_range
          )
          total_model_ssim += structural_similarity(
              gt, pred, data_range=data_range
          )
          num_samples += 1

  avg_base_psnr = total_base_psnr / max(num_samples, 1)
  avg_base_ssim = total_base_ssim / max(num_samples, 1)
  avg_model_psnr = total_model_psnr / max(num_samples, 1)
  avg_model_ssim = total_model_ssim / max(num_samples, 1)

  results = {
      "baseline_psnr": avg_base_psnr,
      "baseline_ssim": avg_base_ssim,
      "model_psnr": avg_model_psnr,
      "model_ssim": avg_model_ssim,
      "psnr_improvement": avg_model_psnr - avg_base_psnr,
      "ssim_improvement": avg_model_ssim - avg_base_ssim,
  }

  if model_sipsnr_metric is not None and base_sipsnr_metric is not None:
    avg_base_sipsnr = base_sipsnr_metric.compute().mean().item()
    avg_model_sipsnr = model_sipsnr_metric.compute().mean().item()
    results.update({
        "baseline_sipsnr": avg_base_sipsnr,
        "model_sipsnr": avg_model_sipsnr,
        "sipsnr_improvement": avg_model_sipsnr - avg_base_sipsnr,
    })

  return results


def evaluate_model_25d(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_mode: str = "clean",
    tta_mode: str = "x4",
    data_range: float = 1.0,
    clamp: bool = True,
    pad_multiple: int = 16,
    use_sipsnr: bool = True,
    use_amp: bool = False,
) -> Dict[str, float]:
  """Evaluates 2.5D multi-slice volumetric models against the center slice ground truth."""
  model.eval()
  total_base_psnr = 0.0
  total_base_ssim = 0.0
  total_model_psnr = 0.0
  total_model_ssim = 0.0
  num_samples = 0

  if use_sipsnr and SampleSIPSNR is not None:
    base_sipsnr_metric = SampleSIPSNR(
        n_channels=1, use_scale_invariance=True
    ).to(device)
    model_sipsnr_metric = SampleSIPSNR(
        n_channels=1, use_scale_invariance=True
    ).to(device)
  else:
    base_sipsnr_metric = None
    model_sipsnr_metric = None

  with torch.inference_mode():
    loop = tqdm(dataloader, desc="Evaluating 2.5D validation set")
    for batch in loop:
      noisy_imgs = batch["noisy"].to(device, non_blocking=True).float()
      gt_imgs = batch["gt"].to(device, non_blocking=True).float()

      noisy_input, original_hw = (
          pad_to_multiple(noisy_imgs, multiple=pad_multiple)
          if pad_multiple
          else (noisy_imgs, noisy_imgs.shape[-2:])
      )

      with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        preds = predict_tta(
            model=model, x=noisy_input, output_mode=output_mode, tta_mode=tta_mode
        )

      preds = crop_to_original(preds, original_hw).float()
      if not torch.isfinite(preds).all():
        preds = torch.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)

      if clamp:
        preds = torch.clamp(preds, 0.0, 1.0)
        noisy_eval = torch.clamp(noisy_imgs, 0.0, 1.0)
        gt_eval = torch.clamp(gt_imgs, 0.0, 1.0)
      else:
        noisy_eval, gt_eval = noisy_imgs, gt_imgs

      center_idx = noisy_eval.shape[1] // 2
      noisy_eval_center = noisy_eval[:, center_idx : center_idx + 1, :, :]

      if base_sipsnr_metric is not None:
        base_sipsnr_metric.update(noisy_eval_center, gt_eval)
        model_sipsnr_metric.update(preds, gt_eval)

      preds_np = preds.detach().cpu().numpy()[:, 0]
      noisy_np_center = noisy_eval_center.detach().cpu().numpy()[:, 0]
      gt_np = gt_eval.detach().cpu().numpy()[:, 0]

      for pred, raw, gt in zip(preds_np, noisy_np_center, gt_np):
        if not np.isfinite(pred).all():
          pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)

        total_base_psnr += peak_signal_noise_ratio(
            gt, raw, data_range=data_range
        )
        total_base_ssim += structural_similarity(gt, raw, data_range=data_range)
        total_model_psnr += peak_signal_noise_ratio(
            gt, pred, data_range=data_range
        )
        total_model_ssim += structural_similarity(
            gt, pred, data_range=data_range
        )
        num_samples += 1

  avg_base_psnr = total_base_psnr / max(num_samples, 1)
  avg_base_ssim = total_base_ssim / max(num_samples, 1)
  avg_model_psnr = total_model_psnr / max(num_samples, 1)
  avg_model_ssim = total_model_ssim / max(num_samples, 1)

  results = {
      "baseline_psnr": avg_base_psnr,
      "baseline_ssim": avg_base_ssim,
      "model_psnr": avg_model_psnr,
      "model_ssim": avg_model_ssim,
      "psnr_improvement": avg_model_psnr - avg_base_psnr,
      "ssim_improvement": avg_model_ssim - avg_base_ssim,
  }

  if model_sipsnr_metric is not None and base_sipsnr_metric is not None:
    avg_base_sipsnr = base_sipsnr_metric.compute().mean().item()
    avg_model_sipsnr = model_sipsnr_metric.compute().mean().item()
    results.update({
        "baseline_sipsnr": avg_base_sipsnr,
        "model_sipsnr": avg_model_sipsnr,
        "sipsnr_improvement": avg_model_sipsnr - avg_base_sipsnr,
    })

  return results