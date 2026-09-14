import argparse
import os
import random
from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml

from monai.data import CacheDataset, Dataset
from monai.transforms import Compose

from src.data.dataset_25d import (
    AlbumentationsDictTransform25D,
    CustomTiffLoader25D,
    PairedPercentileScaler25D,
    ToTensor25D,
    build_25d_pairs,
)
from src.data.dataset_2d import (
    AlbumentationsDictTransform,
    CustomTiffLoader,
    IndependentPercentileScaler,
    ToTensor2D,
    build_pairs,
)
from src.data.splitters import get_2d_kfold_splits, get_group_kfold_splits
from src.losses.composite_loss import CharbonnierMSEEdgeLoss
from src.models.care_unet import CARE_UNet
from src.models.hybrid_unet import CustomSOTAHybrid
from src.utils.evaluation import evaluate_model, evaluate_model_25d
from src.utils.padding import crop_to_original, pad_to_multiple


def set_seed(seed: int = 42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
  with open(path, "r") as f:
    return yaml.safe_load(f)


def build_pipeline_model(
    cfg: Dict[str, Any], in_channels: int, device: torch.device
) -> nn.Module:
  model_name = cfg["model"].get("name", "CustomSOTAHybrid")
  if model_name == "CARE_UNet":
    model = CARE_UNet(
        in_channels=in_channels,
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"].get("base_dim", 32),
        residual=cfg["model"].get("residual", False),
    )
  else:
    model = CustomSOTAHybrid(
        in_channels=in_channels,
        out_channels=cfg["model"]["out_channels"],
        base_dim=cfg["model"].get("base_dim", 32),
        crop_size=cfg["model"].get("crop_size", 256),
        use_mamba=cfg["model"].get("use_mamba", True),
    )
  return model.to(device)


def build_pipeline_loss(
    cfg: Dict[str, Any], device: torch.device
) -> nn.Module:
  loss_type = cfg["loss"].get("type", "composite")
  if loss_type == "l1":
    return nn.L1Loss().to(device)
  return CharbonnierMSEEdgeLoss(
      eps=cfg["loss"].get("eps", 1e-3),
      charbonnier_weight=cfg["loss"].get("charbonnier_weight", 0.70),
      mse_weight=cfg["loss"].get("mse_weight", 0.25),
      edge_weight=cfg["loss"].get("edge_weight", 0.01),
  ).to(device)


def run_training(cfg: Dict[str, Any]):
  set_seed(cfg["experiment"].get("seed", 42))
  device = torch.device(
      cfg["experiment"].get("device", "cuda")
      if torch.cuda.is_available()
      else "cpu"
  )
  is_25d = cfg["dataset"]["type"] == "2.5d"
  in_channels = cfg["model"]["in_channels"]

  # 1. Dataset loading & splitting
  if not is_25d:
    d_cfg = cfg["dataset"]
    if d_cfg["name"] == "Nuclei":
      small_pairs = build_pairs(
          d_cfg["noisy_dir_small"], d_cfg["clean_dir_small"], "small"
      )
      large_pairs = build_pairs(
          d_cfg["noisy_dir_large"], d_cfg["clean_dir_large"], "large"
      )
      data_dicts = small_pairs + large_pairs
    else:
      data_dicts = build_pairs(
          d_cfg["noisy_dir"], d_cfg["clean_dir"], d_cfg["name"].lower()
      )

    splits = get_2d_kfold_splits(
        data_dicts,
        n_splits=cfg["training"]["n_folds"],
        seed=cfg["experiment"].get("seed", 42),
    )
    train_trans = Compose([
        CustomTiffLoader(keys=("noisy", "gt")),
        IndependentPercentileScaler(
            keys=("noisy", "gt"),
            lower=d_cfg.get("percentile_lower", 0.1),
            upper=d_cfg.get("percentile_upper", 99.9),
            clip=True,
        ),
        AlbumentationsDictTransform(
            keys=("noisy", "gt"),
            crop_size=d_cfg.get("crop_size", 256),
            min_patch_std=d_cfg.get("min_patch_std", 0.005),
        ),
    ])
    eval_trans = Compose([
        CustomTiffLoader(keys=("noisy", "gt")),
        IndependentPercentileScaler(
            keys=("noisy", "gt"),
            lower=d_cfg.get("percentile_lower", 0.1),
            upper=d_cfg.get("percentile_upper", 99.9),
            clip=True,
        ),
        ToTensor2D(keys=("noisy", "gt")),
    ])
  else:
    d_cfg = cfg["dataset"]
    if d_cfg["name"] == "Planaria":
      small_pairs = build_25d_pairs(
          d_cfg["noisy_dir_small"],
          d_cfg["clean_dir_small"],
          "planaria_small",
          window_size=d_cfg.get("window_size", 5),
      )
      large_pairs = build_25d_pairs(
          d_cfg["noisy_dir_large"],
          d_cfg["clean_dir_large"],
          "planaria_large",
          window_size=d_cfg.get("window_size", 5),
      )
      data_dicts = small_pairs + large_pairs
    else:
      data_dicts = build_25d_pairs(
          d_cfg["noisy_dir"],
          d_cfg["clean_dir"],
          "tribolium",
          window_size=d_cfg.get("window_size", 5),
      )

    for d in data_dicts:
      if "vol_id" not in d:
        d["vol_id"] = "_".join(d["id"].split("_")[:-1])

    splits = get_group_kfold_splits(
        data_dicts, group_key="vol_id", n_splits=cfg["training"]["n_folds"]
    )
    train_trans = Compose([
        CustomTiffLoader25D(
            keys=("noisy", "gt"), expected_channels=in_channels
        ),
        PairedPercentileScaler25D(
            keys=("noisy", "gt"),
            lower=d_cfg.get("percentile_lower", 0.1),
            upper=d_cfg.get("percentile_upper", 99.9),
            clip=True,
        ),
        AlbumentationsDictTransform25D(
            keys=("noisy", "gt"),
            crop_size=d_cfg.get("crop_size", 256),
            min_patch_std=d_cfg.get("min_patch_std", 0.005),
        ),
    ])
    eval_trans = Compose([
        CustomTiffLoader25D(
            keys=("noisy", "gt"), expected_channels=in_channels
        ),
        PairedPercentileScaler25D(
            keys=("noisy", "gt"),
            lower=d_cfg.get("percentile_lower", 0.1),
            upper=d_cfg.get("percentile_upper", 99.9),
            clip=True,
            save_stats=True,
        ),
        ToTensor25D(keys=("noisy", "gt")),
    ])

  save_dir = cfg["training"].get("save_dir", "./checkpoints")
  os.makedirs(save_dir, exist_ok=True)
  fold_psnrs, fold_sipsnrs = [], []

  # 2. Cross-Validation Loop
  for fold, (train_data, val_data) in enumerate(splits):
    print(f"\n{'='*60}\nINITIATING FOLD {fold + 1} / {len(splits)}\n{'='*60}")
    train_ds = Dataset(data=train_data, transform=train_trans)
    val_ds = CacheDataset(
        data=val_data,
        transform=eval_trans,
        cache_rate=1.0 if not is_25d else 0.0,
        num_workers=0,
    )

    max_fast = cfg["training"].get("max_val_samples", 250)
    fast_indices = torch.randperm(len(val_ds))[: min(max_fast, len(val_ds))]
    val_fast_loader = DataLoader(
        Subset(val_ds, fast_indices.tolist()), batch_size=1, shuffle=False
    )

    if is_25d:
      max_full = cfg["training"].get("max_val_samples_full", 2000)
      full_indices = torch.randperm(len(val_ds))[: min(max_full, len(val_ds))]
      val_full_loader = DataLoader(
          Subset(val_ds, full_indices.tolist()), batch_size=1, shuffle=False
      )
    else:
      val_full_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=cfg["training"].get("num_workers", 2),
    )

    model = build_pipeline_model(cfg, in_channels, device)
    criterion = build_pipeline_loss(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=float(cfg["training"].get("scheduler_eta_min", 1e-6)),
    )
    scaler = torch.amp.GradScaler(
        device=device.type, enabled=cfg["training"].get("use_amp", False)
    )

    best_score = -1e9
    best_path = os.path.join(save_dir, f"model_fold{fold+1}_best.pth")

    for epoch in range(cfg["training"]["epochs"]):
      model.train()
      train_iter = iter(train_loader)
      for _ in range(cfg["training"]["steps_per_epoch"]):
        try:
          batch = next(train_iter)
        except StopIteration:
          train_iter = iter(train_loader)
          batch = next(train_iter)

        noisy = batch["noisy"].to(device, non_blocking=True).float()
        gt = batch["gt"].to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=cfg["training"].get("use_amp", False),
        ):
          padded_noisy, hw = pad_to_multiple(
              noisy, cfg["evaluation"].get("pad_multiple", 16)
          )
          preds = crop_to_original(model(padded_noisy), hw)
          loss = criterion(preds, gt)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["training"].get("grad_clip_max_norm", 1.0)
        )
        scaler.step(optimizer)
        scaler.update()

      scheduler.step()

      # Periodic validation
      if (epoch + 1) % cfg["training"][
          "validate_every"
      ] == 0 or epoch == cfg["training"]["epochs"] - 1:
        eval_fn = evaluate_model_25d if is_25d else evaluate_model
        metrics = eval_fn(
            model=model,
            dataloader=val_fast_loader,
            device=device,
            output_mode=cfg["model"].get("output_mode", "clean"),
            tta_mode=cfg["evaluation"].get("fast_tta_mode", "x4"),
            pad_multiple=cfg["evaluation"].get("pad_multiple", 16),
            use_sipsnr=cfg["evaluation"].get("use_sipsnr", True),
            use_amp=cfg["training"].get("use_amp", False),
        )
        score = metrics.get("model_sipsnr") or metrics["model_psnr"]
        if score > best_score:
          best_score = score
          torch.save({"model": model.state_dict(), "cfg": cfg}, best_path)

    # Re-evaluate fold using full test-time augmentation (TTA x8)
    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model"])
    eval_fn = evaluate_model_25d if is_25d else evaluate_model
    final_metrics = eval_fn(
        model=model,
        dataloader=val_full_loader,
        device=device,
        output_mode=cfg["model"].get("output_mode", "clean"),
        tta_mode=cfg["evaluation"].get("final_tta_mode", "x8"),
        pad_multiple=cfg["evaluation"].get("pad_multiple", 16),
        use_sipsnr=cfg["evaluation"].get("use_sipsnr", True),
        use_amp=cfg["training"].get("use_amp", False),
    )
    fold_psnrs.append(final_metrics["model_psnr"])
    if "model_sipsnr" in final_metrics:
      fold_sipsnrs.append(final_metrics["model_sipsnr"])

  print(f"\n{'='*60}\nFINAL CROSS-VALIDATION SUMMARY\n{'='*60}")
  print(f"Mean PSNR:    {np.mean(fold_psnrs):.4f} ± {np.std(fold_psnrs):.4f} dB")
  if fold_sipsnrs:
    print(
        f"Mean SI-PSNR: {np.mean(fold_sipsnrs):.4f} ±"
        f" {np.std(fold_sipsnrs):.4f} dB"
    )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Train Restoration Model via Config"
  )
  parser.add_argument(
      "--config", type=str, required=True, help="Path to YAML configuration"
  )
  args = parser.parse_args()
  run_training(load_config(args.config))