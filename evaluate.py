import argparse
import os
from typing import Any, Dict
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import DataLoader
import yaml

from monai.data import Dataset
from monai.transforms import Compose

from src.data.dataset_25d import (
    CustomTiffLoader25D,
    PairedPercentileScaler25D,
    ToTensor25D,
    build_25d_pairs,
)
from src.data.dataset_2d import (
    CustomTiffLoader,
    IndependentPercentileScaler,
    ToTensor2D,
    build_pairs,
)
from src.models.care_unet import CARE_UNet
from src.models.hybrid_unet import CustomSOTAHybrid
from src.utils.evaluation import evaluate_model, evaluate_model_25d


def main():
  parser = argparse.ArgumentParser(
      description="Evaluate Trained Model on Test Split"
  )
  parser.add_argument(
      "--config", type=str, required=True, help="Path to YAML config"
  )
  parser.add_argument(
      "--weights", type=str, required=True, help="Path to checkpoint .pth file"
  )
  parser.add_argument(
      "--tta",
      type=str,
      default="x8",
      choices=["none", "x4", "x8"],
      help="TTA mode",
  )
  args = parser.parse_args()

  with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  is_25d = cfg["dataset"]["type"] == "2.5d"
  in_channels = cfg["model"]["in_channels"]

  # Reconstruct model
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

  checkpoint = torch.load(args.weights, map_location=device)
  model.load_state_dict(checkpoint["model"])
  model.to(device)
  model.eval()

  # Set up data
  d_cfg = cfg["dataset"]
  if not is_25d:
    data_dicts = build_pairs(
        d_cfg["noisy_dir"], d_cfg["clean_dir"], d_cfg["name"].lower()
    )
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
    val_ds = Dataset(data=data_dicts, transform=eval_trans)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        output_mode=cfg["model"].get("output_mode", "clean"),
        tta_mode=args.tta,
        pad_multiple=cfg["evaluation"].get("pad_multiple", 16),
        use_sipsnr=cfg["evaluation"].get("use_sipsnr", True),
    )
  else:
    data_dicts = build_25d_pairs(
        d_cfg["noisy_dir"],
        d_cfg["clean_dir"],
        d_cfg["name"].lower(),
        window_size=d_cfg.get("window_size", 5),
    )
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
    val_ds = Dataset(data=data_dicts, transform=eval_trans)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    metrics = evaluate_model_25d(
        model=model,
        dataloader=val_loader,
        device=device,
        output_mode=cfg["model"].get("output_mode", "clean"),
        tta_mode=args.tta,
        pad_multiple=cfg["evaluation"].get("pad_multiple", 16),
        use_sipsnr=cfg["evaluation"].get("use_sipsnr", True),
    )

  print(f"\nFinal Evaluation Output:")
  for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
  main()