"""固定一个 batch，并排比较 plain AE 与 iFSQ AE 的过拟合能力。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from motion_ae.config import load_config
from motion_ae.dataset import build_datasets, dataloader_io_options, try_preload_one_dataset
from motion_ae.losses import ReconstructionLoss
from motion_ae.models.autoencoder import MotionAutoEncoder
from motion_ae.models.plain_autoencoder import PlainMotionAutoEncoder
from motion_ae.utils.experiment import get_device
from motion_ae.utils.seed import set_seed
from scripts.cli_args import add_config_args, add_runtime_args, apply_cli_overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare plain AE and iFSQ AE on the same fixed batch."
    )
    add_config_args(parser)
    add_runtime_args(parser)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--steps", type=int, default=200, help="固定 batch 上训练多少步。")
    parser.add_argument("--log_every", type=int, default=1000, help="每多少步打印一次 loss。")
    parser.add_argument(
        "--batch_index",
        type=int,
        default=0,
        help="从顺序 DataLoader 中取第几个 batch 作为固定 batch。",
    )
    parser.add_argument(
        "--history_json",
        type=str,
        default="outputs/debug_compare_overfit_history.json",
        help="保存 loss history 的 JSON 路径。",
    )
    return parser


def get_fixed_batch(loader: DataLoader, batch_index: int) -> torch.Tensor:
    """从 DataLoader 中取一个固定 batch。"""
    for idx, batch in enumerate(loader):
        if idx == batch_index:
            return batch
    raise IndexError(f"batch_index={batch_index} out of range for loader with {len(loader)} batches")


def forward_reconstruction(
    model: nn.Module, batch: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """兼容 plain AE 与 iFSQ AE 的前向，统一返回重建结果和调试信息。"""
    outputs = model(batch)
    if isinstance(outputs, tuple) and len(outputs) == 2:
        x_hat, info = outputs
        return x_hat, info
    if isinstance(outputs, tuple) and len(outputs) == 3:
        x_hat, _z_d, info = outputs
        return x_hat, info
    raise TypeError(f"Unsupported model forward output type: {type(outputs)}")


def overfit_single_batch(
    model: nn.Module,
    batch: torch.Tensor,
    criterion: ReconstructionLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps: int,
    log_every: int = 0,
    label: str = "model",
) -> List[float]:
    """在固定 batch 上重复训练并返回 loss 历史。"""
    batch = batch.to(device)
    model = model.to(device)
    criterion = criterion.to(device)

    history: List[float] = []
    model.train()
    for step in range(steps):
        x_hat, _info = forward_reconstruction(model, batch)
        loss, _loss_dict = criterion(x_hat, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        history.append(loss_value)
        if log_every > 0 and (step == 0 or (step + 1) % log_every == 0 or step == steps - 1):
            print(f"[{label}] step {step + 1:4d}/{steps} loss={loss_value:.6f}")

    return history


def summarize_history(history: List[float]) -> Dict[str, float]:
    """汇总一条 loss history。"""
    return {
        "first": float(history[0]),
        "last": float(history[-1]),
        "min": float(min(history)),
    }


def evaluate_model_before_training(
    model: nn.Module,
    batch: torch.Tensor,
    criterion: ReconstructionLoss,
    device: torch.device,
) -> float:
    """计算模型在固定 batch 上的初始 loss。"""
    model = model.to(device)
    batch = batch.to(device)
    criterion = criterion.to(device)
    model.eval()
    with torch.no_grad():
        x_hat, _info = forward_reconstruction(model, batch)
        loss, _ = criterion(x_hat, batch)
    return float(loss.item())


def compare_models_on_batch(
    batch: torch.Tensor,
    criterion: ReconstructionLoss,
    plain_model: PlainMotionAutoEncoder,
    ifsq_model: MotionAutoEncoder,
    device: torch.device,
    steps: int,
    log_every: int = 0,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
) -> Dict[str, Dict[str, Any]]:
    """在同一个 batch 上比较 plain AE 与 iFSQ AE 的过拟合能力。"""
    zero_loss, _ = criterion(torch.zeros_like(batch), batch)
    zero_baseline_loss = float(zero_loss.item())

    plain_init_loss = evaluate_model_before_training(plain_model, batch, criterion, device)
    print(f"[plain-ae] zero_baseline_loss={zero_baseline_loss:.6f}")
    print(f"[plain-ae] init_model_loss={plain_init_loss:.6f}")
    plain_optimizer = torch.optim.Adam(
        plain_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    plain_history = overfit_single_batch(
        model=plain_model,
        batch=batch,
        criterion=criterion,
        optimizer=plain_optimizer,
        device=device,
        steps=steps,
        log_every=log_every,
        label="plain-ae",
    )

    ifsq_init_loss = evaluate_model_before_training(ifsq_model, batch, criterion, device)
    print(f"[ifsq-ae] zero_baseline_loss={zero_baseline_loss:.6f}")
    print(f"[ifsq-ae] init_model_loss={ifsq_init_loss:.6f}")
    ifsq_optimizer = torch.optim.Adam(
        ifsq_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    ifsq_history = overfit_single_batch(
        model=ifsq_model,
        batch=batch,
        criterion=criterion,
        optimizer=ifsq_optimizer,
        device=device,
        steps=steps,
        log_every=log_every,
        label="ifsq-ae",
    )

    return {
        "plain_ae": {
            "zero_baseline_loss": zero_baseline_loss,
            "init_model_loss": plain_init_loss,
            "history": plain_history,
            "summary": summarize_history(plain_history),
        },
        "ifsq_ae": {
            "zero_baseline_loss": zero_baseline_loss,
            "init_model_loss": ifsq_init_loss,
            "history": ifsq_history,
            "summary": summarize_history(ifsq_history),
        },
    }


def save_history_json(payload: Dict[str, Any], path: str) -> str:
    """保存对照 history 到 JSON 文件。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(output_path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    set_seed(cfg.training.seed)

    device_name = get_device(cfg.training.device)
    device = torch.device(device_name)
    print(f"[compare-overfit] device={device}")

    train_ds, val_ds, _normalizer, feature_slices = build_datasets(cfg)
    dataset = train_ds if args.split == "train" else val_ds
    data_on_gpu = try_preload_one_dataset(dataset, device, cfg.training.preload_to_gpu)
    nw, pm = dataloader_io_options(device, data_on_gpu, cfg.training.num_workers)
    print(
        f"[compare-overfit] split={args.split} windows={len(dataset)} "
        f"batch_size={cfg.training.batch_size} feature_dim={feature_slices.total_dim}"
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pm,
        drop_last=False,
    )
    fixed_batch = get_fixed_batch(loader, args.batch_index)
    print(
        "[compare-overfit] "
        f"fixed_batch_shape={tuple(fixed_batch.shape)} batch_index={args.batch_index}"
    )

    plain_model = PlainMotionAutoEncoder(
        feature_dim=feature_slices.total_dim,
        window_size=cfg.window_size,
        encoder_hidden_dims=cfg.model.encoder_hidden_dims,
        decoder_hidden_dims=cfg.model.decoder_hidden_dims,
        latent_dim=cfg.model.latent_dim,
        activation=cfg.model.activation,
        use_layer_norm=cfg.model.use_layer_norm,
    )
    ifsq_model = MotionAutoEncoder(
        feature_dim=feature_slices.total_dim,
        window_size=cfg.window_size,
        encoder_hidden_dims=cfg.model.encoder_hidden_dims,
        decoder_hidden_dims=cfg.model.decoder_hidden_dims,
        ifsq_levels=cfg.model.ifsq_levels,
        activation=cfg.model.activation,
        use_layer_norm=cfg.model.use_layer_norm,
    )
    criterion = ReconstructionLoss(
        group_slices=feature_slices.as_dict(),
        group_weights=cfg.loss.group_weights,
    )

    payload: Dict[str, Any] = {
        "meta": {
            "config": args.config,
            "split": args.split,
            "steps": args.steps,
            "batch_index": args.batch_index,
            "batch_shape": list(fixed_batch.shape),
            "device": str(device),
            "learning_rate": cfg.training.learning_rate,
            "weight_decay": cfg.training.weight_decay,
        }
    }
    payload.update(
        compare_models_on_batch(
            batch=fixed_batch,
            criterion=criterion,
            plain_model=plain_model,
            ifsq_model=ifsq_model,
            device=device,
            steps=args.steps,
            log_every=args.log_every,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
    )

    print(
        "[plain-ae] summary "
        f"first={payload['plain_ae']['summary']['first']:.6f} "
        f"last={payload['plain_ae']['summary']['last']:.6f} "
        f"min={payload['plain_ae']['summary']['min']:.6f}"
    )
    print(
        "[ifsq-ae] summary "
        f"first={payload['ifsq_ae']['summary']['first']:.6f} "
        f"last={payload['ifsq_ae']['summary']['last']:.6f} "
        f"min={payload['ifsq_ae']['summary']['min']:.6f}"
    )

    history_path = save_history_json(payload, args.history_json)
    print(f"[compare-overfit] history_json={history_path}")


if __name__ == "__main__":
    main()
