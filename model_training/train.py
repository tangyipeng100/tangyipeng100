"""
Main training script for small-sample image classification.

Usage:
    # Standard training
    python train.py --config configs/default.yaml

    # Few-shot (5-shot) training
    python train.py --config configs/fewshot.yaml

    # Override config via CLI
    python train.py --config configs/default.yaml --shots 10 --epochs 50
"""
import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from data import build_dataloader
from models import build_model
from utils import compute_metrics, AverageMeter, setup_logger
from utils.metrics import accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Small-sample model training')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--shots', type=int, help='Override shots_per_class')
    parser.add_argument('--epochs', type=int, help='Override epochs')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--output', default='outputs/', help='Output directory')
    parser.add_argument('--resume', default='', help='Resume from checkpoint')
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg['training']['optimizer'] == 'adamw':
        return torch.optim.AdamW(
            params,
            lr=cfg['training']['lr'],
            weight_decay=cfg['training']['weight_decay'],
        )
    return torch.optim.SGD(
        params, lr=cfg['training']['lr'],
        momentum=0.9, weight_decay=cfg['training']['weight_decay'],
    )


def build_scheduler(optimizer, cfg, steps_per_epoch):
    sched = cfg['training'].get('scheduler', 'cosine')
    total_steps = cfg['training']['epochs'] * steps_per_epoch
    warmup_steps = cfg['training'].get('warmup_epochs', 0) * steps_per_epoch

    if sched == 'cosine':
        from torch.optim.lr_scheduler import OneCycleLR
        return OneCycleLR(
            optimizer,
            max_lr=cfg['training']['lr'],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps if total_steps > 0 else 0.1,
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['training']['epochs'],
    )


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, logger):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        top1 = accuracy(logits, labels, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(top1, images.size(0))

        if batch_idx % 10 == 0:
            logger.debug(
                f'  Step [{batch_idx}/{len(loader)}] '
                f'loss={loss_meter.avg:.4f} acc={acc_meter.avg:.2f}%'
            )

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        loss_meter.update(loss.item(), images.size(0))
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    top1 = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) * 100
    return loss_meter.avg, top1, all_preds, all_labels


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.shots:
        cfg['data']['shots_per_class'] = args.shots
    if args.epochs:
        cfg['training']['epochs'] = args.epochs
    if args.lr:
        cfg['training']['lr'] = args.lr

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger('train', log_file=str(output_dir / 'train.log'))
    logger.info(f'Config: {args.config}')
    logger.info(f'Shots per class: {cfg["data"].get("shots_per_class", "all")}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    train_loader = build_dataloader(cfg, 'train')
    val_loader = build_dataloader(cfg, 'val')
    logger.info(
        f'Dataset — train: {len(train_loader.dataset)} | '
        f'val: {len(val_loader.dataset)} | '
        f'classes: {train_loader.dataset.classes}'
    )

    model = build_model(cfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f'Model params: {trainable:,} trainable / {total:,} total')

    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg['training'].get('label_smoothing', 0.0)
    )
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        logger.info(f'Resumed from {args.resume}')

    best_acc = 0.0
    patience = cfg['training'].get('early_stopping_patience', 20)
    no_improve = 0

    for epoch in range(1, cfg['training']['epochs'] + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, logger
        )
        val_loss, val_acc, preds, labels = evaluate(
            model, val_loader, criterion, device
        )

        if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f'Epoch [{epoch:03d}/{cfg["training"]["epochs"]}] '
            f'time={elapsed:.1f}s  '
            f'train_loss={train_loss:.4f} train_acc={train_acc:.2f}%  '
            f'val_loss={val_loss:.4f} val_acc={val_acc:.2f}%'
        )

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            ckpt_path = output_dir / 'best_model.pth'
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'val_acc': val_acc, 'cfg': cfg}, ckpt_path)
            logger.info(f'  -> Saved best model (acc={best_acc:.2f}%)')
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break

    logger.info(f'Training complete. Best val acc: {best_acc:.2f}%')

    # Full report on best model
    best_ckpt = torch.load(output_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(best_ckpt['model'])
    _, _, preds, labels = evaluate(model, val_loader, criterion, device)
    report, _ = compute_metrics(preds, labels, train_loader.dataset.classes)
    logger.info(f'\nClassification Report:\n{report}')


if __name__ == '__main__':
    main()
