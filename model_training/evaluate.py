"""
Evaluate a trained checkpoint and print a full classification report.

Usage:
    python evaluate.py --checkpoint outputs/best_model.pth --data data/test
"""
import argparse
import torch
import yaml

from data.dataset import SmallSampleDataset, build_transforms
from torch.utils.data import DataLoader
from models import build_model
from utils import compute_metrics, setup_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True, help='Test data directory')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger('evaluate')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt['cfg']

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    transform = build_transforms(args.image_size, augment=False, is_train=False)
    dataset = SmallSampleDataset(args.data, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    logger.info(f'Evaluating on {len(dataset)} samples, {len(dataset.classes)} classes')

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    report, cm = compute_metrics(all_preds, all_labels, dataset.classes)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    logger.info(f'Overall accuracy: {acc * 100:.2f}%')
    logger.info(f'\nClassification Report:\n{report}')
    logger.info(f'Confusion matrix:\n{cm}')


if __name__ == '__main__':
    main()
