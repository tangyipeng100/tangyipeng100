"""
Generate a toy demo dataset for testing the pipeline.
Creates synthetic images labeled into N classes.

Usage:
    python scripts/prepare_demo_data.py --classes 5 --shots 10 --output data/
"""
import argparse
import random
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np


COLORS = [
    (220, 50, 50), (50, 180, 50), (50, 50, 220),
    (200, 150, 30), (150, 50, 200), (50, 200, 200),
    (200, 100, 50), (100, 200, 50),
]

SHAPES = ['circle', 'rectangle', 'triangle', 'ellipse', 'line']


def draw_sample(class_idx: int, size: int = 224) -> Image.Image:
    img = Image.new('RGB', (size, size), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    color = COLORS[class_idx % len(COLORS)]
    shape = SHAPES[class_idx % len(SHAPES)]

    jitter = lambda v, d=20: v + random.randint(-d, d)
    cx, cy = size // 2, size // 2
    r = size // 3

    if shape == 'circle':
        draw.ellipse([jitter(cx - r), jitter(cy - r),
                      jitter(cx + r), jitter(cy + r)], fill=color)
    elif shape == 'rectangle':
        draw.rectangle([jitter(cx - r), jitter(cy - r),
                        jitter(cx + r), jitter(cy + r)], fill=color)
    elif shape == 'triangle':
        pts = [(jitter(cx), jitter(cy - r)),
               (jitter(cx - r), jitter(cy + r)),
               (jitter(cx + r), jitter(cy + r))]
        draw.polygon(pts, fill=color)
    elif shape == 'ellipse':
        draw.ellipse([jitter(cx - r), jitter(cy - r // 2),
                      jitter(cx + r), jitter(cy + r // 2)], fill=color)
    else:
        draw.line([jitter(cx - r), jitter(cy - r),
                   jitter(cx + r), jitter(cy + r)],
                  fill=color, width=20)

    # Add some noise
    arr = np.array(img).astype(np.float32)
    arr += np.random.normal(0, 10, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_split(output_dir: Path, n_classes: int, shots: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    for cls_idx in range(n_classes):
        cls_dir = output_dir / f'class_{cls_idx:02d}'
        cls_dir.mkdir(exist_ok=True)
        for i in range(shots):
            img = draw_sample(cls_idx)
            img.save(cls_dir / f'sample_{i:04d}.png')
    print(f'  Generated {n_classes * shots} images in {output_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', type=int, default=5)
    parser.add_argument('--shots', type=int, default=10,
                        help='Training samples per class')
    parser.add_argument('--val-shots', type=int, default=20,
                        help='Validation samples per class')
    parser.add_argument('--output', default='data/')
    args = parser.parse_args()

    root = Path(args.output)
    print(f'Generating demo dataset: {args.classes} classes')
    print(f'  Train: {args.shots} shots/class')
    generate_split(root / 'train', args.classes, args.shots)
    print(f'  Val: {args.val_shots} shots/class')
    generate_split(root / 'val', args.classes, args.val_shots)
    print('Done. Now run: python train.py --config configs/fewshot.yaml')


if __name__ == '__main__':
    main()
