"""
Export trained model to ONNX for mobile/edge deployment.

Usage:
    python export.py --checkpoint outputs/best_model.pth --output exports/model.onnx

After export, you can use the ONNX model in:
- Android (ONNX Runtime Mobile)
- iOS (CoreML via onnx2coreml or ONNX Runtime)
- Edge devices (Raspberry Pi, Jetson Nano, etc.)
"""
import argparse
from pathlib import Path

import torch
import onnx
import yaml

from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--output', default='exports/model.onnx')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--simplify', action='store_true', default=True)
    return parser.parse_args()


def export_onnx(model, output_path: str, image_size: int, simplify: bool):
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=['image'],
        output_names=['logits'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
        },
        opset_version=17,
    )

    if simplify:
        try:
            import onnxsim
            model_onnx = onnx.load(output_path)
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'Simplified ONNX model could not be validated'
            onnx.save(model_onnx, output_path)
            print('ONNX model simplified successfully.')
        except ImportError:
            print('onnxsim not installed, skipping simplification.')

    # Print model size
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f'Exported ONNX model: {output_path}  ({size_mb:.1f} MB)')


def main():
    args = parse_args()
    device = torch.device('cpu')  # ONNX export on CPU

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt['cfg']
    print(f'Loaded checkpoint from epoch {ckpt["epoch"]} '
          f'(val_acc={ckpt["val_acc"]:.2f}%)')

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt['model'])

    export_onnx(model, args.output, args.image_size, args.simplify)


if __name__ == '__main__':
    main()
