#!/usr/bin/env python3
"""
Script to test (inference) the trained VAN (VisualAttentionNetwork) model from the DALE repository.
Supports testing a single checkpoint or an entire directory of checkpoints, saving the predicted attention maps for each input image.
"""
import os
import sys
import argparse
import glob
import torch
from torch.utils.data import DataLoader

# Ensure project root is on Python path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from model.VisualAttentionNetwork import VisualAttentionNetwork
from data.dataset_DALE import DALETest
from train.train_utils import tensor2im, save_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test VAN model inference on low-light images, single checkpoint or directory."
    )
    parser.add_argument(
        '--checkpoint', '-c', type=str, required=True,
        help='Path to a single VAN checkpoint file or a directory containing multiple .pth files'
    )
    parser.add_argument(
        '--data_dir', '-d', type=str,
        default=os.path.join(proj_root, 'dataset', 'TEST'),
        help='Directory containing test images'
    )
    parser.add_argument(
        '--output_dir', '-o', type=str,
        default=os.path.join(proj_root, 'van_results'),
        help='Base directory where attention maps will be saved'
    )
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to run inference on'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare dataset once
    test_ds = DALETest(args.data_dir)

    # Identify checkpoint paths
    if os.path.isdir(args.checkpoint):
        ckpt_paths = sorted(glob.glob(os.path.join(args.checkpoint, '*.pth')))
        if not ckpt_paths:
            print(f"No .pth files found in directory {args.checkpoint}")
            sys.exit(1)
    else:
        ckpt_paths = [args.checkpoint]

    device = torch.device(args.device)

    for ckpt_path in ckpt_paths:
        # Load model and checkpoint
        model = VisualAttentionNetwork().to(device).eval()
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)

        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        out_dir = os.path.join(args.output_dir, ckpt_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== Testing checkpoint '{ckpt_name}' on device {device} ===")

        loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        with torch.no_grad():
            for img_tensor, filename in loader:
                img_tensor = img_tensor.to(device)
                # Run VAN to get attention map
                attn = model(img_tensor)
                # Convert to numpy image
                attn_np = tensor2im(attn)

                base, _ = os.path.splitext(filename[0])
                save_path = os.path.join(out_dir, f"{base}_attn.png")
                save_images(attn_np, save_path)
                print(f"Saved attention map: {save_path}")

if __name__ == '__main__':
    main()
