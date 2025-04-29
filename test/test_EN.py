#!/usr/bin/env python3
"""
Script to test (inference) the trained EN (EnhancementNet) model from the DALE repository.
Since the EN model expects both an input image and an attention map, but you only have the EN model,
we’ll provide a dummy zero attention map so you can run inference on your low-light images.
Now accepts a directory of checkpoints and runs inference on each, saving outputs per checkpoint.
"""
import os
import sys
import argparse
import glob
import torch
from torch.utils.data import DataLoader

# Ensure project root is on Python path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, proj_root)

from model.EnhancementNet import EnhancementNet
from data.dataset_DALE import DALETest
from train.train_utils import tensor2im, save_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test EN model inference on low-light images, single checkpoint or directory of checkpoints."
    )
    parser.add_argument(
        '--checkpoint', '-c', type=str, required=True,
        help='Path to a single EN checkpoint file or a directory containing multiple .pth files'
    )
    parser.add_argument(
        '--data_dir', '-d', type=str,
        default=os.path.join(proj_root, 'dataset', 'TEST'),
        help='Directory containing test images'
    )
    parser.add_argument(
        '--output_dir', '-o', type=str,
        default=os.path.join(proj_root, 'test_results'),
        help='Base directory where enhanced images will be saved'
    )
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda','cpu'],
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
        ckpt_paths = sorted(glob.glob(os.path.join(args.checkpoint, "*.pth")))
        if not ckpt_paths:
            print(f"No .pth files found in directory {args.checkpoint}")
            sys.exit(1)
    else:
        ckpt_paths = [args.checkpoint]

    device = torch.device(args.device)

    # Iterate over each checkpoint
    for ckpt_path in ckpt_paths:
        # Load model and checkpoint
        model = EnhancementNet().to(device).eval()
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt)
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]

        # Prepare output sub-directory
        ckpt_out_dir = os.path.join(args.output_dir, ckpt_name)
        os.makedirs(ckpt_out_dir, exist_ok=True)

        print(f"\n=== Testing checkpoint '{ckpt_name}' on device {device} ===")

        # Re-create loader for each checkpoint to avoid exhaustion
        loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        with torch.no_grad():
            for img_tensor, filename in loader:
                img_tensor = img_tensor.to(device)
                # Dummy attention map of zeros
                att_tensor = torch.zeros_like(img_tensor)
                # Run inference
                output = model(img_tensor, att_tensor)
                img_np = tensor2im(output)

                base, _ = os.path.splitext(filename[0])
                save_path = os.path.join(ckpt_out_dir, f"{base}_enhanced.png")
                save_images(img_np, save_path)
                print(f"Saved enhanced image: {save_path}")

if __name__ == '__main__':
    main()
