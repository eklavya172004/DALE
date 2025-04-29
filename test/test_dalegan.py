#!/usr/bin/env python3
"""
Batch test for DALEGAN generator across multiple checkpoints using a fixed VAN model.
Loads a single VisualAttentionNetwork checkpoint (for attention maps) and iterates through all
generator checkpoints in a directory, running inference on every image in a test dataset.
Saves enhanced outputs per generator checkpoint.

Usage:
  cd <DALE_repo_root>
  source venv/bin/activate
  python test/test_GAN_all.py \
    --van_checkpoint checkpoint/DALE_VAN/VAN_epoch_200.pth \
    --gen_checkpoint_dir checkpoint/DALEGAN \
    --data_dir dataset \
    --output_dir GAN_all_results \
    --device cuda
"""
import os
import sys
import glob
import argparse
import torch
from torch.utils.data import DataLoader

# Ensure project root is on Python path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, proj_root)

from model.VisualAttentionNetwork import VisualAttentionNetwork
from model.EnhancementNet import EnhancementNet
from data.dataset_DALE import DALETest
from train.train_utils import tensor2im, save_images


def parse_args():
    parser = argparse.ArgumentParser(description="Batch test DALEGAN generator across checkpoints.")
    parser.add_argument('--van_checkpoint', '-v', type=str, required=True,
                        help='Path to the fixed VAN checkpoint for attention maps')
    parser.add_argument('--gen_checkpoint_dir', '-g', type=str, required=True,
                        help='Directory containing generator .pth checkpoints (EnhancementNet)')
    parser.add_argument('--data_dir', '-d', type=str,
                        default=os.path.join(proj_root, 'dataset'),
                        help='Directory containing test images')
    parser.add_argument('--output_dir', '-o', type=str,
                        default=os.path.join(proj_root, 'GAN_all_results'),
                        help='Base directory to save enhanced outputs')
    parser.add_argument('--device', choices=['cpu','cuda'],
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device for inference')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of DataLoader workers')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test dataset
    test_ds = DALETest(args.data_dir)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers)

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load fixed VAN model
    van = VisualAttentionNetwork().to(device).eval()
    van_state = torch.load(args.van_checkpoint, map_location=device)
    van.load_state_dict(van_state)
    print(f"Loaded VAN checkpoint: {args.van_checkpoint}")

    # Collect generator checkpoints
    gen_ckpts = sorted(glob.glob(os.path.join(args.gen_checkpoint_dir, '*.pth')))
    if not gen_ckpts:
        print(f"No generator checkpoints found in {args.gen_checkpoint_dir}")
        sys.exit(1)

    # Iterate generator checkpoints
    for gen_path in gen_ckpts:
        gen_name = os.path.splitext(os.path.basename(gen_path))[0]
        out_dir = os.path.join(args.output_dir, gen_name)
        os.makedirs(out_dir, exist_ok=True)

        # Load generator
        G = EnhancementNet().to(device).eval()
        G_state = torch.load(gen_path, map_location=device)
        G.load_state_dict(G_state)
        print(f"\nLoaded generator checkpoint: {gen_name}")

        # Inference loop
        with torch.no_grad():
            for img_tensor, filenames in loader:
                img = img_tensor.to(device)
                # Generate attention map and enhanced output
                att = van(img)
                enhanced = G(img, att)

                # Convert to numpy images
                enhanced_np = tensor2im(enhanced)

                # Save enhanced image
                base = os.path.splitext(filenames[0])[0]
                save_path = os.path.join(out_dir, f"{base}_enhanced.png")
                save_images(enhanced_np, save_path)
                print(f"Saved enhanced: {save_path}")

    print("\nAll generator checkpoints processed.")

if __name__ == '__main__':
    main()
