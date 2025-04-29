#!/usr/bin/env python3
"""
Script to test (inference) the trained VAN (VisualAttentionNetwork) model from the DALE repository.
Loads a VAN checkpoint, runs on all images in a test folder, and saves the visual attention map and summed image.
"""
import os
import sys
import torch
from torch.utils.data import DataLoader

# Ensure project root is on Python path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, proj_root)

from model.VisualAttentionNetwork import VisualAttentionNetwork
from data.dataset_DALE import DALETest
from train.train_utils import tensor2im, save_images


def main():
    # Paths (adjust these if your folders differ)
    test_data_root = os.path.join(proj_root, 'dataset', 'DarkPair', 'ExDark', 'Bicycle')
    checkpoint_path = os.path.join(proj_root, 'checkpoint', 'DALE_VAN', 'VAN_epoch_200.pth')
    output_dir = os.path.join(proj_root, 'VAN_TEST')
    os.makedirs(output_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    net = VisualAttentionNetwork().to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint)
    print(f"Loaded VAN checkpoint from {checkpoint_path}")

    # Prepare test dataset
    test_ds = DALETest(test_data_root)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    # Inference loop
    with torch.no_grad():
        for img_tensor, filenames in loader:
            img_tensor = img_tensor.to(device)
            # Predict attention map
            att_map = net(img_tensor)

            # Convert to uint8 images
            att_img_np = tensor2im(att_map)
            sum_img_np = tensor2im(img_tensor + att_map)

            # Save outputs
            base = os.path.splitext(filenames[0])[0]
            save_path1 = os.path.join(output_dir, f"visual_attention_map_{base}.png")
            save_path2 = os.path.join(output_dir, f"sum_{base}.png")
            save_images(att_img_np, save_path1)
            save_images(sum_img_np, save_path2)
            print(f"Saved: {save_path1}\n       {save_path2}")

if __name__ == '__main__':
    main()
