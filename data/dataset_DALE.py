# dataset_DALE.py

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from data import dataset_utils as utils

# Shared tensor conversion
_to_tensor = transforms.ToTensor()

class DALETrain(Dataset):
    def __init__(self, root_dir, args, patch_size=240):
        """
        root_dir should contain two subfolders: 'SuperPixel' and 'GT'
        args is passed to augmentation
        """
        super().__init__()
        self.low_light_dir = os.path.join(root_dir, 'SuperPixel')
        self.ground_truth_dir = os.path.join(root_dir, 'GT')

        self.low_light_img_list = sorted(os.listdir(self.low_light_dir))
        self.ground_truth_img_list = sorted(os.listdir(self.ground_truth_dir))

        assert len(self.low_light_img_list) == len(self.ground_truth_img_list), \
            f"Mismatch: {len(self.low_light_img_list)} inputs vs {len(self.ground_truth_img_list)} GTs"

        self.args = args
        self.patch_size = patch_size

    def __len__(self):
        return len(self.low_light_img_list)

    def __getitem__(self, idx):
        # Load & ensure RGB
        low_name = self.low_light_img_list[idx]
        gt_name  = self.ground_truth_img_list[idx]
        low_img = Image.open(os.path.join(self.low_light_dir, low_name)).convert('RGB')
        gt_img  = Image.open(os.path.join(self.ground_truth_dir, gt_name)).convert('RGB')

        # Resize small images to patch_size
        w, h = low_img.size
        if w < self.patch_size or h < self.patch_size:
            low_img = low_img.resize((self.patch_size, self.patch_size), Image.BICUBIC)
            gt_img  = gt_img.resize((self.patch_size, self.patch_size), Image.BICUBIC)

        # Random crop patch
        low_patch, gt_patch = utils.get_patch_low_light(low_img, gt_img, self.patch_size)

        # Augmentation
        low_patch, gt_patch = utils.augmentation_low_light(low_patch, gt_patch, self.args)

        # Convert patches to NumPy arrays
        buf1 = np.array(low_patch, dtype=np.uint8)
        buf2 = np.array(gt_patch, dtype=np.uint8)

        # Drop alpha channel if present
        if buf1.ndim == 3 and buf1.shape[2] == 4:
            buf1 = buf1[..., :3]
        if buf2.ndim == 3 and buf2.shape[2] == 4:
            buf2 = buf2[..., :3]

        # Expand grayscale to 3 channels
        if buf1.ndim == 2:
            buf1 = np.stack([buf1]*3, axis=2)
        if buf2.ndim == 2:
            buf2 = np.stack([buf2]*3, axis=2)

        # Attention map: positive difference
        att = np.clip(buf2.astype(np.int16) - buf1.astype(np.int16), 0, 255).astype(np.uint8)

        # Convert all to tensors [C, H, W], values in [0,1]
        low_tensor = _to_tensor(buf1)
        gt_tensor  = _to_tensor(buf2)
        att_tensor = _to_tensor(att)

        return low_tensor, gt_tensor, att_tensor, gt_name


class DALETrainGlobal(Dataset):
    def __init__(self, root_dir, args, patch_size=240):
        super().__init__()
        self.low_light_dir = os.path.join(root_dir, 'SuperPixel')
        self.ground_truth_dir = os.path.join(root_dir, 'GT')

        self.low_light_img_list = sorted(os.listdir(self.low_light_dir))
        self.ground_truth_img_list = sorted(os.listdir(self.ground_truth_dir))

        assert len(self.low_light_img_list) == len(self.ground_truth_img_list), \
            f"Mismatch: {len(self.low_light_img_list)} inputs vs {len(self.ground_truth_img_list)} GTs"

        self.args = args
        self.patch_size = patch_size

    def __len__(self):
        return len(self.low_light_img_list)

    def __getitem__(self, idx):
        low_name = self.low_light_img_list[idx]
        gt_name  = self.ground_truth_img_list[idx]
        low_img = Image.open(os.path.join(self.low_light_dir, low_name)).convert('RGB')
        gt_img  = Image.open(os.path.join(self.ground_truth_dir, gt_name)).convert('RGB')

        # Resize small images
        w, h = low_img.size
        if w < self.patch_size or h < self.patch_size:
            low_img = low_img.resize((self.patch_size, self.patch_size), Image.BICUBIC)
            gt_img  = gt_img.resize((self.patch_size, self.patch_size), Image.BICUBIC)

        # Global random crop
        low_patch, gt_patch = utils.get_patch_low_light_global(low_img, gt_img, self.patch_size)

        # Augmentation
        low_patch, gt_patch = utils.augmentation_low_light(low_patch, gt_patch, self.args)

        # Convert to NumPy and enforce 3 channels
        buf1 = np.array(low_patch, dtype=np.uint8)
        buf2 = np.array(gt_patch, dtype=np.uint8)
        if buf1.ndim == 3 and buf1.shape[2] == 4:
            buf1 = buf1[..., :3]
        if buf2.ndim == 3 and buf2.shape[2] == 4:
            buf2 = buf2[..., :3]
        if buf1.ndim == 2:
            buf1 = np.stack([buf1]*3, axis=2)
        if buf2.ndim == 2:
            buf2 = np.stack([buf2]*3, axis=2)

        # Attention map
        att = np.clip(buf2.astype(np.int16) - buf1.astype(np.int16), 0, 255).astype(np.uint8)

        # To tensors
        low_tensor = _to_tensor(buf1)
        gt_tensor  = _to_tensor(buf2)
        att_tensor = _to_tensor(att)

        return low_tensor, gt_tensor, att_tensor, gt_name


class DALETest(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.test_img_list = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.test_img_list)

    def __getitem__(self, idx):
        name = self.test_img_list[idx]
        img  = Image.open(os.path.join(self.root_dir, name)).convert('RGB')
        tensor = _to_tensor(img)
        return tensor, name
