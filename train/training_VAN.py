#!/usr/bin/env python3

import os
import sys
# Ensure the project root is on Python path so we can import option.py
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, proj_root)

from option import args
from data.dataset_DALE import DALETrain, DALETest
from torch.utils.data import DataLoader
from model.VisualAttentionNetwork import VisualAttentionNetwork
from train.train_utils import tensor2im, save_images
from collections import OrderedDict
import numpy as np
import visdom
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from loss.ploss import PerceptualLoss
from loss.tvloss import TVLoss
from torchvision import transforms
from PIL import Image

# Create checkpoint directory if it doesn't exist
model_save_root_dir = os.path.join(proj_root, 'checkpoint', 'DALE_VAN')
os.makedirs(model_save_root_dir, exist_ok=True)

# Loss functions
L1_loss = nn.L1Loss().cuda()
Perceptual_loss = PerceptualLoss().cuda()
TvLoss = TVLoss().cuda()

# Visdom setup
vis = visdom.Visdom(env="DALE_VAN")
loss_plot = {'X': [], 'Y': [], 'legend': ['mse', 'p']}


def plot_losses(step, losses):
    loss_plot['X'].append(step)
    loss_plot['Y'].append([losses['mse'], losses['p']])
    vis.line(
        X=np.stack([np.array(loss_plot['X'])]*2, axis=1),
        Y=np.array(loss_plot['Y']),
        win=1,
        opts=dict(title='Training Loss', legend=loss_plot['legend'], xlabel='Step', ylabel='Loss'),
        update='append'
    )


def plot_images(batch, val_batch=None):
    imgs = OrderedDict()
    imgs['input'] = batch['low']
    imgs['pred'] = batch['out']
    imgs['attn'] = batch['att']
    if val_batch:
        imgs['val_input'] = val_batch['low']
        imgs['val_pred'] = val_batch['out']
    for i, (title, tensor_img) in enumerate(imgs.items()):
        img_np = tensor2im(tensor_img.data)
        vis.image(img_np.transpose(2,0,1), opts=dict(title=title), win=10+i)


def main(opt):
    # Override args
    opt.cuda = True
    opt.batch_size = 8
    opt.lr = 1e-5
    # Set number of epochs to 50
    opt.epochs = 50

    # Determine save interval to limit to 50 checkpoints max
    max_saves = 50
    if opt.epochs <= max_saves:
        save_interval = 1
    else:
        save_interval = opt.epochs // max_saves

    # Paths
    train_root = os.path.join(proj_root, 'dataset', 'TRAIN')
    test_root  = os.path.join(proj_root, 'dataset', 'TEST')

    # Datasets & loaders
    train_ds = DALETrain(train_root, opt)
    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_threads)

    test_ds = DALETest(test_root)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Model, optimizer, scheduler
    net = VisualAttentionNetwork().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # Prepare one test batch for visualization
    val_batch = None
    if len(test_ds) > 0:
        img, name = test_ds[0]
        val_batch = {'low': img.unsqueeze(0).cuda()}
        with torch.no_grad():
            val_batch['out'] = net(val_batch['low'])

    step = 0
    for epoch in range(1, opt.epochs+1):
        net.train()
        for itr, (low, gt, att, _) in enumerate(train_loader, 1):
            low, gt, att = low.cuda(), gt.cuda(), att.cuda()
            optimizer.zero_grad()
            out = net(low)
            mse = L1_loss(out, att)
            p_loss = Perceptual_loss(out, att) * 10
            loss = mse + p_loss
            loss.backward()
            optimizer.step()

            if epoch > 10 and itr == 1:
                scheduler.step()
                print(f"LR updated to {scheduler.get_last_lr()}")

            if itr % 100 == 0:
                print(f"Epoch {epoch}/{opt.epochs} | Iter {itr}/{len(train_loader)} | mse={mse.item():.4f} p={p_loss.item():.4f}")
                plot_losses(step, {'mse': mse.item(), 'p': p_loss.item()})
                batch_data = {'low': low, 'out': out, 'att': att}
                if val_batch:
                    with torch.no_grad():
                        val_batch['out'] = net(val_batch['low'])
                plot_images(batch_data, val_batch)
                step += 1

        # Save checkpoint only at the defined interval
        if epoch % save_interval == 0 or epoch == opt.epochs:
            ckpt_file = os.path.join(model_save_root_dir, f"VAN_epoch_{epoch}.pth")
            torch.save(net.state_dict(), ckpt_file)
            print(f"Saved checkpoint: {ckpt_file}")

if __name__ == '__main__':
    main(args)
