#!/usr/bin/env python3

import sys
import os
# Add the project root to Python path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.append(proj_root)

from option import args
from data import dataset_DALE
from torch.utils.data import DataLoader
from model.VisualAttentionNetwork import VisualAttentionNetwork
from model.EnhancementNet import EnhancementNet, Discriminator
from train import train_utils
from collections import OrderedDict
import numpy as np
from loss import ploss, tvloss
import visdom
import PIL.Image as Image
from torchvision import transforms
import torch.nn as nn
import torch

# Loss functions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
L2_loss = nn.MSELoss().to(device)
Perceptual_loss = ploss.PerceptualLoss().to(device)
TvLoss = tvloss.TVLoss().to(device)

# Visdom setup (graceful fallback)
try:
    viz = visdom.Visdom(env="DALEGAN")
    if not viz.check_connection():
        print("[Visdom] server not reachable, disabling visualizations.")
        viz = None
except Exception:
    print("[Visdom] import or connect error, disabling visualizations.")
    viz = None

loss_data = {'X': [], 'Y': [], 'legend_U': ['e_loss','tv_loss','p_loss','g_loss','d_loss']}


def visdom_loss(loss_step, loss_dict):
    if viz is None:
        return
    loss_data['X'].append(loss_step)
    loss_data['Y'].append([loss_dict[k] for k in loss_data['legend_U']])
    viz.line(
        X=np.stack([np.array(loss_data['X'])] * len(loss_data['legend_U']), 1),
        Y=np.array(loss_data['Y']),
        win=1,
        opts=dict(xlabel='Step', ylabel='Loss', title='Training loss', legend=loss_data['legend_U']),
        update='append'
    )


def visdom_image(img_dict, window):
    if viz is None:
        return
    for idx, key in enumerate(img_dict):
        win = window + idx
        tensor_img = train_utils.tensor2im(img_dict[key].data)
        viz.image(tensor_img.transpose([2, 0, 1]), opts=dict(title=key), win=win)


def test(args, loader_test, model_AttentionNet, epoch, root_dir):
    model_AttentionNet.eval()
    for itr, (testImg, fileName) in enumerate(loader_test):
        if args.cuda:
            testImg = testImg.cuda()
        with torch.no_grad():
            test_result = model_AttentionNet(testImg)
            test_result_img = train_utils.tensor2im(test_result)
            save_path = os.path.join(
                root_dir,
                f"{os.path.splitext(fileName[0])[0]}_epoch_{epoch}_itr_{itr}.png"
            )
            train_utils.save_images(test_result_img, save_path)


def main(args):
    # Clamp to max 50 epochs
    args.epochs = min(getattr(args, 'epochs', 200), 50)
    args.cuda = torch.cuda.is_available()
    args.lr = 1e-5
    args.batch_size = 5

    # Paths
    train_data_root = os.path.join(proj_root, 'dataset', 'TRAIN')
    model_save_root_dir = os.path.join(proj_root, 'checkpoint')
    model_root = os.path.join(model_save_root_dir, 'DALEGAN')

    VISUALIZATION_STEP = 50
    SAVE_STEP = 1

    print(f"Training for {args.epochs} epochs (max 50)")
    print("DALE => Data Loading")
    train_data = dataset_DALE.DALETrainGlobal(train_data_root, args)
    loader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    print("DALE => Model Building")
    VAN = VisualAttentionNetwork().to(device)
    van_ckpt = os.path.join(model_root, 'visual_attention_network_model.pth')
    if os.path.isfile(van_ckpt):
        VAN.load_state_dict(torch.load(van_ckpt, map_location=device))

    G = EnhancementNet().to(device)
    D = Discriminator().to(device)
    gan_ckpt = os.path.join(model_root, 'enhance_GAN.pth')
    if os.path.isfile(gan_ckpt):
        G.load_state_dict(torch.load(gan_ckpt, map_location=device))

    print("DALE => Set Optimization")
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    print("DALE => Training")
    loss_step = 0
    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        for itr, (low_light_img, ground_truth_img, gt_Attention_img, file_name) in enumerate(loader_train, 1):
            if args.cuda:
                low_light_img = low_light_img.cuda()
                ground_truth_img = ground_truth_img.cuda()
                gt_Attention_img = gt_Attention_img.cuda()

            # Discriminator update
            optD.zero_grad()
            attn = VAN(low_light_img)
            fake_detached = G(low_light_img, attn).detach()
            loss_D = -D(ground_truth_img).mean() + D(fake_detached).mean()
            loss_D.backward(); optD.step()
          
            # Clamp weights for WGAN
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

            # Generator update every 5 iterations
            if itr % 5 == 0:
                optG.zero_grad()
                attn = VAN(low_light_img)
                out = G(low_light_img, attn)
                loss_G = -D(out).mean() * 0.5
                e_loss = L2_loss(out, ground_truth_img)
                p_loss = Perceptual_loss(out, ground_truth_img) * 10
                tv_loss = TvLoss(out) * 5
                total_loss = e_loss + p_loss + tv_loss + loss_G
                total_loss.backward(); optG.step()

            # Visualization
            if itr % VISUALIZATION_STEP == 0:
                print(f"Epoch[{epoch}/{args.epochs}]({itr}/{len(loader_train)}): "
                      f"e_loss:{e_loss:.6f}, tv_loss:{tv_loss:.6f}, p_loss:{p_loss:.6f}")
                loss_dict = {
                    'e_loss': e_loss.item(),
                    'tv_loss': tv_loss.item(),
                    'p_loss': p_loss.item(),
                    'g_loss': loss_G.item(),
                    'd_loss': loss_D.item()
                }
                visdom_loss(loss_step, loss_dict)

                # only visualize validation if file exists
                val_path = os.path.join(proj_root, 'validation', '15.jpg')
                if os.path.isfile(val_path) and viz is not None:
                    with torch.no_grad():
                        v_img = Image.open(val_path).convert('RGB')
                        v_t = transforms.ToTensor()(v_img).unsqueeze(0).to(device)
                        val_att = VAN(v_t)
                        val_out = G(v_t, val_att)
                    img_list = OrderedDict([
                        ('input', low_light_img),
                        ('output', out),
                        ('attention_output', attn),
                        ('gt_Attention_img', gt_Attention_img),
                        ('ground_truth', ground_truth_img),
                        ('val_result', val_out)
                    ])
                    visdom_image(img_list, window=10)

                loss_step += 1

        print("DALE => Testing")
        if epoch % SAVE_STEP == 0 or epoch == args.epochs:
            save_g = os.path.join(model_save_root_dir, 'DALEGAN')
            save_d = os.path.join(model_save_root_dir, 'DALE_Discriminator')
            train_utils.save_checkpoint(G, epoch, save_g)
            train_utils.save_checkpoint(D, epoch, save_d)
            print(f"Saved GAN checkpoints for epoch {epoch}")

if __name__ == "__main__":
    opt = args
    main(opt)
