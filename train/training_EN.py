import sys
import os
# Add the root directory of the project to the Python path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from option import args
from data import dataset_DALE
from torch.utils.data import DataLoader
from model import VisualAttentionNetwork, EnhancementNet
from train import train_utils
from collections import OrderedDict
import numpy as np
from loss import ploss, tvloss
import visdom
import PIL.Image as Image
from torchvision import transforms
import torch.nn as nn
import torch
from torch.optim import lr_scheduler

# Ensure os is available for path checks
# Setting Loss
L2_loss = nn.MSELoss().cuda()
Perceptual_loss = ploss.PerceptualLoss().cuda()
TvLoss = tvloss.TVLoss().cuda()

# Setting Visdom (optional)
vis = None
try:
    vis = visdom.Visdom(env="DALE_EN")
except Exception as e:
    print(f"[Visdom] Could not connect: {e}")
loss_data = {'X': [], 'Y': [], 'legend_U': ['mse_loss', 'tv_loss', 'p_loss']}

# Visualization helpers
def visdom_loss(visdom_obj, loss_step, loss_dict):
    try:
        loss_data['X'].append(loss_step)
        loss_data['Y'].append([loss_dict[k] for k in loss_data['legend_U']])
        visdom_obj.line(
            X=np.stack([np.array(loss_data['X'])] * len(loss_data['legend_U']), 1),
            Y=np.array(loss_data['Y']),
            win=1,
            opts=dict(xlabel='Step', ylabel='Loss', title='Training loss', legend=loss_data['legend_U']),
            update='append'
        )
    except Exception as e:
        print(f"[Visdom] Skipping loss graph: {e}")


def visdom_image(visdom_obj, img_dict, window):
    for idx, key in enumerate(img_dict):
        try:
            win = window + idx
            # convert tensor to numpy image
            tensor_img = train_utils.tensor2im(img_dict[key].detach())
            visdom_obj.image(tensor_img.transpose([2, 0, 1]), opts=dict(title=key), win=win)
        except Exception as e:
            print(f"[Visdom] Skipping image '{key}': {e}")

# Test function remains unchanged
def test(args, loader_test, model_AttentionNet, epoch, root_dir):
    model_AttentionNet.eval()
    for itr, data in enumerate(loader_test):
        testImg, fileName = data
        if args.cuda:
            testImg = testImg.cuda()
        with torch.no_grad():
            test_result = model_AttentionNet(testImg)
            test_result_img = train_utils.tensor2im(test_result)
            base, _ = os.path.splitext(fileName[0])
            result_save_dir = os.path.join(root_dir, f"{base}_epoch_{epoch}_itr_{itr}.png")
            train_utils.save_images(test_result_img, result_save_dir)

# Main training loop

def main(args):
    args.cuda = True
    args.epochs = 200
    args.lr = 1e-5
    args.batch_size = 4

    # Paths
    train_data_root = '/home/snu/Desktop/eklavya/DALE/dataset/TRAIN/'
    model_save_root_dir = './checkpoint/DALE'
    model_root = './checkpoint/'

    VIS_STEP = 50
    SAVE_STEP = 1

    print("DALE => Data Loading")
    train_data = dataset_DALE.DALETrain(train_data_root, args)
    loader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    print("DALE => Model Building")
    VisualAttentionNet = VisualAttentionNetwork.VisualAttentionNetwork()
    van_path = os.path.join(model_root, 'VAN.pth')
    if os.path.isfile(van_path):
        state_dict = torch.load(van_path, map_location='cpu')
        VisualAttentionNet.load_state_dict(state_dict)
    else:
        print(f"Warning: VAN.pth not found at {van_path}")
    EnhanceNet = EnhancementNet.EnhancementNet()

    print("DALE => Set Optimization")
    optG = torch.optim.Adam(EnhanceNet.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = lr_scheduler.ExponentialLR(optG, gamma=0.99)

    params1 = sum(p.numel() for p in EnhanceNet.parameters() if p.requires_grad)
    print("Parameters |", params1)

    if args.cuda:
        print("DALE => Use GPU")
        VisualAttentionNet = VisualAttentionNet.cuda()
        EnhanceNet = EnhanceNet.cuda()

    loss_step = 0
    print("DALE => Training")

    for epoch in range(1, args.epochs + 1):
        EnhanceNet.train()
        for itr, (low, gt, att, fname) in enumerate(loader_train):
            if args.cuda:
                low, gt, att = low.cuda(), gt.cuda(), att.cuda()
            optG.zero_grad()

            attention_result = VisualAttentionNet(low)
            enhance_result = EnhanceNet(low, attention_result.detach())

            mse_loss = L2_loss(enhance_result, gt)
            p_loss = Perceptual_loss(enhance_result, gt) * 50
            tv_loss = TvLoss(enhance_result) * 20
            total_loss = mse_loss + p_loss + tv_loss

            total_loss.backward()
            optG.step()

            if epoch > 100 and itr == 0:
                scheduler.step()
                print("LR =>", scheduler.get_last_lr())

            if itr != 0 and itr % VIS_STEP == 0:
                print(f"Epoch[{epoch}/{args.epochs}]({itr}/{len(loader_train)}): "
                      f"mse:{mse_loss:.6f}, tv:{tv_loss:.6f}, p:{p_loss:.6f}")
                loss_dict = {'mse_loss': mse_loss.item(), 'tv_loss': tv_loss.item(), 'p_loss': p_loss.item()}
                if vis:
                    visdom_loss(vis, loss_step, loss_dict)

                # Optional validation image
                val_path = os.path.abspath(os.path.join(root_path, '..', 'validation', '15.jpg'))
                if os.path.isfile(val_path) and vis:
                    with torch.no_grad():
                        v_img = Image.open(val_path).convert('RGB')
                        v_tensor = transforms.ToTensor()(v_img).unsqueeze(0)
                        if args.cuda: v_tensor = v_tensor.cuda()
                        v_att = VisualAttentionNet(v_tensor)
                        v_out = EnhanceNet(v_tensor, v_att)
                    img_list = OrderedDict([
                        ('input', low), ('output', enhance_result),
                        ('attention', attention_result), ('gt', gt), ('val_out', v_out)
                    ])
                    visdom_image(vis, img_list, window=10)
                loss_step += 1

        print("DALE => Saving Checkpoint")
        if epoch % SAVE_STEP == 0:
            train_utils.save_checkpoint(EnhanceNet, epoch, model_save_root_dir)

if __name__ == "__main__":
    main(args)
