import torch.nn as nn
import numpy as np
from utils import arange
from networks.networks import DRDN
import pdb
from networks.network_swinir import SwinIR as net
from networks.unet import UNet
# from networks.vit_seg_modeling import VisionTransformer as ViT_seg
# from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from networks.uca import MICCANlong



def set_gpu(network, gpu_ids):
    network.to(gpu_ids[0])
    network = nn.DataParallel(network, device_ids=gpu_ids)

    return network


# self.kspace_Unet = net(upscale=1, in_chans=4, img_size=256, window_size=8,
#                     img_range=1., depths=[3, 3, 3,3], embed_dim=180, num_heads=[3,3,3,3],
#                     mlp_ratio=2, upsampler='', resi_connection='1conv')



def get_generator(name, opts):

    if name == 'DRDN':
        ic = 2
        if opts.use_prior:
            ic = ic + 2
        network = DRDN(n_channels=ic, G0=32, kSize=3, D=3, C=4, G=32, dilateSet=[1,2,3,3])
    elif name =='SwinIR':
        network = net(upscale=1, in_chans=2, img_size=256, window_size=8,
                           img_range=1., depths=[3,3], embed_dim=90, num_heads=[3,3],
                           mlp_ratio=2, upsampler='', resi_connection='1conv')
    elif name =='unet':
        network = UNet(n_channels=1, n_classes=1, bilinear=True)
    # elif name == 'transunet':
    #     config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    #     config_vit.n_skip = 3
    #     if 'R50-ViT-B_16'.find('R50') != -1:
    #         config_vit.patches.grid = (
    #         int(256 / 16), int(256 / 16))
    #     network = ViT_seg(config_vit, img_size=256, num_classes=1)
    elif name =='uca':
        network = MICCANlong(2, 2, 5, 'uca')
    else:
        raise NotImplementedError

    num_param = sum([p.numel() for p in network.parameters() if p.requires_grad])
    print('Number of parameters: {}'.format(num_param))
    return set_gpu(network, opts.gpu_ids)