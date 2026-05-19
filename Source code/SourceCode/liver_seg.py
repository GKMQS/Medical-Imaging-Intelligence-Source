import argparse

from do_liver.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from do_liver.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

import numpy as np
import torch
import cv2


def liver_seg(img):

    img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_CUBIC)
    vit_name = 'R50-ViT-B_16'
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 2
    config_vit.n_skip = 3
    config_vit.patches.size = (16, 16)
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(400 / 16), int(400 / 16))
    net = ViT_seg(config_vit, img_size=400, num_classes=config_vit.n_classes)

    snapshot = './do_liver/liver_model/epoch_149.pth'

    net.load_state_dict(torch.load(snapshot, map_location=torch.device('cpu')))
    # print(net)
    input = torch.from_numpy(img).unsqueeze(
        0).unsqueeze(0).float()
    # print(input)
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()
        img = prediction
        print(img.shape)

    # img = org_img+img
    img = abs(img) / abs(img).max()
    img = img * 255
    print(img.max())
    img = img.astype(np.uint8)
    cv2.imwrite('out.png', img)
    return img

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='McMRSR')
        parser.add_argument('--path', type=str, default='/Users/yymacpro13/Desktop/Game/TZB/data/liver_seg/liver_47.png',
                            help='give a experiment name before training')
        opts = parser.parse_args()
        img = cv2.imread(opts.path)
        img = img[:,:,0]
        print(img.shape)
        liver_seg(img)