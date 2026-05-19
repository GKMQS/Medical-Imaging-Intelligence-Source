import os
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data
from datasets.utilizes import *
from models.utils import fft2, ifft2, to_tensor
import h5py

class MRIDataset_Cartesian(data.Dataset):
    def __init__(self, opts, mode):
        self.mode = mode
        if self.mode == 'TRAIN':
            self.data_dir_flair = os.path.join(opts.data_root, 'train')
            self.data_mask = os.path.join(opts.mask_root)
            self.seed = None

        if self.mode == 'VALI':
            self.data_dir_flair = os.path.join(opts.data_root, 'vali')
            self.data_mask = os.path.join(opts.mask_root)
            self.seed = 1234

        if self.mode == 'TEST':
            self.data_dir_flair = os.path.join(opts.data_root, 'train')
            self.data_mask = os.path.join(opts.mask_root)
            self.seed = 5678

        self.data_dir_flair = os.path.join(self.data_dir_flair)    # tag kspace directory (T2 / FLAIR)
        self.filenames_tag_kspace = sorted(os.listdir(self.data_dir_flair))

    def __getitem__(self, idx):
        mask = sio.loadmat(self.data_mask)['mask']
        mask = mask[:,:, np.newaxis]
        mask = np.concatenate([mask, mask], axis=2)
        mask256 = torch.from_numpy(mask.astype(np.float32))

# ----------------------------------------------------t1 zf gt  k_zf k_gt
        flair_data= sio.loadmat(os.path.join(self.data_dir_flair, self.filenames_tag_kspace[idx]))
        flair_data_zf = flair_data['ZF']/abs(flair_data['ZF']).max()
        flair_data_real = flair_data_zf.real
        flair_data_real = flair_data_real[:, :, np.newaxis]
        flair_data_imag = flair_data_zf.imag
        flair_data_imag = flair_data_imag[:, :, np.newaxis]
        flair_zf_img = np.concatenate([flair_data_real, flair_data_imag], axis=2)
        flair_zf_img = to_tensor(flair_zf_img).float()#.permute(2, 0, 1) #  t1 gt
        flair_zf_k = fft2(flair_zf_img)#t1 gt k
        flair_zf_i = flair_zf_img.permute(2, 0, 1)#2,192,192

        flair_data_gt = flair_data['GT'] / abs(flair_data['GT']).max()
        flair_data_real = flair_data_gt.real
        flair_data_real = flair_data_real[:, :, np.newaxis]
        flair_data_imag = flair_data_gt.imag
        flair_data_imag = flair_data_imag[:, :, np.newaxis]
        flair_gt_img = np.concatenate([flair_data_real, flair_data_imag], axis=2)
        flair_gt_img = to_tensor(flair_gt_img).float()#.permute(2, 0, 1) #  t1 gt
        flair_gt_k = fft2(flair_gt_img)#t1 gt k
        flair_gt_i = flair_gt_img.permute(2, 0, 1)  # 2,192,192
        return {
                'ref_kspace_mask2d': mask256,
                'tag_kspace_full': flair_gt_k,
                'tag_kspace_sub': flair_zf_k,
                'tag_image_full': flair_gt_i,
                'tag_image_sub': flair_zf_i,
                'tag_kspace_mask2d': mask256}

    def __len__(self):
        return len(self.filenames_tag_kspace)


if __name__ == '__main__':
    a = 1
