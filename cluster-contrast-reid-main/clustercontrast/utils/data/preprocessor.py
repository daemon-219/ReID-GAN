from __future__ import absolute_import
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F
import numpy as np
import random
import math
import pandas as pd
from PIL import Image
from scipy import ndimage
import torch
from collections import defaultdict

from .pose_utils import load_pose_cords_from_strings, cords_to_map


# class Preprocessor(Dataset):
#     def __init__(self, dataset, root=None, transform=None):
#         super(Preprocessor, self).__init__()
#         self.dataset = dataset
#         self.root = root
#         self.transform = transform

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, indices):
#         return self._get_single_item(indices)

#     def _get_single_item(self, index):
#         fname, pid, camid = self.dataset[index]
#         fpath = fname
#         if self.root is not None:
#             fpath = osp.join(self.root, fname)

#         img = Image.open(fpath).convert('RGB')

#         if self.transform is not None:
#             img = self.transform(img)

#         return img, fname, pid, camid, index
    
class Preprocessor(Dataset):
    def __init__(self, 
                 dataset, 
                 root=None, 
                 pose_file=None, 
                #  height=256, 
                #  width=128, 
                 only_gan=False,
                 with_gan=False,
                 load_size=128,
                #  pose_aug='no', 
                 transform=None, 
                #  gan_transform=None, 
                #  gan_transform_p=None,
                 GAN_transform=None):
        
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

        self.only_gan = only_gan
        self.with_gan = with_gan

        self.with_pose = pose_file is not None

        if self.with_gan or self.only_gan:
            # DPTN
            if isinstance(load_size, int):
                self.load_size = (128, 64)
            else:
                self.load_size = load_size
            if self.with_pose:
                self.annotation_file = pd.read_csv(pose_file, sep=':')
                self.annotation_file = self.annotation_file.set_index('name')
            self.trans = GAN_transform

            # self.pid_name = defaultdict(list)

            # for fname, pid, _ in self.dataset:
            #     if pid < 0:
            #         continue
            #     self.pid_name[pid].append(fname)

            # # fdgan
            # self.height = height
            # self.width = width
            # self.gan_transform = gan_transform
            # self.gan_transform_p = gan_transform_p
            # self.pose_aug = pose_aug
            # self.pose_dict = self.read_pose_csv(pose_file) if self.with_gan else None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        self.flip = torch.rand(1) < 0.5
        # self.flip = False
        if self.only_gan:
            return self._get_single_gan_item(indices)
        elif self.with_gan:
            return self._get_single_item_with_gan(indices)
        else:
            return self._get_single_item(indices)
    
    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.flip:
            img = torch.flip(img, [2])

        return img, fname, pid, camid, index

    def _get_single_item_with_gan(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)

        if self.flip:
            img = torch.flip(img, [2])

        return [img, fname, pid, camid, index], self.get_DPTN_input(fname, pid, flip=self.flip)    
    
    def _get_single_gan_item(self, index):
        fname, pid, camid = self.dataset[index]

        return self.get_DPTN_input(fname, pid, flip=self.flip)
    
    def get_DPTN_input(self, fname, pid, flip=False):
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        Xs = Image.open(fpath).convert('RGB')
        Xs_name = osp.split(fpath)[-1]

        # prepare for source image Xs and target image Xt
        # Xs_name, Xt_name = self.name_pairs[index]

        # strategy for sampling target images: randomly select

        # fpath_t = random.choice(self.pid_name[pid])
        # if self.root is not None:
        #     fpath_t = osp.join(self.root, fpath_t)
        # Xt = Image.open(fpath_t).convert('RGB')
        # Xt_name = osp.split(fpath_t)[-1]

        # Xt = Xs.transpose(Image.FLIP_LEFT_RIGHT)
        # Pt = torch.flip(Ps, [2])

        Xs = F.resize(Xs, self.load_size)
        # Xt = F.resize(Xt, self.load_size)

        Xs = self.trans(Xs)
        if flip:     
            Xs = torch.flip(Xs, [2])

        gt_label = int(Xs_name.split('_', 1)[0])

        if self.with_pose:
            Ps = self.obtain_bone(Xs_name)
            
            if flip: 
                Ps = torch.flip(Ps, [2])
            
            # Pt = self.obtain_bone(Xt_name)
            # Xt = self.trans(Xt)

            # return {'Xs': Xs, 'Ps': Ps, 'Xt': Xt, 'Pt': Pt,
            #         'Xs_path': Xs_name, 'Xt_path': Xt_name} 
            
            return {'Xs': Xs, 'Ps': Ps, 'Xs_path': Xs_name, 'gt_label': gt_label}
        
        return {'Xs': Xs, 'Xs_path': Xs_name, 'gt_label': gt_label}
    
    def obtain_bone(self, name):
        string = self.annotation_file.loc[name]
        array = load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
        pose = cords_to_map(array, self.load_size, (128, 64))
        pose = np.transpose(pose,(2, 0, 1))
        pose = torch.Tensor(pose)
        return pose
