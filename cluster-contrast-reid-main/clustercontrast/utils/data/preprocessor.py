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
                 with_pose=False,
                 load_size=128,
                #  pose_aug='no', 
                 transform=None, 
                #  gan_transform=None, 
                #  gan_transform_p=None,
                 DPTN_transform=None):
        
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

        self.with_pose = with_pose

        if self.with_pose:
            # DPTN
            if isinstance(load_size, int):
                self.load_size = (128, 64)
            else:
                self.load_size = load_size
            self.annotation_file = pd.read_csv(pose_file, sep=':')
            self.annotation_file = self.annotation_file.set_index('name')
            self.trans = DPTN_transform

            self.pid_name = defaultdict(list)

            for fname, pid, _ in self.dataset:
                if pid < 0:
                    continue
                self.pid_name[pid].append(fname)

            # # fdgan
            # self.height = height
            # self.width = width
            # self.gan_transform = gan_transform
            # self.gan_transform_p = gan_transform_p
            # self.pose_aug = pose_aug
            # self.pose_dict = self.read_pose_csv(pose_file) if self.with_pose else None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if self.with_pose:
            return self._get_single_item_with_pose(indices)
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

        return img, fname, pid, camid, index

    def _get_single_item_with_pose(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        
        if self.transform is not None:
            reid_img = self.transform(img)

        return [reid_img, fname, pid, camid, index], self.get_DPTN_input(fname, pid)

    def get_DPTN_input(self, fname, pid):
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        Xs = Image.open(fpath).convert('RGB')
        Xs_name = osp.split(fpath)[-1]

        # prepare for source image Xs and target image Xt
        # Xs_name, Xt_name = self.name_pairs[index]

        # strategy for sampling target images: randomly select

        fpath_t = random.choice(self.pid_name[pid])
        if self.root is not None:
            fpath_t = osp.join(self.root, fpath_t)
        Xt = Image.open(fpath_t).convert('RGB')
        Xt_name = osp.split(fpath_t)[-1]

        # Xt = Xs.transpose(Image.FLIP_LEFT_RIGHT)
        # Pt = torch.flip(Ps, [2])

        Xs = F.resize(Xs, self.load_size)
        Xt = F.resize(Xt, self.load_size)

        Ps = self.obtain_bone(Xs_name)
        Xs = self.trans(Xs)
        
        Pt = self.obtain_bone(Xt_name)
        Xt = self.trans(Xt)

        return {'Xs': Xs, 'Ps': Ps, 'Xt': Xt, 'Pt': Pt,
                'Xs_path': Xs_name, 'Xt_path': Xt_name}
    
    def obtain_bone(self, name):
        string = self.annotation_file.loc[name]
        array = load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
        pose = cords_to_map(array, self.load_size, (128, 64))
        pose = np.transpose(pose,(2, 0, 1))
        pose = torch.Tensor(pose)
        return pose

    # def read_pose_csv(self, pose_file):
    #     pose_dict = {}

    #     with open(pose_file, 'r') as csvfile:
    #         items = np.loadtxt(csvfile, str, delimiter = ":", skiprows = 1, usecols = (0, 1, 2))

    #         for item in items:
    #             img_name = item[0].replace('.jpg', '') 
    #             pose_x = np.array(item[2].strip('[ ]').split(',')).astype(np.float64).reshape(-1, 1)
    #             pose_y = np.array(item[1].strip('[ ]').split(',')).astype(np.float64).reshape(-1, 1)
    #             # key_point = np.concatenate((pose_x, pose_y), axis=1) 
    #             key_point = np.concatenate((pose_y, pose_x), axis=1) 

    #             pose_dict[img_name] = key_point

    #     return pose_dict

    # def get_FDGAN_input(self, fname, pid):
    #     fpath = fname
    #     if self.root is not None:
    #         fpath = osp.join(self.root, fname)
    #     img = Image.open(fpath).convert('RGB')

    #     if self.gan_transform is not None:
    #         gan_img = self.gan_transform(img)

    #     # pid_query = list(self.pid_imgs[pid])
    #     # if fname in pid_query and len(pid_query)>1:
    #     #     pid_query.remove(fname)
    #     # pname = osp.splitext(random.choice(pid_query))[0]

    #     # ppath = pname+'.txt'
    #     # if self.pose_root is not None:
    #     #     ppath = osp.join(self.pose_root, ppath)
    #     # gtpath = pname+'.jpg'
    #     # if self.root is not None:
    #     #     gtpath = osp.join(self.root, gtpath)

    #     # gt_img = Image.open(gtpath).convert('RGB')
    #     gt_img = img

    #     img_name = osp.splitext(osp.split(fpath)[-1])[0]
    #     pose_list = self.pose_dict[img_name]

    #     landmark = self._load_landmark(pose_list, self.height/gt_img.size[1], self.width/gt_img.size[0])
    #     maps = self._generate_pose_map(landmark)

    #     flip_flag = random.choice([True, False])
    #     if flip_flag:
    #         gt_img = gt_img.transpose(Image.FLIP_LEFT_RIGHT)
    #         maps = np.flip(maps,2)

    #     maps = torch.from_numpy(maps.copy()).float()
    #     if self.gan_transform_p is not None:
    #         gt_img = self.gan_transform_p(gt_img)

    #     return {'origin': gan_img, 'target': gt_img, 'posemap': maps, 'pid': torch.LongTensor([pid])}

    # def _load_landmark(self, pose_list, scale_h, scale_w):
    #     landmark = torch.tensor(pose_list) * torch.Tensor([scale_h, scale_w])
    #     return landmark

    # def _generate_pose_map(self, landmark, gauss_sigma=5):
    #     maps = []
    #     randnum = landmark.size(0)+1
    #     if self.pose_aug=='erase':
    #         randnum = random.randrange(landmark.size(0))
    #     elif self.pose_aug=='gauss':
    #         gauss_sigma = random.randint(gauss_sigma-1,gauss_sigma+1)
    #     elif self.pose_aug!='no':
    #         assert ('Unknown landmark augmentation method, choose from [no|erase|gauss]')
    #     for i in range(landmark.size(0)):
    #         map = np.zeros([self.height,self.width])
    #         if landmark[i,0]!=-1 and landmark[i,1]!=-1 and i!=randnum:
    #             map[landmark[i,0],landmark[i,1]]=1
    #             map = ndimage.filters.gaussian_filter(map,sigma = gauss_sigma)
    #             map = map/map.max()
    #         maps.append(map)
    #     maps = np.stack(maps, axis=0)
    #     return maps