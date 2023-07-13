from __future__ import absolute_import
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image
from scipy import ndimage
import torch


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
                 height=256, 
                 width=128, 
                 with_pose=False,
                 pose_aug='no', 
                 transform=None, 
                 gan_transform=None, 
                 gan_transform_p=None):
        
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

        # fdgan
        self.height = height
        self.width = width
        self.gan_transform = gan_transform
        self.gan_transform_p = gan_transform_p
        self.pose_aug = pose_aug
        self.with_pose = with_pose
        self.pose_dict = self.read_pose_csv(pose_file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if self.with_pose:
            return self._get_single_item_with_pose(indices)
        else:
            return self._get_single_item(indices)
    
    def read_pose_csv(self, pose_file):
        pose_dict = {}

        with open(pose_file, 'r') as csvfile:
            items = np.loadtxt(csvfile, str, delimiter = ":", skiprows = 1, usecols = (0, 1, 2))

            for item in items:
                img_name = item[0].replace('.jpg', '') 
                pose_x = np.array(item[2].strip('[ ]').split(',')).astype(np.float64).reshape(-1, 1)
                pose_y = np.array(item[1].strip('[ ]').split(',')).astype(np.float64).reshape(-1, 1)
                # key_point = np.concatenate((pose_x, pose_y), axis=1) 
                key_point = np.concatenate((pose_y, pose_x), axis=1) 

                pose_dict[img_name] = key_point

        return pose_dict

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

        if self.gan_transform is not None:
            gan_img = self.gan_transform(img)

        # pid_query = list(self.pid_imgs[pid])
        # if fname in pid_query and len(pid_query)>1:
        #     pid_query.remove(fname)
        # pname = osp.splitext(random.choice(pid_query))[0]

        # ppath = pname+'.txt'
        # if self.pose_root is not None:
        #     ppath = osp.join(self.pose_root, ppath)
        # gtpath = pname+'.jpg'
        # if self.root is not None:
        #     gtpath = osp.join(self.root, gtpath)

        # gt_img = Image.open(gtpath).convert('RGB')
        gt_img = img

        img_name = osp.splitext(osp.split(fpath)[-1])[0]
        pose_list = self.pose_dict[img_name]

        landmark = self._load_landmark(pose_list, self.height/gt_img.size[1], self.width/gt_img.size[0])
        maps = self._generate_pose_map(landmark)

        flip_flag = random.choice([True, False])
        if flip_flag:
            gt_img = gt_img.transpose(Image.FLIP_LEFT_RIGHT)
            maps = np.flip(maps,2)

        maps = torch.from_numpy(maps.copy()).float()
        if self.gan_transform_p is not None:
            gt_img = self.gan_transform_p(gt_img)

        return reid_img, fname, pid, camid, index, {'origin': gan_img, 'target': gt_img, 'posemap': maps, 'pid': torch.LongTensor([pid])}

    def _load_landmark(self, pose_list, scale_h, scale_w):
        landmark = torch.tensor(pose_list) * torch.Tensor([scale_h, scale_w])
        return landmark

    def _generate_pose_map(self, landmark, gauss_sigma=5):
        maps = []
        randnum = landmark.size(0)+1
        if self.pose_aug=='erase':
            randnum = random.randrange(landmark.size(0))
        elif self.pose_aug=='gauss':
            gauss_sigma = random.randint(gauss_sigma-1,gauss_sigma+1)
        elif self.pose_aug!='no':
            assert ('Unknown landmark augmentation method, choose from [no|erase|gauss]')
        for i in range(landmark.size(0)):
            map = np.zeros([self.height,self.width])
            if landmark[i,0]!=-1 and landmark[i,1]!=-1 and i!=randnum:
                map[landmark[i,0],landmark[i,1]]=1
                map = ndimage.filters.gaussian_filter(map,sigma = gauss_sigma)
                map = map/map.max()
            maps.append(map)
        maps = np.stack(maps, axis=0)
        return maps
