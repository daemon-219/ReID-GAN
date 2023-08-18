   

import torch
from clustercontrast.utils.data import transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

def my_resize(X, size=(256, 128)):
    return F.resize(X, size, interpolation=InterpolationMode.BICUBIC)

def my_pad(X, pad=10):
    return F.pad(X, pad)

def my_normalize(X, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return F.normalize(X, mean, std)

def my_transform(X):
    return my_normalize(my_resize(X))

def pair_rand_flip(Xa, Xb, flip_prob=0.5):
    randf = torch.rand(Xa.size(0), 1, 1, 1, device=Xa.device)
    return torch.where(randf < flip_prob, Xa.flip(3), Xa), torch.where(randf < flip_prob, Xb.flip(3), Xb)

# T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
# T.RandomHorizontalFlip(p=0.5),
# T.Pad(10),
# T.RandomCrop((height, width)),
# T.ToTensor(),
# normalizer,
# T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])

# def pair_rand_crop(Xa, Xb, size=(256, 128)):
#     h, w = Xa.size(2), Xa.size(3)
#     translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
#     set_seed_DiffAug(param)
#     translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
#     grid_batch, grid_x, grid_y = torch.meshgrid(
#         torch.arange(x.size(0), dtype=torch.long, device=x.device),
#         torch.arange(x.size(2), dtype=torch.long, device=x.device),
#         torch.arange(x.size(3), dtype=torch.long, device=x.device),
#     )
#     grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
#     grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
#     x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
#     x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
#     return x


# def rand_cutout(x, param):
#     ratio = param.ratio_cutout
#     cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
#     set_seed_DiffAug(param)
#     offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
#     set_seed_DiffAug(param)
#     offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
#     if param.Siamese:  # Siamese augmentation:
#         offset_x[:] = offset_x[0]
#         offset_y[:] = offset_y[0]
#     grid_batch, grid_x, grid_y = torch.meshgrid(
#         torch.arange(x.size(0), dtype=torch.long, device=x.device),
#         torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
#         torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
#     )
#     grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
#     grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
#     mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
#     mask[grid_batch, grid_x, grid_y] = 0
#     x = x * mask.unsqueeze(1)
#     return x