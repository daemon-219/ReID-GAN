from __future__ import absolute_import
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from torchvision.models.resnet import Bottleneck
import copy
from .pooling import build_pooling_layer


__all__ = ['ResNet_MP', 'resnet_mp18', 'resnet_mp34', 'resnet_mp50', 'resnet_mp101',
           'resnet_mp152']


class ResNet_MP(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=2048, norm=False, dropout=0, num_proj=256, pooling_type='avg',
                 need_predictor=False):
        super(ResNet_MP, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        self.need_predictor = need_predictor
        # Construct base (pretrained) resnet
        if depth not in ResNet_MP.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet_MP.__factory[depth](pretrained=pretrained)
        # resnet = ResNet.__factory[depth](weights="DEFAULT")

        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3)        
        
        self.res_g = resnet.layer4

        self.res_p = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        
        self.res_p.load_state_dict(resnet.layer4.state_dict())

        self.gpool2d = build_pooling_layer(pooling_type)

        self.num_features = num_features
        self.norm = norm
        self.dropout = dropout
        self.has_embedding = num_features > 0
        self.num_proj = num_proj

        out_planes = resnet.fc.in_features

        self.num_features = out_planes
        
        feat_bn = nn.BatchNorm1d(self.num_features)
        feat_bn.bias.requires_grad_(False)
        init.constant_(feat_bn.weight, 1)
        init.constant_(feat_bn.bias, 0)

        self.feat_bn_g = copy.deepcopy(feat_bn)
        self.feat_bn_p1 = copy.deepcopy(feat_bn)
        self.feat_bn_p2 = copy.deepcopy(feat_bn)

        self.feat_bn_gan = nn.BatchNorm2d(self.num_features)
        self.feat_bn_gan.bias.requires_grad_(False)
        init.constant_(self.feat_bn_gan.weight, 1)
        init.constant_(self.feat_bn_gan.bias, 0)

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

        if self.need_predictor:
            print("build predictor for cl loss")
            self._build_predictor_mlps(self.num_features, 2*self.num_features)

        self.fc_id_g = nn.Linear(self.num_features, self.num_features // 2, bias=False)
        self.fc_id_p1 = nn.Linear(self.num_features, self.num_features // 4, bias=False)
        self.fc_id_p2 = nn.Linear(self.num_features, self.num_features // 4, bias=False)
       
        self.proj_gan = nn.Conv2d(self.num_features, self.num_proj, 1, bias=False)
        init.kaiming_normal_(self.proj_gan.weight, mode='fan_out')      

        if not pretrained:
            self.reset_params()
    
        self._init_fc(self.fc_id_g)
        self._init_fc(self.fc_id_p1)
        self._init_fc(self.fc_id_p2)

    # @torch.cuda.amp.autocast()
    def forward(self, x, clustering=False, fusion='sum'):
        bs = x.size(0)
        x = self.base(x)

        x_g = self.res_g(x)
        # print(x_g.shape)
        # [b, 2048, 8, 4]
        x_p = self.res_p(x)
        # [b, 2048, 16, 8]
        # print(x_p.shape)

        _, _, fh, _ = x_p.shape
        div = fh // 2
        x_p1 = self.gpool2d(x_p[:,:,:div,:]).view(bs, -1)
        x_p2 = self.gpool2d(x_p[:,:,div:,:]).view(bs, -1)
    
        x_g = self.gpool2d(x_g).view(bs, -1)

        x_g = self.feat_bn_g(x_g)
        x_p1 = self.feat_bn_p1(x_p1)
        x_p2 = self.feat_bn_p2(x_p2)
        x_p = self.feat_bn_gan(x_p)

        if fusion == "cat":
            x_gc = torch.cat([self.fc_id_g(x_g), self.fc_id_p1(x_p1), self.fc_id_p2(x_p2)], dim=1)
        elif fusion == "sum":
            x_gc = x_g + x_p1 + x_p2
        else:
            x_gc = x_g
            
        # x_p1 = self.fc_id_p1(x_p1)
        # x_p2 = self.fc_id_p2(x_p2)
        # x_gc = torch.cat([self.fc_id_g(x_g), x_p1, x_p2], dim=1)

        # x_gan = self.proj_gan(x_p)

        if self.norm:
            # [b, 2048]
            # [b, 2048]
            # [b, 256, 16, 8]
            f_g = F.normalize(x_g)
            f_p1 = F.normalize(x_p1)
            f_p2 = F.normalize(x_p2)
            f_gc = F.normalize(x_gc)
            # f_gan = F.normalize(x_gan, dim=1)
            # f_gan = x_gan

        if (self.training is False):
            if clustering:
                return f_gc, f_g
            return f_gc

        if self.dropout > 0:
            f_g = self.drop(f_g)
            f_gc = self.drop(f_gc)
            # f_gan = self.drop(f_gan)

        # return f_g, f_p1, f_p2, f_gc, f_gan
        return f_g, f_p1, f_p2, f_gc

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_predictor_mlps(self, dim, mlp_dim):
        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)

    @staticmethod
    def _init_fc(fc):
        init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        # nn.init.constant_(fc.bias, 0.)
        if fc.bias is not None:
            init.constant_(fc.bias, 0)


def resnet_mp18(**kwargs):
    return ResNet_MP(18, **kwargs)


def resnet_mp34(**kwargs):
    return ResNet_MP(34, **kwargs)


def resnet_mp50(**kwargs):
    return ResNet_MP(50, **kwargs)


def resnet_mp101(**kwargs):
    return ResNet_MP(101, **kwargs)


def resnet_mp152(**kwargs):
    return ResNet_MP(152, **kwargs)
