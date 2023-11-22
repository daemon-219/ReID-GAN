from __future__ import absolute_import
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import copy
import torchvision
from .pooling import build_pooling_layer


__all__ = ['ResNetBIP', 'resnet_bip50']


class ResNetBIP(nn.Module):
    __factory = {
        50: torchvision.models.resnet50,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, pooling_type='avg',
                 need_predictor=False):
        super(ResNetBIP, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        self.need_predictor = need_predictor
        # Construct base (pretrained) resnet
        if depth not in ResNetBIP.__factory:
            raise KeyError("Unsupported depth:", depth)
        if self.pretrained:
            resnet = ResNetBIP.__factory[depth](weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = ResNetBIP.__factory[depth]()
            
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        # self.base = nn.Sequential(
        #     resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        #     resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2)
        
        self.p1 = nn.Sequential(copy.deepcopy(resnet.layer3), copy.deepcopy(resnet.layer4))
        self.p2 = nn.Sequential(copy.deepcopy(resnet.layer3), copy.deepcopy(resnet.layer4))

        self.gap = build_pooling_layer(pooling_type)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn1 = nn.BatchNorm1d(self.num_features)
                self.feat_bn2 = nn.BatchNorm1d(self.num_features)
            self.feat_bn1.bias.requires_grad_(False)
            self.feat_bn2.bias.requires_grad_(False)

            if self.dropout > 0:
                # print('dropout:', self.dropout)
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
            if self.need_predictor:
                print("build predictor for cl loss")
                self._build_predictor_mlps(self.num_features, 2*self.num_features)

        init.constant_(self.feat_bn1.weight, 1)
        init.constant_(self.feat_bn2.weight, 1)
        init.constant_(self.feat_bn1.bias, 0)        
        init.constant_(self.feat_bn2.bias, 0)

        if not pretrained:
            self.reset_params()
    
    def forward(self, x, fuse=True, output_balance=1.):
        bs = x.size(0)
        x = self.base(x)

        # bi path
        x1 = self.p1(x)
        x2 = self.p2(x)
        # [b, 2048, 16, 8]

        x1 = self.gap(x1)
        x1 = x1.view(bs, -1)        
        x2 = self.gap(x2)
        x2 = x2.view(bs, -1)

        if self.cut_at_pooling:
            return x

        bn_x1 = self.feat_bn1(x1)
        bn_x2 = self.feat_bn2(x2)

        if self.norm:
            bn_x1 = F.normalize(bn_x1)
            bn_x2 = F.normalize(bn_x2)

        if fuse:
            # print('diff:', (bn_x1-bn_x2).sum())
            bn_x = output_balance * bn_x1 + (1 - output_balance) * bn_x2
            if self.norm:
                bn_x = F.normalize(bn_x)

            if self.dropout > 0:
                bn_x = self.drop(bn_x)

            # print(bn_x)
            return bn_x

        if self.dropout > 0:
            bn_x1 = self.drop(bn_x1)
            bn_x2 = self.drop(bn_x2)

        return bn_x1, bn_x2

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

def resnet_bip50(**kwargs):
    return ResNetBIP(50, **kwargs)

