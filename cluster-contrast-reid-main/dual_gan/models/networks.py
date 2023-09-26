import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from .base_function import *
from .PTM import PTM
from clustercontrast.utils.data.diff_augs import my_resize, my_transform, my_normalize


###############################################################################
# Functions
###############################################################################
def define_G(opt, image_nc, pose_nc, ngf=64, img_f=1024, encoder_layer=3, norm='batch',
                 activation='ReLU', use_spect=True, use_coord=False, output_nc=3, num_blocks=3, affine=True, nhead=2, num_CABs=2, num_TTBs=2):
    print(opt.model_gen)
    if opt.model_gen == 'DPTN':
        netG = DPTNGenerator(image_nc, pose_nc, ngf, img_f, encoder_layer, norm, activation, use_spect, use_coord, output_nc, num_blocks, affine, nhead, num_CABs, num_TTBs)
    elif opt.model_gen == 'AE':
        netG = AEGenerator(image_nc, ngf, img_f, encoder_layer, norm, activation, use_spect, use_coord, output_nc, num_blocks)    
    elif opt.model_gen == 'DEC':
        netG = DECGenerator(ngf, img_f, encoder_layer, norm, activation, use_spect, use_coord, output_nc)    
    elif opt.model_gen == 'FD':
        netG = FDGenerator(img_f, ngf, output_nc=3, noise_nc=512, fuse_mode='add')
    else:
        raise('generator not implemented!')
    return init_net(netG, opt.init_type)


def define_D(opt, input_nc=3, ndf=64, img_f=1024, layers=3, norm='none', activation='LeakyReLU', use_spect=True,):
    netD = ResDiscriminator(input_nc, ndf, img_f, layers, norm, activation, use_spect)
    return init_net(netD, opt.init_type)


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Generator
##############################################################################
class SourceEncoder(nn.Module):
    """
    Source Image Encoder (En_s)
    :param image_nc: number of channels in input image
    :param ngf: base filter channel
    :param img_f: the largest feature channels
    :param encoder_layer: encoder layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectual normalization
    :param use_coord: use coordConv operation
    """
    def __init__(self, image_nc, ngf=64, img_f=1024, encoder_layer=3, norm='batch',
                 activation='ReLU', use_spect=True, use_coord=False):
        super(SourceEncoder, self).__init__()

        self.encoder_layer = encoder_layer

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        input_nc = image_nc

        self.block0 = EncoderBlockOptimized(input_nc, ngf, norm_layer,
                                   nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(encoder_layer - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = EncoderBlock(ngf * mult_prev, ngf * mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

    def forward(self, source):
        inputs = source
        out = self.block0(inputs)
        for i in range(self.encoder_layer - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        return out
    
# class Adaptor(nn.Module):
#     """
#     Adaptation from synthetic images to real images
#     :param image_nc: number of channels in input image
#     :param ngf: base filter channel
#     :param img_f: the largest feature channels
#     :param encoder_layer: encoder layers
#     :param norm: normalization function 'instance, batch, group'
#     :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
#     :param use_spect: use spectual normalization
#     :param use_coord: use coordConv operation
#     """
#     def __init__(self, image_nc, ngf=64, img_f=1024, encoder_layer=3, norm='batch',
#                  activation='ReLU', use_spect=True, use_coord=False):
#         super(Adaptor, self).__init__()

#         self.encoder_layer = encoder_layer

#         norm_layer = get_norm_layer(norm_type=norm)
#         nonlinearity = get_nonlinearity_layer(activation_type=activation)
#         input_nc = image_nc

#         # ResBlocks
#         self.num_blocks = num_blocks
#         for i in range(num_blocks):
#             block = ResBlock(ngf * mult, ngf * mult, norm_layer=norm_layer,
#                              nonlinearity=nonlinearity, use_spect=use_spect, use_coord=use_coord)
#             setattr(self, 'mblock' + str(i), block)

#         # Decoder
#         for i in range(self.layers):
#             mult_prev = mult
#             mult = min(2 ** (self.layers - i - 2), img_f // ngf) if i != self.layers - 1 else 1
#             up = ResBlockDecoder(ngf * mult_prev, ngf * mult, ngf * mult, norm_layer,
#                                  nonlinearity, use_spect, use_coord)
#             setattr(self, 'decoder' + str(i), up)
#         self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spect, use_coord)

#     def forward(self, source):
#         inputs = source
#         out = self.block0(inputs)
#         for i in range(self.encoder_layer - 1):
#             model = getattr(self, 'encoder' + str(i))
#             out = model(out)
#         return out

class Resize_ReID(nn.Module):
    """
    resize from sythesized image (128, 64) to reid inputs (256, 128)
    """
    def __init__(self, image_nc, ngf=64, norm='batch',
                 activation='ReLU', use_spect=True, use_coord=False):
        super(Resize_ReID, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        # ResBlocks
        self.resblock1 = ResBlock(image_nc, ngf, norm_layer=norm_layer, nonlinearity=nonlinearity, 
                                     sample_type='none', use_spect=use_spect, use_coord=use_coord)
        self.resblock2 = ResBlock(ngf, ngf, norm_layer=norm_layer, nonlinearity=nonlinearity, 
                                     sample_type='none', use_spect=use_spect, use_coord=use_coord)
        self.resblock3 = ResBlock(ngf, image_nc, norm_layer=norm_layer, nonlinearity=nonlinearity, 
                                     sample_type='none', use_spect=use_spect, use_coord=use_coord)

    def forward(self, inputs):
        x = my_resize(inputs)
        out = self.resblock3(self.resblock2(self.resblock1(x)))
        return x + out


class DPTNGenerator(nn.Module):
    """
    Dual-task Pose Transformer Network (DPTN)
    :param image_nc: number of channels in input image
    :param pose_nc: number of channels in input pose
    :param ngf: base filter channel
    :param img_f: the largest feature channels
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectual normalization
    :param use_coord: use coordConv operation
    :param output_nc: number of channels in output image
    :param num_blocks: number of ResBlocks
    :param affine: affine in Pose Transformer Module
    :param nhead: number of heads in attention module
    :param num_CABs: number of CABs
    :param num_TTBs: number of TTBs
    """
    def __init__(self, image_nc, pose_nc, ngf=64, img_f=256, layers=3, norm='batch',
                 activation='ReLU', use_spect=True, use_coord=False, output_nc=3, num_blocks=3, affine=True, nhead=2, num_CABs=2, num_TTBs=2):
        super(DPTNGenerator, self).__init__()

        self.layers = layers
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        input_nc = 2 * pose_nc + image_nc

        # Encoder En_c
        self.block0 = EncoderBlockOptimized(input_nc, ngf, norm_layer,
                                   nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(self.layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = EncoderBlock(ngf * mult_prev, ngf * mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        # ResBlocks
        self.num_blocks = num_blocks
        for i in range(num_blocks):
            block = ResBlock(ngf * mult, ngf * mult, norm_layer=norm_layer,
                             nonlinearity=nonlinearity, use_spect=use_spect, use_coord=use_coord)
            setattr(self, 'mblock' + str(i), block)

        # Pose Transformer Module (PTM)
        self.PTM = PTM(d_model=ngf * mult, nhead=nhead, num_CABs=num_CABs,
                 num_TTBs=num_TTBs, dim_feedforward=ngf * mult,
                 activation="LeakyReLU", affine=affine, norm=norm)

        # Encoder En_s
        self.source_encoder = SourceEncoder(image_nc, ngf, img_f, layers, norm, activation, use_spect, use_coord)

        # Decoder
        for i in range(self.layers):
            mult_prev = mult
            mult = min(2 ** (self.layers - i - 2), img_f // ngf) if i != self.layers - 1 else 1
            up = ResBlockDecoder(ngf * mult_prev, ngf * mult, ngf * mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'decoder' + str(i), up)
        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spect, use_coord)

    def forward(self, source, source_B, target_B, is_train=True):
        # Self-reconstruction Branch
        # Source-to-source Inputs
        input_s_s = torch.cat((source, source_B, source_B), 1)
        # Source-to-source Encoder
        F_s_s = self.block0(input_s_s)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            F_s_s = model(F_s_s)
        # Source-to-source Resblocks
        for i in range(self.num_blocks):
            model = getattr(self, 'mblock' + str(i))
            F_s_s = model(F_s_s)

        # Transformation Branch
        # Source-to-target Inputs
        input_s_t = torch.cat((source, source_B, target_B), 1)
        # Source-to-target Encoder
        F_s_t = self.block0(input_s_t)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            F_s_t = model(F_s_t)
        # Source-to-target Resblocks
        for i in range(self.num_blocks):
            model = getattr(self, 'mblock' + str(i))
            F_s_t = model(F_s_t)

        # Source Image Encoding
        F_s = self.source_encoder(source)

        # Pose Transformer Module for Dual-task Correlation
        F_s_t = self.PTM(F_s_s, F_s_t, F_s)

        # Source-to-source Decoder (only for training)
        out_image_s = None
        if is_train:
            for i in range(self.layers):
                model = getattr(self, 'decoder' + str(i))
                F_s_s = model(F_s_s)
            out_image_s = self.outconv(F_s_s)

        # Source-to-target Decoder
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            F_s_t = model(F_s_t)
        out_image_t = self.outconv(F_s_t)

        return out_image_t, out_image_s
    

class AEGenerator(nn.Module):
    """
    Autoencoder 
    :param image_nc: number of channels in input image
    :param pose_nc: number of channels in input pose
    :param ngf: base filter channel
    :param img_f: the largest feature channels
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectual normalization
    :param use_coord: use coordConv operation
    :param output_nc: number of channels in output image
    :param num_blocks: number of ResBlocks
    :param affine: affine in Pose Transformer Module
    :param nhead: number of heads in attention module
    :param num_CABs: number of CABs
    :param num_TTBs: number of TTBs
    """
    def __init__(self, image_nc, ngf=64, img_f=256, layers=3, norm='batch',
                 activation='ReLU', use_spect=True, use_coord=False, output_nc=3, num_blocks=3):
        super(AEGenerator, self).__init__()

        self.layers = layers
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        input_nc = image_nc

        # Encoder En_c
        self.block0 = EncoderBlockOptimized(input_nc, ngf, norm_layer,
                                   nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(self.layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = EncoderBlock(ngf * mult_prev, ngf * mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        # ResBlocks
        self.num_blocks = num_blocks
        for i in range(num_blocks):
            block = ResBlock(ngf * mult, ngf * mult, norm_layer=norm_layer,
                             nonlinearity=nonlinearity, use_spect=use_spect, use_coord=use_coord)
            setattr(self, 'mblock' + str(i), block)

        # Decoder
        for i in range(self.layers):
            mult_prev = mult
            mult = min(2 ** (self.layers - i - 2), img_f // ngf) if i != self.layers - 1 else 1
            up = ResBlockDecoder(ngf * mult_prev, ngf * mult, ngf * mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'decoder' + str(i), up)
        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spect, use_coord)

    def forward(self, inputs):
        F_s = self.forward_enc(inputs)
        out_image =self.forward_dec(F_s)
        return out_image    
    
    def forward_enc(self, source):
        # Enc
        F_s = self.block0(source)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            F_s = model(F_s)
        # Resblocks
        for i in range(self.num_blocks):
            model = getattr(self, 'mblock' + str(i))
            F_s = model(F_s)

        return F_s

    def forward_dec(self, feature):
        # Decoder
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            feature = model(feature)
        out_image = self.outconv(feature)

        return out_image

class DECGenerator(nn.Module):

    def __init__(self, ngf=64, img_f=2048, layers=3, norm='batch',
                 activation='ReLU', use_spect=True, use_coord=False, output_nc=3, num_blocks=3):
        super(DECGenerator, self).__init__()

        self.layers = layers
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        mult = 4

        # # (b, 2048)->(b, 512, 4, 2)
        # # don't use instance norm here
        # self.feat_up = ResUP12Block(input_nc=img_f, output_nc=img_f // 4, hidden_nc=img_f // 2,
        #                          nonlinearity=nonlinearity, use_spect=use_spect, use_coord=use_coord)

        # img_f = img_f // 4

        # ResBlocks
        self.resblock = ResBlock(img_f, ngf * mult, norm_layer=norm_layer,
                         nonlinearity=nonlinearity, use_spect=use_spect, use_coord=use_coord)
   
        # Decoder
        # (b, 512, 4, 2)->(b, 64, 128, 64)
        for i in range(self.layers):
            mult_prev = mult
            mult = min(2 ** (self.layers - i - 2), img_f // ngf) if i != self.layers - 1 else 1
            up = ResBlockDecoder(ngf * mult_prev, ngf * mult, ngf * mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'decoder' + str(i), up)
        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spect, use_coord)

    def forward(self, inputs):
        # feature = self.feat_up(inputs)
        feature = self.resblock(inputs)
        # Decoder
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            feature = model(feature)
        out_image = self.outconv(feature)

        return out_image
    

class FDGenerator(nn.Module):
    def __init__(self, reid_feature_nc, ngf=64, noise_nc=3, pose_nc=18, output_nc=3, 
                        dropout=0.0, norm_layer=nn.BatchNorm2d, fuse_mode='none'):
        super(FDGenerator, self).__init__()
        self.fuse_mode = fuse_mode
        self.norm_layer = norm_layer
        self.dropout = dropout

        if type(norm_layer) == functools.partial:
            self.use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            self.use_bias = norm_layer == nn.InstanceNorm2d

        input_channel = [8, 8, 4, 2, 1]

        ##################### Decoder #########################
        if fuse_mode=='cat':
            de_avg = [nn.ReLU(True),
                    nn.ConvTranspose2d(reid_feature_nc + noise_nc, ngf * 8,
                        kernel_size=(8,4), bias=self.use_bias),
                    norm_layer(ngf * 8),
                    nn.Dropout(dropout)]
        elif fuse_mode=='add':
            nc = max(reid_feature_nc, noise_nc)
            self.W_reid = nn.Linear(reid_feature_nc, nc, bias=False)
            self.W_noise = nn.Linear(noise_nc, nc, bias=False)
            de_avg = [nn.ReLU(True),
                    nn.ConvTranspose2d(nc, ngf * 8,
                        kernel_size=(8, 4), bias=self.use_bias),
                    norm_layer(ngf * 8),
                    nn.Dropout(dropout)]
        elif fuse_mode=='none':
            nc = reid_feature_nc
            self.W_reid = nn.Linear(reid_feature_nc, nc, bias=False)
            de_avg = [nn.ReLU(True),
                    nn.ConvTranspose2d(nc, ngf * 8,
                        kernel_size=(8, 4), bias=self.use_bias),
                    norm_layer(ngf * 8),
                    nn.Dropout(dropout)]
        else:
            raise ('Wrong fuse mode, please select from [cat|add]')
        self.de_avg = nn.Sequential(*de_avg)
        # N*512*8*4

        self.de_conv5 = self._make_layer_decode(ngf * input_channel[0], ngf * 8)
        # N*512*16*8
        self.de_conv4 = self._make_layer_decode(ngf * input_channel[1], ngf * 4)
        # N*256*32*16
        self.de_conv3 = self._make_layer_decode(ngf * input_channel[2], ngf * 2)
        # N*128*64*32
        self.de_conv2 = self._make_layer_decode(ngf * input_channel[3], ngf)
        # N*64*128*64
        de_conv1 = [nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * input_channel[4], output_nc,
                        kernel_size=4, stride=2,
                        padding=1, bias=self.use_bias),
                    nn.Tanh()]
        self.de_conv1 = nn.Sequential(*de_conv1)
        # N*3*256*128

    def _make_layer_decode(self, in_nc, out_nc):
        block = [nn.ReLU(True),
                nn.ConvTranspose2d(in_nc, out_nc,
                    kernel_size=4, stride=2,
                    padding=1, bias=self.use_bias),
                self.norm_layer(out_nc),
                nn.Dropout(self.dropout)]
        return nn.Sequential(*block)

    def decode(self, model, fake_feature):
        return model(fake_feature)

    def forward(self, reid_feature, noise=None):
        batch_size = reid_feature.shape[0]

        if self.fuse_mode=='cat':
            feature = torch.cat((reid_feature, noise), dim=1)
        elif self.fuse_mode=='add':
            feature = self.W_reid(reid_feature.view(batch_size, -1)) + \
                      self.W_noise(noise.view(batch_size,-1))
            feature = feature.view(batch_size,-1,1,1)
        elif self.fuse_mode=='none':
            feature = self.W_reid(reid_feature.view(batch_size, -1))
            feature = feature.view(batch_size,-1,1,1)

        fake_feature = self.de_avg(feature)

        fake_feature_5 = self.decode(self.de_conv5, fake_feature)
        fake_feature_4 = self.decode(self.de_conv4, fake_feature_5)
        fake_feature_3 = self.decode(self.de_conv3, fake_feature_4)
        fake_feature_2 = self.decode(self.de_conv2, fake_feature_3)
        fake_feature_1 = self.decode(self.de_conv1, fake_feature_2)

        fake_imgs = fake_feature_1
        return fake_imgs

##############################################################################
# Discriminator
##############################################################################
class ResDiscriminator(nn.Module):
    """
    ResNet Discriminator Network
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param layers: down and up sample layers
    :param img_f: the largest feature channels
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectual normalization
    :param use_coord: use coordConv operation
    """
    def __init__(self, input_nc=3, ndf=64, img_f=1024, layers=3, norm='none', activation='LeakyReLU', use_spect=True,
                 use_coord=False):
        super(ResDiscriminator, self).__init__()

        self.layers = layers

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.nonlinearity = nonlinearity

        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ndf, ndf, norm_layer, nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ndf)
            block = ResBlockEncoder(ndf*mult_prev, ndf*mult, ndf*mult_prev, norm_layer, nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)
        self.conv = SpectralNorm(nn.Conv2d(ndf*mult, 1, 1))

    def forward(self, x):
        out = self.block0(x)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = self.conv(self.nonlinearity(out))
        return out