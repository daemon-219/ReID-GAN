import numpy as np
import torch
import os
import itertools
# from torch.autograd import Variable
# from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import external_function
from . import base_function
from torch import nn
import torch.nn.functional as F


class AEModel(BaseModel):
    def name(self):
        return 'AEModel'

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--init_type', type=str, default='orthogonal', help='initial type')
        parser.add_argument('--use_spect_g', action='store_false', help='use spectual normalization in generator')
        parser.add_argument('--use_spect_d', action='store_false', help='use spectual normalization in generator')
        parser.add_argument('--use_coord', action='store_true', help='use coordconv')
        parser.add_argument('--lambda_style', type=float, default=500, help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=0.5, help='weight for the VGG19 content loss')
        parser.add_argument('--layers_g', type=int, default=3, help='number of layers in G')
        parser.add_argument('--save_input', action='store_true', help="whether save the input images when testing")
        parser.add_argument('--num_blocks', type=int, default=3, help="number of resblocks")

        # if is_train:
        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')
        parser.add_argument('--lambda_rec', type=float, default=2.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_g', type=float, default=5.0, help='weight for generation loss')
        parser.add_argument('--lambda_fus', type=float, default=0.8, help='fusion ratio between samples')
        parser.add_argument('--dis_layers', type=int, default=3, help='number of layers in D')
        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=True)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.old_size = opt.old_size
        self.loss_names = ['app_gen', 'content_gen', 'style_gen', 'ad_gen', 'dis_img_gen', 'G', 'D']
        # self.loss_names = ['app_gen', 'ad_gen', 'dis_img_gen', 'G', 'D']
        self.model_names = ['G']
        # self.visual_names = ['source_image', 'source_pose', 'target_image', 'target_pose', 'fake_image', 'fake_image_n']
        self.visual_names = ['source_image', 'source_pose', 'target_image', 'target_pose', 'fake_image']
        self.model_gen = opt.model_gen
        num_feats = 2048 if (opt.model_gen == 'DEC' or opt.model_gen == 'FD') else 256
        G_layer = 4 if opt.model_gen == 'DEC' else 4

        # for feature fusion output
        self.gap = nn.AdaptiveAvgPool2d(1).to(opt.device)
        self.feat_bn = nn.BatchNorm1d(num_feats).to(opt.device)

        self.net_G = networks.define_G(opt, image_nc=opt.image_nc, pose_nc=opt.pose_nc, ngf=64, img_f=num_feats,
                                       encoder_layer=G_layer, norm=opt.norm, activation='LeakyReLU',
                                       use_spect=opt.use_spect_g, use_coord=opt.use_coord, output_nc=3, num_blocks=3)
        
        self.use_adp = opt.use_adp
        if self.use_adp:
            self.model_names = ['G', 'A']
            self.net_A = networks.Resize_ReID(image_nc=opt.image_nc)

        # Discriminator network
        if self.gan_train:
            self.model_names = ['G', 'D']
            self.net_D = networks.define_D(opt, ndf=32, img_f=128, layers=opt.dis_layers, use_spect=opt.use_spect_d)

        if self.opt.verbose:
                print('---------- Networks initialized -------------')
        # set loss functions and optimizers
        if self.gan_train:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            #self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.gan_lr

            self.GANloss = external_function.GANLoss(opt.gan_mode).to(opt.device)
            self.L1loss = torch.nn.L1Loss()
            # self.Vggloss = external_function.VGGLoss().to(opt.device)

            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                filter(lambda p: p.requires_grad, self.net_G.parameters())),
                lr=opt.gan_lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizer_D = torch.optim.Adam(itertools.chain(
                filter(lambda p: p.requires_grad, self.net_D.parameters())),
                lr=opt.gan_lr * opt.ratio_g2d, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)

            self.schedulers = [base_function.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        elif self.use_adp:
            print("use adaptor")
            # use adaptor, only train adaptor
            self.net_G.eval()
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            #self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.gan_lr

            # define the optimizer
            self.optimizer_A = torch.optim.Adam(itertools.chain(
                filter(lambda p: p.requires_grad, self.net_A.parameters())),
                lr=opt.gan_lr, betas=(opt.beta1, 0.999))
            
            self.schedulers = base_function.get_scheduler(self.optimizer_A, opt) 
        else:
            self.net_G.eval()

        # if not self.gan_train or opt.continue_train:
        #     print('model resumed from latest')
        #     self.load_networks(opt.which_epoch)
        
        if self.load_pretrain != "" or opt.continue_train:
            print('model loaded from pretrained')
            self.load_networks(opt.which_epoch)

    def set_input(self, inputs, b_id=None):
        self.input = inputs
        if b_id is not None:
            # get input from b_id for each group
            source_image = torch.index_select(inputs, 0, b_id)
            # target_image = torch.index_select(input['Xt'], 0, b_id)
        else:
            source_image = inputs
            # target_image = input['Xt']
        self.source_image = source_image.cuda()
        # self.target_image = target_image.cuda()

    def forward(self):
        # Encode Inputs
        self.fake_image = self.net_G(self.source_image)
    
    # @torch.cuda.amp.autocast()
    def synthesize(self, features=None, is_train=False):
        # features = self.feature_fusion(features, torch.flip(features, dims=[0]))
        if features is not None:
            self.fake_image = self.net_G(features)
        else:
            # self.fake_image = self.net_G(self.source_image)
            F_s = self.net_G.module.forward_enc(self.source_image)
            # print(F_s.shape)
            # self.fake_image =self.net_G.module.forward_dec(F_s) 

        return F_s
        # return self.fake_image
            
        # self.fake_image = self.net_A(self.net_G(self.source_image))
        # return self.fake_image, None
        
        # F_s = self.net_G.module.forward_enc(self.source_image)

        # # print(F_s.shape)
        # # (b, 256, 16, 8)

        # self.fake_image = self.net_G.module.forward_dec(F_s)
        
        # F_n = self.feature_fusion(F_s, torch.flip(F_s, dims=[0]))

        # self.fake_image_n = self.net_G.module.forward_dec(F_n)

        # if self.use_adp:
        #     self.fake_image_n = self.net_A(self.fake_image_n)

        # if is_train:
        #     self.fake_image_s = self.net_G.module.forward_dec(F_s)
        #     if self.use_adp:
        #         self.fake_image_s = self.net_A(self.fake_image_s)
        #     return self.fake_image_s, self.fake_image_n  

        # return self.fake_image_n   

    # @torch.cuda.amp.autocast()
    def synthesize_fc(self, group_size=16):
        # self.fake_image = self.net_G(self.source_image)
        F_s = self.net_G.module.forward_enc(self.source_image)
        F_c = torch.mean(torch.stack(torch.split(F_s, group_size, dim=0), dim=0), dim=1)
        self.fake_image = self.net_G.module.forward_dec(F_c)

        return self.fake_image
 

    def feature_fusion(self, F_s, F_t, div=2):
        # feature fusion strategy 
        anchor = F_s.detach().clone()
        target = F_t.detach().clone()
        anchor_feature = F.normalize(torch.flatten(F.adaptive_avg_pool2d(anchor, (1,1)), start_dim=1), dim=-1)
        FH = anchor.shape[2]
        part_h =  FH // div
        part_features = []
        # cut, flatten and norm each part feature
        for i in range(div):
            h1 = i * part_h
            h2 = min((i+1) * part_h, FH)
            pf = torch.flatten(F.adaptive_avg_pool2d(target[:,:,h1:h2,:], (1,1)), start_dim=1)
            part_features.append(F.normalize(pf, dim=1))

        target_parts = torch.stack(part_features, dim=1)
        # calculate similarity between anchor and each target parts
        sim_id = torch.argmax(torch.einsum('n c, n d c -> n d', [anchor_feature, target_parts]), dim=-1)
        F_n = torch.zeros_like(F_s)
        for i in range(div):
            h1 = i * part_h
            h2 = min((i+1) * part_h, FH)
            ratio_mask = torch.where(sim_id == i, 1-self.opt.lambda_fus, self.opt.lambda_fus).reshape(-1, 1, 1, 1)
            F_n[:,:,h1:h2,:] = ratio_mask * F_s[:,:,h1:h2,:] + (1- ratio_mask) * F_t[:,:,h1:h2,:]

        # return F_n

        x = self.gap(F_n)
        x = x.view(x.size(0), -1)
        bn_x = self.feat_bn(x)
        bn_x = F.normalize(bn_x)
        return bn_x

    def backward_D_basic(self, netD, real, fake):
        # Real
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgan-gp
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss += gradient_penalty

        return D_loss

    def backward_D(self):
        base_function._unfreeze(self.net_D)
        self.loss_dis_img_gen = self.backward_D_basic(self.net_D, self.source_image, self.fake_image)
        self.loss_D = self.loss_dis_img_gen
        self.loss_D.backward()

    def backward_G_basic(self, fake_image, target_image, use_d):
        # Calculate reconstruction loss
        loss_app_gen = self.L1loss(fake_image, target_image)
        loss_app_gen = loss_app_gen * self.opt.lambda_rec

        # Calculate GAN loss
        loss_ad_gen = None
        if use_d:
            base_function._freeze(self.net_D)
            D_fake = self.net_D(fake_image)
            loss_ad_gen = self.GANloss(D_fake, True, False) * self.opt.lambda_g

        # Calculate perceptual loss
        loss_content_gen, loss_style_gen = self.Vggloss(fake_image, target_image)
        loss_style_gen = loss_style_gen * self.opt.lambda_style
        loss_content_gen = loss_content_gen * self.opt.lambda_content
        # loss_style_gen, loss_content_gen = None, None

        return loss_app_gen, loss_ad_gen, loss_style_gen, loss_content_gen

    def backward_G(self, loss_nl=None):
        base_function._unfreeze(self.net_D)

        self.loss_app_gen, self.loss_ad_gen, self.loss_style_gen, self.loss_content_gen = self.backward_G_basic(self.fake_image, self.source_image, use_d=True)
        # self.loss_app_gen, self.loss_ad_gen, self.loss_style_gen, self.loss_content_gen = self.backward_G_basic(self.fake_image, self.source_image, use_d=False)
        # self.loss_G = torch.tensor(0.0).cuda()
        # self.loss_G = self.loss_app_gen + self.loss_ad_gen
        self.loss_G = self.loss_app_gen + self.loss_ad_gen + self.loss_style_gen + self.loss_content_gen
        # loss bp from reid part
        if loss_nl is not None:
           self.loss_G = (self.loss_G + loss_nl)
        
        self.loss_G.backward()

    # def get_loss_G(self, loss_nl=None):
    #     base_function._unfreeze(self.net_D)

    #     self.loss_app_gen, self.loss_ad_gen, self.loss_style_gen, self.loss_content_gen = self.backward_G_basic(self.fake_image, self.source_image, use_d=True)
    #     self.loss_G = self.loss_app_gen + self.loss_style_gen + self.loss_content_gen + self.loss_ad_gen
    #     # loss bp from reid part
    #     if loss_nl is not None:
    #        self.loss_G = self.loss_G + loss_nl 
        
    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def optimize_generated(self, loss_nl=None):
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G(loss_nl)
        # self.loss_G.backward()
        self.optimizer_G.step()

    def optimize_parameters_adaptor(self, loss):
        self.optimizer_A.zero_grad()
        loss.backward()
        self.optimizer_A.step()

