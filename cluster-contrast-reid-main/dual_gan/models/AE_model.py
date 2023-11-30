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
        parser.add_argument('--num_feats', type=int, default=256, help='number of layers in G')
        parser.add_argument('--save_input', action='store_true', help="whether save the input images when testing")
        parser.add_argument('--num_blocks', type=int, default=3, help="number of resblocks")
        parser.add_argument('--affine', action='store_true', default=True, help="affine in PTM")
        parser.add_argument('--nhead', type=int, default=2, help="number of heads in PTM")
        parser.add_argument('--num_CABs', type=int, default=2, help="number of CABs in PTM")
        parser.add_argument('--num_TTBs', type=int, default=2, help="number of CABs in PTM")
        parser.add_argument('--bipath_gan', action='store_true', help='bipath gan')

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
        # self.loss_names = ['app_gen', 'content_gen', 'style_gen', 'ad_gen', 'dis_img_gen', 'G', 'D']
        self.loss_names = ['G', 'D']
        self.model_names = ['G']
        # self.visual_names = ['source_image', 'source_pose', 'target_image', 'target_pose', 'fake_image', 'fake_image_n']
        # self.visual_names = ['source_image', 'source_pose', 'target_image', 'target_pose', 'fake_image', 'mixed_image']
        self.visual_names = ['source_image', 'source_pose', 'target_image', 'target_pose', 'fake_image', 'mixed_image']
        self.model_gen = opt.model_gen
        num_feats = opt.num_feats
        G_layer = opt.layers_g
        # num_feats = 2048 if (opt.model_gen == 'DEC' or opt.model_gen == 'FD') else 256
        # G_layer = 4 if opt.model_gen == 'DEC' else 4

        # for feature fusion output
        self.gap = nn.AdaptiveAvgPool2d(1).to(opt.device)
        self.feat_bn = nn.BatchNorm1d(num_feats).to(opt.device)

        if opt.model_gen == 'Pose':
            self.net_G = networks.define_G(opt, image_nc=opt.image_nc, pose_nc=opt.pose_nc, ngf=64, img_f=num_feats,
                                       encoder_layer=G_layer, norm=opt.norm, activation='LeakyReLU',
                                       use_spect=opt.use_spect_g, use_coord=opt.use_coord, output_nc=3, num_blocks=opt.num_blocks, 
                                       affine=True, nhead=opt.nhead, num_CABs=opt.num_CABs, num_TTBs=opt.num_TTBs)
        else:
            self.net_G = networks.define_G(opt, image_nc=opt.image_nc, pose_nc=opt.pose_nc, ngf=64, img_f=num_feats,
                                       encoder_layer=G_layer, norm=opt.norm, activation='LeakyReLU',
                                       use_spect=opt.use_spect_g, use_coord=opt.use_coord, output_nc=3, num_blocks=opt.num_blocks)
            
        ##################################################################################################################
        # bipath
        if opt.bipath_gan:
            self.model_names.append('Gb')
            if opt.model_gen == 'Pose':
                self.net_Gb = networks.define_G(opt, image_nc=opt.image_nc, pose_nc=opt.pose_nc, ngf=64, img_f=num_feats,
                                        encoder_layer=G_layer, norm=opt.norm, activation='LeakyReLU',
                                        use_spect=opt.use_spect_g, use_coord=opt.use_coord, output_nc=3, num_blocks=opt.num_CABs,
                                        affine=True, nhead=opt.nhead, num_CABs=opt.num_CABs, num_TTBs=opt.num_TTBs)
            else:
                self.net_Gb = networks.define_G(opt, image_nc=opt.image_nc, pose_nc=opt.pose_nc, ngf=64, img_f=num_feats,
                                        encoder_layer=G_layer, norm=opt.norm, activation='LeakyReLU',
                                        use_spect=opt.use_spect_g, use_coord=opt.use_coord, output_nc=3, num_blocks=opt.num_CABs)
                
        ##################################################################################################################
        
        self.use_adp = opt.use_adp
        if self.use_adp:
            self.model_names.append('A')
            self.net_A = networks.Resize_ReID(image_nc=opt.image_nc)

        # Discriminator network
        if self.gan_train:
            self.model_names.append('D')
            self.net_D = networks.define_D(opt, ndf=32, img_f=128, layers=opt.dis_layers, use_spect=opt.use_spect_d)
            # self.net_D = networks.define_D(opt, ndf=64, img_f=256, layers=opt.dis_layers, use_spect=opt.use_spect_d)

            ##################################################################################################################
            # bipath
            if opt.bipath_gan:
                self.model_names.append('Db')
                self.net_Db = networks.define_D(opt, ndf=32, img_f=128, layers=opt.dis_layers, use_spect=opt.use_spect_d)
                
            ##################################################################################################################

        if self.opt.verbose:
                print('---------- Networks initialized -------------')
        # set loss functions and optimizers
        if self.gan_train:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            #self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.gan_lr

            self.GANloss = external_function.GANLoss(opt.gan_mode).to(opt.device)
            # self.L1loss = torch.nn.L1Loss()
            self.L1loss = torch.nn.L1Loss(reduction="none")
            if not opt.no_vgg_loss:
                self.Vggloss = external_function.VGGLoss().to(opt.device)

            # define the optimizer
            
            ##################################################################################################################
            # bipath
            if opt.bipath_gan:
                self.optimizer_G = torch.optim.Adam(
                    [
                        {"params": itertools.chain(filter(lambda p: p.requires_grad, self.net_G.parameters()))},
                        {"params": itertools.chain(filter(lambda p: p.requires_grad, self.net_Gb.parameters()))},
                    ],
                    lr=opt.gan_lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.Adam(itertools.chain(
                    filter(lambda p: p.requires_grad, self.net_G.parameters())),
                    lr=opt.gan_lr, betas=(opt.beta1, 0.999))
            ##################################################################################################################
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            ##################################################################################################################
            # bipath
            if opt.bipath_gan:
                self.optimizer_D = torch.optim.Adam(
                    [
                        {"params": itertools.chain(filter(lambda p: p.requires_grad, self.net_D.parameters()))},
                        {"params": itertools.chain(filter(lambda p: p.requires_grad, self.net_Db.parameters()))},
                    ],
                    lr=opt.gan_lr * opt.ratio_g2d, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_D = torch.optim.Adam(itertools.chain(
                    filter(lambda p: p.requires_grad, self.net_D.parameters())),
                    lr=opt.gan_lr * opt.ratio_g2d, betas=(opt.beta1, 0.999))
            ##################################################################################################################
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
            # source_image = torch.index_select(inputs, 0, b_id)
            source_image, source_pose = torch.index_select(inputs['Xs'], 0, b_id), torch.index_select(inputs['Ps'], 0, b_id)
            # target_image = torch.index_select(input['Xt'], 0, b_id)
        else:
            # source_image = inputs
            source_image = inputs['Xs']
            if self.opt.model_gen == 'Pose':
                source_pose = inputs['Ps']
            # target_image = input['Xt']
        self.source_image = source_image.cuda()
        if self.opt.model_gen == 'Pose':
            self.source_pose = source_pose.cuda()
        # self.target_image = target_image.cuda()

    def forward(self):
        # Encode Inputs
        self.fake_image = self.net_G(self.source_image)

    def synthesize(self, features):
        self.fake_image = self.net_G(features)    
        
    def synthesize_p(self, features):
        self.fake_image = self.net_G(features, self.source_pose)
        return self.fake_image

    def synthesize_mix(self, f_gan, f_out, f_cluster, group_size, lambda_fus):
        bs, fdim, fh, fw = f_gan.shape

        sim = torch.exp(torch.einsum('n c, m c -> n m', [f_cluster, f_out]))

        id_mask = torch.eye(f_cluster.shape[0]).repeat_interleave(group_size, dim=1).cuda()
        # select the farthest in id
        in_id = torch.argmin(id_mask * sim + (1-id_mask) * torch.max(sim), dim=1)
        # print(in_id)
        # select the nearest out id
        out_id = torch.argmax((1-id_mask) * sim, dim=1)
        # print(out_id)
        
        F_mix = F.normalize(lambda_fus * f_gan[in_id] + (1-lambda_fus) * f_gan[out_id], dim=1) 
        
        syn_images = self.net_G(torch.cat([f_gan, F_mix], dim=0))
        
        self.fake_image = syn_images[:bs]
        self.mixed_image = syn_images[bs:]

        return self.mixed_image
    
    def synthesize_mix_p(self, f_gan, f_gan_ex):
        bs = f_gan.shape[0]
        exbs = f_gan_ex.shape[0]
        
        # p_idx = torch.randperm(bs)    
        p_idx = torch.randint(bs, (exbs,))    
        syn_images = self.net_G( torch.cat([f_gan, f_gan_ex], dim=0),
                                 torch.cat([self.source_pose, self.source_pose[p_idx]], dim=0))
        
        self.fake_image = syn_images[:bs]
        self.mixed_image = syn_images[bs:]

        return self.mixed_image.detach()
    
    def synthesize_fgan(self):
        F_s = self.net_G.module.forward_enc(self.source_image)
        return F_s.detach()
    
    def synthesize_fc(self, reid_f, group_size=16):
        # self.fake_image = self.net_G(self.source_image)
        F_s = self.net_G.module.forward_enc(self.source_image)
        # attention here 
        # F_c = torch.mean(torch.stack(torch.split(F_s, group_size, dim=0), dim=0), dim=1)

        # self-attention blocks
        # hard postive and negative samples
        # attention map threshold 
        # skip connected and normed
        # num_ids = F_s.shape[0] // group_size

        # F_c = self.opt.lambda_fus * F_c + (1-self.opt.lambda_fus) * torch.flip(F_c, dims=[0])

        self.fake_image = self.net_G.module.forward_dec(self.hard_mix(F_s, reid_f, group_size))

        return self.fake_image
    
    def hard_mix(self, F_s, reid_f, group_size):
        _, fdim = reid_f.shape
        
        anchor_feature = F.normalize(torch.mean(reid_f.reshape(-1, group_size, fdim), dim=1))
        instacne_feature = F.normalize(reid_f)

        sim = torch.exp(torch.einsum('n c, m c -> n m', [anchor_feature, instacne_feature]))

        id_mask = torch.eye(anchor_feature.shape[0]).repeat_interleave(group_size, dim=1).cuda()
        # select the farthest in id
        in_id = torch.argmin(id_mask * sim + (1-id_mask) * torch.max(sim), dim=1)
        # select the nearest out id
        out_id = torch.argmax((1-id_mask) * sim, dim=1)

        # return torch.where(torch.rand_like(F_c) < self.opt.lambda_fus, F_s[in_id], F_s[out_id])

        # return self.opt.lambda_fus * F_s[in_id] + (1-self.opt.lambda_fus) * self.cross_attention(F_s[in_id], F_s[out_id])
    
        return self.opt.lambda_fus * F_s[in_id] + (1-self.opt.lambda_fus) * F_s[out_id]

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
        loss_app_gen = None
        loss_app_gen = self.L1loss(fake_image, target_image)
        loss_app_gen = loss_app_gen * self.opt.lambda_rec

        # Calculate GAN loss
        loss_ad_gen = None
        if use_d:
            base_function._freeze(self.net_D)
            D_fake = self.net_D(fake_image)
            loss_ad_gen = self.GANloss(D_fake, True, False) * self.opt.lambda_g

        # Calculate perceptual loss
        if self.opt.no_vgg_loss:
            loss_style_gen, loss_content_gen = None, None
        else:
            loss_content_gen, loss_style_gen = self.Vggloss(fake_image, target_image)
            loss_style_gen = loss_style_gen * self.opt.lambda_style
            loss_content_gen = loss_content_gen * self.opt.lambda_content

        return loss_app_gen, loss_ad_gen, loss_style_gen, loss_content_gen

    def backward_G(self, loss_nl=None, group_size=16):
        base_function._unfreeze(self.net_D)

        self.loss_app_gen, self.loss_ad_gen, self.loss_style_gen, self.loss_content_gen = self.backward_G_basic(self.fake_image, self.source_image, use_d=True)
        # self.loss_app_gen, self.loss_ad_gen, self.loss_style_gen, self.loss_content_gen = self.backward_G_basic(self.fake_image, self.source_image, use_d=False)
        # self.loss_G = torch.tensor(0.0).cuda()

        loss_G = self.loss_app_gen.flatten(1).mean(dim=-1) + self.loss_ad_gen.flatten(1).mean(dim=-1)

        self.loss_G = loss_G.mean()
        # loss bp from reid part
        if loss_nl is not None:
           self.loss_G = (self.loss_G + loss_nl)
        
        self.loss_G.backward()    
        
    def get_loss_G(self, group_size=None, cf_temp=0.2, need_cm=True, cluster_features=None):
        base_function._unfreeze(self.net_D)

        self.loss_app_gen, self.loss_ad_gen, self.loss_style_gen, self.loss_content_gen = self.backward_G_basic(self.fake_image, self.source_image, use_d=True)
        # self.loss_app_gen, self.loss_ad_gen, self.loss_style_gen, self.loss_content_gen = self.backward_G_basic(self.fake_image, self.source_image, use_d=False)
        # self.loss_G = torch.tensor(0.0).cuda()

        if need_cm:
            # synthesize from clusters
            
            cluster_image = self.net_G(cluster_features, self.source_pose)
            loss_rec = self.L1loss(cluster_image, self.source_image).flatten(1).mean(dim=-1)
            # conf_mask = (-loss_rec.reshape(-1, group_size) / cf_temp).softmax(dim=-1).flatten(0).detach()
            
            loss_G = self.loss_app_gen.flatten(1).mean(dim=-1) + self.loss_ad_gen.flatten(1).mean(dim=-1)

            self.loss_G = loss_G.mean()
            return self.loss_G, loss_rec
        
        else:
            self.loss_G = self.loss_app_gen.mean() + self.loss_ad_gen.mean() 
            return self.loss_G
        
    def get_L1_loss(self, with_dis=False):
        if with_dis:
            base_function._unfreeze(self.net_D)
            loss_app_gen, loss_ad_gen, _, _ = self.backward_G_basic(self.fake_image, self.source_image, use_d=True)
            loss_rec = loss_app_gen.flatten(1).mean(dim=-1)
            loss_dis = loss_ad_gen.flatten(1).mean(dim=-1)
            # print(loss_rec.mean())
            # print(loss_dis.mean())
            return loss_rec + loss_dis
        else:
            loss_app_gen = self.L1loss(self.fake_image, self.source_image)
            loss_rec = loss_app_gen.flatten(1).mean(dim=-1)
            return loss_rec
        
    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def optimize_generated(self):
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()




