from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch
import torch.nn as nn
from torch.nn import functional as F
from clustercontrast.utils.data.diff_augs import my_resize, my_transform, my_normalize

import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

import wandb

class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, acc_iters=0):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)

            # forward
            f_out = self._forward(inputs)
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)
            loss = self.memory(f_out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)
    
def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


def match_loss(gw_syn, gw_real, dis_metric='ours', dis_dir=None, num_it=0):
    """
    TODO: try contrastive loss
    """
    dis = torch.tensor(0.0).cuda()
    # dis_list={4:[], 3:[], 2:[], 1:[]}

    if dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

            # dis_wb = distance_wb(gwr, gws)
            # # if dis_wb:
            # #     print(gwr.shape, dis_wb)
            # dis += dis_wb
            # dis_list[len(gwr.shape)].append(dis_wb.cpu().detach().numpy())

    elif dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    elif dis_metric == 'cos_m':
        for ig in range(len(gw_real)):
            if len(gw_real[ig].shape) == 4:
                # only for conv layers
                gw_real_vec = gw_real[ig].reshape(-1)
                gw_syn_vec = gw_syn[ig].reshape(-1)
                dis += 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        raise('unknown distance function: %s'%dis_metric)
    
    # plot_loss(dis_list, dis_dir, num_it)

    return dis

def plot_loss(dis_list, dis_dir, num_it):
    v = dis_list[4]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(x=range(len(v)), height=v)
    ax.set_title("Gradient matching loss", fontsize=15)
    layer_type = "conv"
    plt.savefig(osp.join(dis_dir, 'GM'+layer_type+str(num_it)), bbox_inches='tight')
    plt.close()

 
class GANTrainer(object):
    def __init__(self, GAN=None, encoder=None, writer=None, opt=None):
        super(GANTrainer, self).__init__()
        if GAN is None:
            raise('GAN not implemented!')
        self.gan = GAN
        self.encoder = encoder
        self.writer = writer

        if opt is not None:
            self.opt = opt
            self.T = opt.cl_temp
                
    def train_gan(self, epoch, data_loader, print_freq=10, train_iters=400, acc_iters=0):
        print("train gan")

        batch_time = AverageMeter()
        data_time = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            # print(len(inputs))
            data_time.update(time.time() - end)

            # process inputs
            # gan_inputs = inputs[1]['Xs']
            gan_inputs = inputs[1]
            self.gan.set_input(gan_inputs)
            # self.gan.target_image = reid_inputs         
            # self.gan.set_input(gan_inputs[::group_size])
            # self.gan.target_image = reid_inputs[::group_size]

            # gan_inputs = self.memory.features[labels]
            # gan_inputs = F.normalize(gan_inputs)
            # fake_images = self.gan.cond_synthesize(gan_inputs)

            fake_images = self.gan.synthesize()
            
            self.gan.optimize_generated()

            # add writer
            if self.writer is not None:
                # gan model
                total_steps = acc_iters + i
                gan_losses = self.gan.get_current_errors()
                self.writer.add_scalar('Loss/G_loss', gan_losses['G'], total_steps)
                self.writer.add_scalar('Loss/D_loss', gan_losses['D'], total_steps)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            wandb.log({
                "GANLoss_G": gan_losses['G'],
                "GANLoss_D": gan_losses['D']})

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    'GANLoss: G:{:.3f} D:{:.3f}\n'
                    .format(epoch, i + 1, len(data_loader),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg,
                            gan_losses['G'], gan_losses['D']
                            ))
                
    def train_gan_with_reid_enc(self, epoch, data_loader, print_freq=10, train_iters=400, acc_iters=0):
        print("train gan with reid encoder")

        self.encoder.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            # print(len(inputs))
            data_time.update(time.time() - end)

            # process inputs
            # gan_inputs = inputs[1]['Xs']
            gan_inputs = inputs[1]
            self.gan.set_input(gan_inputs)
            # self.gan.target_image = reid_inputs         
            # self.gan.set_input(gan_inputs[::group_size])
            # self.gan.target_image = reid_inputs[::group_size]

            # gan_inputs = self.memory.features[labels]
            # gan_inputs = F.normalize(gan_inputs)
            # fake_images = self.gan.cond_synthesize(gan_inputs)

            f_reid = self.encoder(my_transform(gan_inputs['Xs']))

            fake_images = self.gan.synthesize_p(f_reid)
            
            self.gan.optimize_generated()

            # add writer
            if self.writer is not None:
                # gan model
                total_steps = acc_iters + i
                gan_losses = self.gan.get_current_errors()
                self.writer.add_scalar('Loss/G_loss', gan_losses['G'], total_steps)
                self.writer.add_scalar('Loss/D_loss', gan_losses['D'], total_steps)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            wandb.log({
                "GANLoss_G": gan_losses['G'],
                "GANLoss_D": gan_losses['D']})

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    'GANLoss: G:{:.3f} D:{:.3f}\n'
                    .format(epoch, i + 1, len(data_loader),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg,
                            gan_losses['G'], gan_losses['D']
                            ))

class ClusterContrastWithGANTrainer(object):
    def __init__(self, encoder, GAN=None, writer=None, memory=None, opt=None):
        super(ClusterContrastWithGANTrainer, self).__init__()
        self.encoder = encoder
        if GAN is None:
            raise('GAN not implemented!')
        self.gan = GAN
        self.memory = memory
        self.writer = writer
        # self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.f_metric = nn.MSELoss()
        # self.f_metric = nn.L1Loss()
        if opt is not None:
            self.opt = opt
            self.T = opt.cl_temp

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, acc_iters=0):
        """
        TODO: GAN in test mode:
        """
        if self.gan.use_adp:
            print("train adaptor and reid")

        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()

        group_size = self.opt.num_instances 
        
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            # reid_inputs, labels, indexes = self._parse_data(inputs)
            reid_inputs, labels, indexes = self._parse_data(inputs[0])
            gan_inputs = inputs[1]
            self.gan.set_input(gan_inputs)
            # self.gan.set_input(gan_inputs[::group_size])
            # self.gan.target_image = reid_inputs[::group_size]

            """
            TODO: negative samples:

            # self.gan.set_input(inputs[1])
            # fake_image_t, _ = self.gan.synthesize()

            # one sythetic sample for each id
            b_id = torch.arange(0, reid_inputs.shape[0], 16)

            self.gan.set_input(inputs[1], b_id)
            fake_image_n = self.gan.synthesize_pair()
            # print(fake_image_n.shape)
            
            # do transform here

            ex_inputs = my_resize(fake_image_n, (reid_inputs.shape[2], reid_inputs.shape[3]))
            ex_labels = torch.index_select(labels, 0, b_id.cuda())
            # p_inputs = my_resize(inputs[1]["Xs"], (reid_inputs.shape[2], reid_inputs.shape[3]))
            # n_inputs = my_resize(fake_image_n, (reid_inputs.shape[2], reid_inputs.shape[3]))

            # fake target as extended inputs
            new_inputs = torch.cat([reid_inputs, ex_inputs], dim=0)
            # labels = labels.repeat(2) # postive
            labels = torch.cat([labels, ex_labels], dim=0) # extended

            # forward
            f_out = self._forward(new_inputs)

            # triplet loss

            # f_out = self._forward(reid_inputs)
            # with torch.no_grad():
            #     f_p = self._forward(p_inputs)
            #     f_n = self._forward(n_inputs)

            # loss_tri = self.triplet_loss(f_out, f_p, f_n)
            # loss = loss_tri + self.memory(f_out, labels)

            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            """

            """
            TODO: postive samples:
            """

            # with torch.cuda.amp.autocast():

            # forward
            # shape: b, 2048, 16, 8
            # fr_0, fr_1, fr_2, fr_3, fr_4, f_out = self._forward(reid_inputs)
            f_out = self._forward(reid_inputs)

            # shape: b, 256, 16, 8
            # fake_features = self.gan.synthesize()
            # fake_images = self.gan.synthesize()
            fc_image = self.gan.synthesize_fc(group_size)
            
            # shape: b, 2048, 16, 8
            # f_gan = self.encoder.module.feature_mapping(fake_features.detach())

            # loss_f = self.opt.lambda_nl * self.f_metric(f_out.detach(), f_gan) / f_gan.shape[0]
            # loss_f = self.opt.lambda_nl * self.f_metric(f_out.detach()[::group_size], f_gan) / f_gan.shape[0]

            self.encoder.eval()
            f_ex = self._forward(my_transform(fc_image))
            # # f_ex = self.gan.feature_fusion(fr_4, torch.flip(fr_4, dims=[0]))
            self.encoder.train()

            # f_out = torch.cat([f_out, f_ex], dim=0)
            # labels = torch.cat([labels, labels[::group_size]], dim=0)

            # # f_ex = 0.9 * f_out + 0.1 * f_gan
            # f_ex = torch.mean(torch.stack(torch.split(f_gan.detach(), group_size, dim=0), dim=0), dim=1)
            # f_ex = torch.mean(torch.stack(torch.split(f_out.detach(), group_size, dim=0), dim=0), dim=1)

            # loss = self.memory(f_out, labels)
            loss = self.memory(f_out, labels, ex_f=f_ex.detach())
            # loss = self.memory(f_out, labels, ex_f=f_ex) + loss_f
            # loss = self.memory(f_out, labels, ex_f=f_gan.detach()) + loss_f
            # loss = self.memory(f_out, labels, ex_f=f_out.detach()[::group_size])

            # f_tar = torch.cat([f_tar_p, f_tar_n], dim=0)
            # loss_cl = self.cl_loss(f_out, self.encoder.module.predictor(f_tar))  
            # loss_cl = self.cl_loss(f_out, f_tar) 
            
            
            # self.gan.set_input(inputs[1])
            # fake_image_s, fake_image_t = self.gan.synthesize(is_train=True)

            # # (batch, feature_dim) = (b, 2048)

            # self.encoder.eval()
            # f_real_out = self._forward(my_transform(self.gan.source_image))
            # f_syn_out = self._forward(my_normalize(fake_image_s))

            # self.encoder.train()
            # get fake image gradient
            # net_parameters = list(params for params in self.encoder.parameters() if params.requires_grad)
            # gw_real = torch.autograd.grad(loss_ori, net_parameters, retain_graph=True)
            # gw_real = list((_.detach().clone() for _ in gw_real))

            # # do not update cluster memory for synthesized inputs
            # loss_syn = self.memory(f_ex, labels, update=False)
            # gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

            # gm_loss = self.opt.lambda_nl * match_loss(gw_syn, gw_real, dis_dir=self.opt.logs_dir, num_it=acc_iters + i)
            
            # backward for adaptor
            # self.gan.optimize_parameters_adaptor(gm_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # add writer
            if self.writer is not None:
                # gan model
                total_steps = acc_iters + i
                # reid model
                self.writer.add_scalar('Loss/reid_loss', losses.val, total_steps)

            wandb.log({
                "reid_loss": losses.val})

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\n'
                    #   'Loss FM {:.3f}\n'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                            #   loss_f.item()
                              ))
                
    # def train_all(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, acc_iters=0):
    #     print("train both gan and reid")
    #     # self.encoder.eval()
    #     self.encoder.train()

    #     batch_time = AverageMeter()
    #     data_time = AverageMeter()

    #     losses = AverageMeter()

    #     end = time.time()

    #     # batch accumulation parameter
    #     # accum_iter = 4 

    #     group_size = self.opt.num_instances

    #     for i in range(train_iters):
    #         # load data
    #         inputs = data_loader.next()
    #         # print(len(inputs))
    #         data_time.update(time.time() - end)

    #         # process inputs
    #         reid_inputs, labels, indexes = self._parse_data(inputs[0])
    #         # gan_inputs = inputs[1]['Xs']
    #         self.gan.set_input(inputs[1])
    #         # self.gan.target_image = reid_inputs         
    #         # self.gan.set_input(gan_inputs[::group_size])
    #         # self.gan.target_image = reid_inputs[::group_size]

    #         # gan_inputs = self.memory.features[labels]
    #         # gan_inputs = F.normalize(gan_inputs)
    #         # fake_images = self.gan.cond_synthesize(gan_inputs)

    #         # self.encoder.eval()
    #         # f_syn_out = self._forward(my_normalize(fake_images))
    #         # self.encoder.train()

    #         f_out = self._forward(reid_inputs)

    #         mixed_images = self.gan.synthesize_hp(self.memory.features, torch.unique(labels), group_size)
    #         # fake_images = self.gan.synthesize_p(self.memory.features[labels])
    #         # fake_images = self.gan.synthesize_p(f_out.detach())

    #         self.encoder.eval()
    #         f_ex = self._forward(my_transform(mixed_images))
    #         self.encoder.train()

    #         # loss_cl = self.memory(f_ex, labels, update=False)
    #         # loss_rec = self.f_metric(f_ex, f_out.detach())

    #         # neg_l1 = -torch.mean(self.f_metric(self.gan.fake_image, self.gan.source_image).flatten(1), dim=-1)
    #         # conf_mask = (torch.stack(torch.split(neg_l1, group_size, dim=0), dim=0) / self.opt.cf_temp).softmax(dim=-1).flatten(0)
            
    #         conf_mask = self.gan.optimize_generated()
    #         # self.gan.optimize_generated(self.opt.lambda_nl * loss_cl)
    #         # self.gan.optimize_generated(gm_loss)

    #         # forward(self, inputs, targets, update=True, ex_f=None, conf_mask=None)
    #         # loss = self.memory(f_out, labels, ex_f=f_ex.detach())
    #         # loss = self.memory(f_out, labels, ex_f=f_ex.detach(), conf_mask=conf_mask)
    #         loss = self.memory(f_out, labels, ex_f=f_ex.detach(), conf_mask=conf_mask)

    #         optimizer.zero_grad()

    #         # self.loss_G.backward()
    #         # loss.backward(retain_graph=True)
    #         loss.backward()

    #         # self.gan.optimizer_D.step()
    #         # self.gan.optimizer_D.zero_grad()
    #         # self.gan.optimizer_G.step()
    #         optimizer.step()
    #         # optimizer.zero_grad()
    #         # self.gan.optimizer_G.zero_grad()

    #         losses.update(loss.item())

    #         # add writer
    #         if self.writer is not None:
    #             # gan model
    #             total_steps = acc_iters + i
    #             gan_losses = self.gan.get_current_errors()
    #             self.writer.add_scalar('Loss/G_loss', gan_losses['G'], total_steps)
    #             self.writer.add_scalar('Loss/D_loss', gan_losses['D'], total_steps)

    #             # reid model
    #             self.writer.add_scalar('Loss/reid_loss', losses.val, total_steps)
    #             # # neg loss bp into gan
    #             # self.writer.add_scalar('Loss/nl_loss', loss_neg.item())
    #             # cl loss from hard negative samples
    #             # self.writer.add_scalar('Loss/cl_loss', loss_cl.item())
    #             # feature distance mse loss
    #             # self.writer.add_scalar('Loss/frec_loss', frec_loss.item())

    #         # print log
    #         batch_time.update(time.time() - end)
    #         end = time.time()

    #         wandb.log({
    #             "GANLoss_G": gan_losses['G'],
    #             "GANLoss_D": gan_losses['D'], 
    #             "reid_loss": losses.val})

    #         if (i + 1) % print_freq == 0:
    #             print('Epoch: [{}][{}/{}]\t'
    #                 'Time {:.3f} ({:.3f})\t'
    #                 'Data {:.3f} ({:.3f})\t'
    #                 'Loss {:.3f} ({:.3f})\t'
    #                 #   'CLLoss: {:.3f}\t'
    #                 #   'NLLoss: {:.3f}\t'
    #                 'GANLoss: G:{:.3f} D:{:.3f}\t'
    #                 #   'GANLoss: G:{:.3f}\t'
    #                 # 'FRECLoss: {:.3f}\t'
    #                 # 'GidLoss: {:.3f}\n'
    #                 #   'Loss GM {:.3f}\n'
    #                 .format(epoch, i + 1, len(data_loader),
    #                         batch_time.val, batch_time.avg,
    #                         data_time.val, data_time.avg,
    #                         losses.val, losses.avg,
    #                         #   loss_cl.item(),
    #                         #   loss_neg.item(),
    #                         gan_losses['G'], gan_losses['D'],
    #                         #   gan_losses['G'],
    #                         # loss_rec.item(),
    #                         # loss_cl.item(), 
    #                         #   gm_loss.item()
    #                         ))
                
    def train_all(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, acc_iters=0):
        print("train both gan and reid")
        # self.encoder.eval()
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()

        # batch accumulation parameter
        # accum_iter = 4 

        group_size = self.opt.num_instances

        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            # print(len(inputs))
            data_time.update(time.time() - end)

            # process inputs
            reid_inputs, labels, indexes = self._parse_data(inputs[0])
            # gan_inputs = inputs[1]['Xs']
            self.gan.set_input(inputs[1])
            
            # self.gan.target_image = reid_inputs         
            # self.gan.set_input(gan_inputs[::group_size])
            # self.gan.target_image = reid_inputs[::group_size]

            # gan_inputs = self.memory.features[labels]
            # gan_inputs = F.normalize(gan_inputs)
            # fake_images = self.gan.cond_synthesize(gan_inputs)

            # self.encoder.eval()
            # f_syn_out = self._forward(my_normalize(fake_images))
            # self.encoder.train()

            f_out = self._forward(reid_inputs)

            # mixed_images = self.gan.synthesize_hp(self.memory.features, labels, group_size, self.opt.lambda_fus)
            # print(mixed_images.requires_grad)
            mixed_images = self.gan.synthesize_mhp(f_out, self.memory.features, labels, group_size, self.opt.lambda_fus)
            # fake_images = self.gan.synthesize_p(self.memory.features[labels])
            # fake_images = self.gan.synthesize_p(f_out)

            # mixed_images = self.gan.synthesize_hp(self.memory.features, torch.unique(labels), group_size)

            self.encoder.eval()
            f_ex = self._forward(my_transform(mixed_images))
            self.encoder.train()

            # loss_cl = self.memory(f_ex, labels, update=False)
            # loss_rec = self.opt.lambda_nl * self.f_metric(f_ex, f_out.detach().clone())

            # neg_l1 = -torch.mean(self.f_metric(self.gan.fake_image, self.gan.source_image).flatten(1), dim=-1)
            # conf_mask = (torch.stack(torch.split(neg_l1, group_size, dim=0), dim=0) / self.opt.cf_temp).softmax(dim=-1).flatten(0)
            
            # conf_mask = self.gan.optimize_generated()
            # self.gan.optimize_generated(self.opt.lambda_nl * loss_cl)
            # self.gan.optimize_generated(gm_loss)

            # bs = reid_inputs.shape[0]

            # cmb_inputs = torch.cat([reid_inputs, my_transform(fake_images)], dim=0)
            # cmb_labels = labels.repeat(2)

            # f_out = self._forward(my_transform(cmb_inputs))

            # if epoch < 25:
            #     cf_temp = self.opt.cf_temp
            # elif epoch:
            #     cf_temp = -self.opt.cf_temp
         
            loss_G, conf_mask = self.gan.get_loss_G(group_size, self.opt.cf_temp)

            if i == 0:
                print(inputs[1]['Xs_path'][:group_size])
                print(conf_mask[:group_size])

            # loss_cl = self.memory(f_out, labels, conf_mask=conf_mask)
            loss_cl = self.memory(f_out, labels, ex_f=f_ex.detach(), conf_mask=conf_mask)
            # # loss_cl = self.memory(f_out, cmb_labels)
            # loss_cl = self.memory(f_out[:bs], labels)
            # loss_cl += self.memory(f_out[bs:], labels, update=False)
            
            loss = loss_cl + loss_G
            # loss = loss_rec + loss_cl + loss_G

            self.gan.optimizer_D.zero_grad()
            self.gan.backward_D()
            self.gan.optimizer_D.step()

            self.gan.optimizer_G.zero_grad()
            optimizer.zero_grad()

            loss.backward()
            # self.gan.backward_G(loss_cl)

            # optimize trainable clusters for one step
            # self.memory.update_clusters(torch.unique(labels))

            self.gan.optimizer_G.step()
            optimizer.step()

            # forward(self, inputs, targets, update=True, ex_f=None, conf_mask=None)
            # loss = self.memory(f_out, labels, ex_f=f_ex.detach())
            # loss = self.memory(f_out, labels, ex_f=f_ex.detach(), conf_mask=conf_mask)

            losses.update(loss.item())

            # add writer
            if self.writer is not None:
                # gan model
                total_steps = acc_iters + i
                gan_losses = self.gan.get_current_errors()
                self.writer.add_scalar('Loss/G_loss', gan_losses['G'], total_steps)
                self.writer.add_scalar('Loss/D_loss', gan_losses['D'], total_steps)

                # reid model
                self.writer.add_scalar('Loss/reid_loss', losses.val, total_steps)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            wandb.log({
                "GANLoss_G": gan_losses['G'],
                "GANLoss_D": gan_losses['D'], 
                "reid_loss": loss_cl.item(),
                "total_loss": losses.val})

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    'Loss {:.3f} ({:.3f})\t'
                    # 'FRECLoss: {:.3f}\t'
                    'Loss_cl {:.3f}\t'
                    'GANLoss: G:{:.3f} D:{:.3f}\n'
                    .format(epoch, i + 1, len(data_loader),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg,
                            losses.val, losses.avg,
                            # loss_rec.item(),
                            loss_cl.item(),
                            gan_losses['G'], gan_losses['D']
                            ))
                
    def train_all_with_memoery(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, acc_iters=0):
        print("train both gan and reid with trainable memory")

        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()

        group_size = self.opt.num_instances

        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            # print(len(inputs))
            data_time.update(time.time() - end)

            # process inputs
            reid_inputs, labels, indexes = self._parse_data(inputs[0])
            # gan_inputs = inputs[1]['Xs']
            self.gan.set_input(inputs[1])

            f_out = self._forward(reid_inputs)

            mixed_images = self.gan.synthesize_hp(self.memory.normed_clusters, torch.unique(labels), group_size)
            
            # fake_images = self.gan.synthesize_p(f_out.detach())

            self.encoder.eval()
            f_ex = self._forward(my_transform(mixed_images))
            self.encoder.train()

            # loss_cl = self.memory(f_ex, labels, update=False)
            # loss_rec = self.f_metric(f_ex, f_out.detach())

            # neg_l1 = -torch.mean(self.f_metric(fake_images, self.gan.source_image).flatten(1), dim=-1)
            # conf_mask = torch.stack(torch.split(neg_l1, group_size, dim=0), dim=0).softmax(dim=-1).flatten(0)

            # loss = self.memory(f_out, labels)
            loss = self.memory(f_out, labels, ex_f=f_ex.detach())
            # loss = self.memory(f_out, labels, ex_f=f_ex.detach())
            
            # self.gan.optimize_generated()
            # self.gan.optimize_generated(gm_loss)
        
            self.gan.optimizer_D.zero_grad()
            self.gan.backward_D()
            self.gan.optimizer_D.step()

            self.gan.optimizer_G.zero_grad()

            self.gan.backward_G()
            # self.gan.backward_G(loss_cl)

            # optimize trainable clusters for one step
            self.memory.update_clusters(torch.unique(labels))

            self.gan.optimizer_G.step()

            optimizer.zero_grad()

            # self.loss_G.backward()
            # loss.backward(retain_graph=True)
            loss.backward()

            # self.gan.optimizer_D.step()
            # self.gan.optimizer_D.zero_grad()
            # self.gan.optimizer_G.step()
            optimizer.step()
            # optimizer.zero_grad()
            # self.gan.optimizer_G.zero_grad()

            losses.update(loss.item())

            # add writer
            if self.writer is not None:
                # gan model
                total_steps = acc_iters + i
                gan_losses = self.gan.get_current_errors()
                self.writer.add_scalar('Loss/G_loss', gan_losses['G'], total_steps)
                self.writer.add_scalar('Loss/D_loss', gan_losses['D'], total_steps)
                # reid model
                self.writer.add_scalar('Loss/reid_loss', losses.val, total_steps)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            wandb.log({
                "GANLoss_G": gan_losses['G'],
                "GANLoss_D": gan_losses['D'], 
                "reid_loss": losses.val})

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    'Loss {:.3f} ({:.3f})\t'
                    'GANLoss: G:{:.3f} D:{:.3f}\n'
                    .format(epoch, i + 1, len(data_loader),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg,
                            losses.val, losses.avg,
                            #   loss_cl.item(),
                            #   loss_neg.item(),
                            gan_losses['G'], gan_losses['D']
                            ))

    def train_reid(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, acc_iters=0):
        print("warm up stage for reid")
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes = self._parse_data(inputs[0])

            # forward
            f_out = self._forward(inputs)
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)
            loss = self.memory(f_out, labels)

            # print(self.memory.features[labels])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(self.memory.features[labels])

            losses.update(loss.item())

            # add writer
            if self.writer is not None:
                total_steps = acc_iters + i
                # reid model
                self.writer.add_scalar('Loss/reid_loss', losses.val, total_steps)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        # return self.encoder.module.forward_train(inputs, in_trainer=True)
        return self.encoder(inputs)
    
    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # Einstein sum is more intuitive
        logits = torch.einsum('n c, m c -> n m', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = torch.arange(N, dtype=torch.long).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * 2 * self.T    
    
    def cl_loss(self, q, k, group_size=16):
        # input: q(n, c), k(2n, c)
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # Einstein sum is more intuitive
        logits = torch.einsum('n c, m c -> n m', [q, k]) / self.T
        # group sum (n, m // group_size)
        logits = torch.sum(torch.stack(torch.split(logits, group_size, dim=1), dim=1), dim=-1)
        N = logits.shape[-1] // 2  # batch size per GPU
        labels = torch.repeat_interleave(torch.arange(N, dtype=torch.long), group_size, dim=0).cuda()
        return nn.CrossEntropyLoss()(logits, labels)

    
