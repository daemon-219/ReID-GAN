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

class ClusterContrastWithGANTrainer(object):
    def __init__(self, encoder, GAN=None, writer=None, memory=None, opt=None):
        super(ClusterContrastWithGANTrainer, self).__init__()
        self.encoder = encoder
        if GAN is None:
            raise('GAN not implemented!')
        self.gan = GAN
        self.memory = memory

        self.memoryb = memory

        self.writer = writer
        # self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        # self.f_metric = nn.MSELoss()
        self.f_metric = nn.L1Loss()
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
            reid_inputs, labels, indexes = self._parse_data(inputs[0])
            gan_inputs = inputs[1]
            self.gan.set_input(gan_inputs)

            """
            resnet
            """
            f_out = self._forward(reid_inputs)
            fc_image = self.gan.synthesize_fc(f_out.detach(), group_size)

            self.encoder.eval()
            f_ex = self._forward(my_transform(fc_image))
            self.encoder.train()

            loss = self.memory(f_out, labels, ex_f=f_ex.detach()).mean()
            # loss = self.memory(f_out, labels).mean()

            """
            resnet_mp
            """
            
            # f_g, f_p1, f_p2, f_gc = self._forward(reid_inputs)

            # fc_image = self.gan.synthesize_fc(f_g.detach(), group_size)
            # # # F_gan = F.normalize(self.gan.synthesize_fgan(), dim=1)

            # # # loss_rec = self.f_metric(f_gan, F_gan)
            # self.encoder.eval()
            # f_ex = self._forward(my_transform(fc_image))
            # self.encoder.train()
            
            # loss_cl = (self.intra_cl(f_p1, f_p1.detach()).mean() 
            #            + self.intra_cl(f_p2, f_p2.detach()).mean() 
            #            + self.intra_cl(f_g, f_g.detach()).mean())

            # loss = self.memory(f_gc, labels, ex_f=f_ex.detach()).mean() + loss_cl
            # # loss = self.memory(f_gc, labels).mean() + loss_cl

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
                    #   'Intra_cl {:.3f}\t'
                      'Loss {:.3f} ({:.3f})\n'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                            #   loss_cl.item(), 
                              losses.val, losses.avg,
                              ))

    def train_all(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, acc_iters=0, conf_weight=None):
        print("train both gan and reid")
        # self.encoder.eval()
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

            """
            resnet
            """
            f_out, f_gan = self._forward(reid_inputs)

            loss_ori = self.memory(f_out, labels)

            """
            resnet_mp
            """
            # f_g, f_p1, f_p2, f_gc, f_gan = self._forward(reid_inputs)
            # # f_out = self._forward(reid_inputs)

            # loss_cl = (self.intra_cl(f_p1, f_p1.detach()).mean() 
            #            + self.intra_cl(f_p2, f_p2.detach()).mean() 
            #            + self.intra_cl(f_g, f_g.detach()).mean())

            # loss_ori = self.memory(f_gc, labels)

            # print(f_gan.shape)
            
            fake_images = self.gan.synthesize_p(f_gan)
            # fake_images = self.gan.synthesize(f_gan)

            # print(fake_images.shape)

            # loss_G = self.gan.get_loss_G(need_cm=False)

            _, fc, fh, fw = f_gan.shape
            temp_cluster = torch.mean(f_gan.reshape(-1, group_size, fc, fh, fw), dim=1)
            temp_cluster = torch.repeat_interleave(temp_cluster, group_size, 0)

            # print(temp_cluster.shape)

            loss_G, cluster_l1 = self.gan.get_loss_G(cluster_features=temp_cluster.detach())
            # loss_G, cluster_l1 = self.gan.get_loss_G(cluster_features=self.memory.features[labels])

            if i == 0:
                b_1 = inputs[1]['gt_label'][:group_size].cuda()
                X = torch.sort(loss_ori[:group_size], -1, descending=False)
                print("cl loss", X[0])
                # print("conf mask", conf_mask[:group_size].gather(-1, X[1]))
                print("gt_label_cl", b_1.gather(-1, X[1]))

                # Y = torch.sort(conf_mask[:group_size], -1, descending=False)
                Y = torch.sort(cluster_l1[:group_size], -1, descending=False)
                print("rec loss", Y[0])
                print("gt_label_rec", b_1.gather(-1, Y[1]))   

            conf_mask = torch.ones_like(indexes).cuda()
            _, blocked_ids = torch.topk(cluster_l1.reshape(-1, group_size), 1)
            # print(cluster_l1.reshape(-1, group_size))
            # print(blocked_ids)
            conf_mask = conf_mask.reshape(-1, group_size).scatter(-1, blocked_ids, 0.).reshape(-1)
            
            # print(conf_mask.reshape(-1, group_size))
            loss_ori = conf_mask * loss_ori

            loss = loss_ori.mean() + loss_G
            # loss = loss_ori.mean() + loss_cl + loss_G

            self.gan.optimizer_D.zero_grad()
            self.gan.backward_D()
            self.gan.optimizer_D.step()

            self.gan.optimizer_G.zero_grad()
            optimizer.zero_grad()

            loss.backward()

            self.gan.optimizer_G.step()
            optimizer.step()

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
                # "reid_loss": loss_cl.item(),
                "total_loss": losses.val})

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    'Loss {:.3f} ({:.3f})\t'
                    # 'FRECLoss: {:.3f}\t'
                    # 'Intra_cl {:.3f}\t'
                    'GANLoss: G:{:.3f} D:{:.3f}\n'
                    .format(epoch, i + 1, len(data_loader),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg,
                            losses.val, losses.avg,
                            # loss_rec.item(),
                            # loss_cl.item(),
                            gan_losses['G'], gan_losses['D']
                            ))
                 
    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs, fuse=True):
        # return self.encoder.module.forward_train(inputs, in_trainer=True)
        if fuse:
            return self.encoder(inputs)
        else:
            return self.encoder(inputs, fuse=fuse)
        
    def intra_cl(self, q, k, group_size=16):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        logits = torch.einsum('n c, m c -> n m', [q, k]) / self.T
        qs, ks = logits.shape
        logits = torch.sum(logits.reshape(qs, -1, group_size), dim=-1)
        
        # N = logits.shape[0]  # batch size per GPU
        targets = torch.arange(group_size, dtype=torch.long).repeat_interleave(group_size).cuda()
        return F.cross_entropy(logits, targets, reduction="none")
    

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

            loss = self.memory(f_out, labels).mean()
            # loss = self.memory(f_out, labels)

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
            
            # gan_inputs = inputs[1]
            gan_inputs = inputs
            self.gan.set_input(gan_inputs)

            # fake_images = self.gan.synthesize_p()
            
            # self.gan.optimize_generated()

            self.gan.optimize_parameters()

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
                
    
