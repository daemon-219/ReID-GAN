from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch
import torch.nn as nn
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
        # self.f_metric = nn.MSELoss()
        # self.f_metric = nn.L1Loss(reduction='sum')
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
            gan_inputs = inputs[1]['Xs']
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
            fc_image = self.gan.synthesize_fc(group_size)
            
            # shape: b, 2048, 16, 8
            # f_gan = self.encoder.module.feature_mapping(fake_features.detach())

            # loss_f = self.opt.lambda_nl * self.f_metric(f_out.detach(), f_gan) / f_gan.shape[0]
            # loss_f = self.opt.lambda_nl * self.f_metric(f_out.detach()[::group_size], f_gan) / f_gan.shape[0]

            f_ex = self._forward(my_normalize(fc_image))
            # # f_ex = self.gan.feature_fusion(fr_4, torch.flip(fr_4, dims=[0]))

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
                
    def train_all(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, acc_iters=0):
        print("train both gan and reid")
        self.encoder.eval()
        # self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()

        # batch accumulation parameter
        # accum_iter = 4 

        # group_size = 16 

        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            # print(len(inputs))
            data_time.update(time.time() - end)

            # process inputs
            reid_inputs, labels, indexes = self._parse_data(inputs[0])
            gan_inputs = inputs[1]['Xs']
            self.gan.set_input(gan_inputs)
            self.gan.target_image = reid_inputs         
            # self.gan.set_input(gan_inputs[::group_size])
            # self.gan.target_image = reid_inputs[::group_size]

            # with torch.cuda.amp.autocast():

            # fr_0, fr_1, fr_2, fr_3, fr_4, f_out = self._forward(reid_inputs)
            # group mean (n, m) -> (n // group_size, m)
            # torch.mean(torch.stack(torch.split(a, group_size, dim=0), dim=0), dim=1)

            # print(torch.cuda.max_memory_allocated())
            # fake_image = self.gan.synthesize(fr_4.detach().clone())
            # fake_image = self.gan.synthesize(f_out.detach().clone())
            # fake_image = self.gan.synthesize()
            # print(fake_image.shape)
            # print(torch.cuda.max_memory_allocated())
            # self.encoder.eval()
            # ff_0, ff_1, ff_2, ff_3, ff_4, f_syn_out = self._forward(fake_image)
            # print(f_syn_out.shape, f_out.shape)
            # print(torch.cuda.max_memory_allocated())
            # self.encoder.train()

            # self.contrastive_loss(f_ex, f_out.detach().clone())

            # frec_loss = torch.tensor(0.0).cuda()
            # frec_loss += self.f_metric(fr_0.detach()[::group_size], ff_0)
            # frec_loss += self.f_metric(fr_1.detach()[::group_size], ff_1)
            # frec_loss += self.f_metric(fr_2.detach()[::group_size], ff_2)
            # frec_loss += self.f_metric(fr_3.detach()[::group_size], ff_3)
            # frec_loss += self.f_metric(fr_4.detach()[::group_size], ff_4)
            # frec_loss += self.f_metric(f_out.detach()[::group_size], f_syn_out)                
            
            # frec_loss += self.f_metric(fr_0.detach(), ff_0)
            # frec_loss += self.f_metric(fr_1.detach(), ff_1)
            # frec_loss += self.f_metric(fr_2.detach(), ff_2)
            # frec_loss += self.f_metric(fr_3.detach(), ff_3)
            # frec_loss += self.f_metric(fr_4.detach(), ff_4)
            # frec_loss += self.f_metric(f_out.detach(), f_syn_out)
            # print(torch.cuda.max_memory_allocated())

            # # TODO: Gradient Matching Loss and feature reconstruction loss
            # # calculate gradient matching 
            # self.encoder.eval()
            # f_real_out, f_map = self._forward(reid_inputs)
            # fake_image = self.gan.synthesize(f_map.detach().clone())

            # f_syn_out, _ = self._forward(fake_image)

            # self.encoder.train()

            # frec_loss = self.f_metric(f_syn_out, f_real_out.detach().clone())
                    
            # # do not update cluster memory for synthesized inputs
            # loss_syn = self.memory(f_syn_out, labels, update=False)
            # loss_ori = self.memory(f_real_out, labels, update=False)

            # # get gradient
            # net_parameters = list(params for params in self.encoder.parameters() if params.requires_grad)
            # # print(net_parameters)
            # gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

            # gw_real = torch.autograd.grad(loss_ori, net_parameters)
            # gw_real = list((_.detach().clone() for _ in gw_real))

            # gm_loss = match_loss(gw_real, gw_syn, self.opt.dis_metric)


            # fake_image = self.gan.synthesize(f_map.detach().clone())
            # self.encoder.eval()
            # with torch.no_grad():
            #     f_ex, _ = self._forward(fake_image)
            # f_ex, _ = self._forward(my_transform(fake_image))
            # f_ex, _ = self._forward(fake_image)
            # self.encoder.train()

            # frec_loss = self.opt.lambda_nl * self.f_metric(f_out.detach().clone(), f_ex)
            # frec_loss = self.opt.lambda_nl * self.contrastive_loss(f_ex, f_out.detach().clone())

            # loss = self.memory(f_out, labels, ex_f=f_ex.detach().clone()) + frec_loss

            # loss_ori = torch.sum(f_out)
            
            # print(f_out, f_ex)

            """
            TODO: do transform here
            """
            # postive_pair = my_transform(self.gan.target_image, (reid_inputs.shape[2], reid_inputs.shape[3]))
            # fake_image_n = my_transform(fake_image_n, (reid_inputs.shape[2], reid_inputs.shape[3]))
            # input_pairs = torch.cat([postive_pair, fake_image_n], dim=0)
            # with torch.no_grad():
            #     self.encoder.eval()
            #     # f_tar_p = self._forward(postive_pair)
            #     # f_tar_n = self._forward(fake_image_n)
            #     f_tar = self._forward(input_pairs)

            # self.encoder.train()

            # self.encoder.num_feature
            # 2048

            # bp gan loss here
            # loss_neg = self.memory(f_tar_n, labels, update=False)

            # f_out, _= self._forward(reid_inputs)
            # loss = self.memory(f_out, labels)
            
            # fake_image = self.gan.synthesize()
    
            # gm_loss = self.opt.lambda_nl * match_loss(gw_real, gw_real, self.opt.dis_metric, dis_dir=self.opt.logs_dir, num_it=acc_iters + i)
            
            # q = self.encoder.module.predictor(f_tar) 
            # k = f_out.detach().clone()
            # q = torch.mean(torch.stack(torch.tensor_split(q, 4, dim=0), dim=0), dim=1)
            # k = torch.mean(torch.stack(torch.tensor_split(k, 4, dim=0), dim=0), dim=1)
            # loss_cl = self.contrastive_loss(q, k)  

            # TODO: do id-level contrastive learning
            # f_tar = torch.cat([f_tar_p, f_tar_n], dim=0)
            # loss_cl = self.cl_loss(f_out, self.encoder.module.predictor(f_tar))  
            # loss_cl = self.cl_loss(f_out, f_tar)  

            # all backward for gan
            # self.gan.optimize_parameters_generated()
            # self.gan.optimize_parameters_generated()
            self.gan.optimize_parameters()         
            
            # TODO: fake target as extended inputs
            # ex_input = fake_image_t.detach().clone()
            # f_ex_out = self._forward(ex_input)
            # loss_ex = self.memory(f_tar, labels)
            
            # loss = self.opt.lambda_ori * loss_ori + self.opt.lambda_cl * loss_cl

            # TODO: forward for reid
            # f_out, _ = self._forward(reid_inputs)

            # loss = self.memory(f_out, labels, ex_f=f_syn_out.detach().clone())
            # loss_b = self.memory(f_out, labels) / accum_iter
            # loss = self.memory(f_out, labels)
            
            # discriminator opt
            
            # self.gan.backward_D()

            # generator and reid opt

            # loss.backward(retain_graph=True)
            # self.gan.backward_G(self.opt.lambda_nl * frec_loss)
            # self.gan.backward_G()
            
            # self.gan.optimize_generated()
            # self.gan.optimize_generated(frec_loss)
            # self.gan.optimize_generated(gm_loss)

            # self.loss_G.backward()
            # loss.backward(retain_graph=True)
            # loss.backward()

            # self.gan.optimizer_D.step()
            # self.gan.optimizer_D.zero_grad()
            # self.gan.optimizer_G.step()
            # optimizer.step()
            # optimizer.zero_grad()
            # self.gan.optimizer_G.zero_grad()

            # losses.update(loss.item())

            # add writer
            if self.writer is not None:
                # gan model
                total_steps = acc_iters + i
                gan_losses = self.gan.get_current_errors()
                self.writer.add_scalar('Loss/G_loss', gan_losses['G'], total_steps)
                self.writer.add_scalar('Loss/D_loss', gan_losses['D'], total_steps)

                # self.writer.add_scalar('Loss/app_gen_s', gan_losses['app_gen_s'], total_steps)
                # self.writer.add_scalar('Loss/content_gen_s', gan_losses['content_gen_s'], total_steps)
                # self.writer.add_scalar('Loss/style_gen_s', gan_losses['style_gen_s'], total_steps)
                # self.writer.add_scalar('Loss/app_gen_t', gan_losses['app_gen_t'], total_steps)
                # self.writer.add_scalar('Loss/ad_gen_t', gan_losses['ad_gen_t'], total_steps)
                # self.writer.add_scalar('Loss/dis_img_gen_t', gan_losses['dis_img_gen_t'], total_steps)
                # self.writer.add_scalar('Loss/content_gen_t', gan_losses['content_gen_t'], total_steps)
                # self.writer.add_scalar('Loss/style_gen_t', gan_losses['style_gen_t'], total_steps)
                # reid model
                # self.writer.add_scalar('Loss/reid_loss', losses.val, total_steps)
                # # neg loss bp into gan
                # self.writer.add_scalar('Loss/nl_loss', loss_neg.item())
                # cl loss from hard negative samples
                # self.writer.add_scalar('Loss/cl_loss', loss_cl.item())
                # feature distance mse loss
                # self.writer.add_scalar('Loss/frec_loss', frec_loss.item())

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
                    # 'Loss {:.3f} ({:.3f})\t'
                    #   'CLLoss: {:.3f}\t'
                    #   'NLLoss: {:.3f}\t'
                    'GANLoss: G:{:.3f} D:{:.3f}\n'
                    #   'GANLoss: G:{:.3f}\t'
                    # 'FRECLoss: {:.3f}\n'
                    #   'Loss GM {:.3f}\n'
                    .format(epoch, i + 1, len(data_loader),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg,
                            # losses.val, losses.avg,
                            #   loss_cl.item(),
                            #   loss_neg.item(),
                            gan_losses['G'], gan_losses['D'],
                            #   gan_losses['G'],
                            # frec_loss.item(),
                            #   gm_loss.item()
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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