from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch
import torch.nn as nn
from clustercontrast.utils.data.diff_augs import my_resize


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
    def __init__(self, encoder, GAN=None, writer=None, memory=None, T=1.0):
        super(ClusterContrastWithGANTrainer, self).__init__()
        self.encoder = encoder
        if GAN is None:
            raise('GAN not implemented!')
        self.gan = GAN
        self.memory = memory
        self.T = T
        self.writer = writer

    def train_all(self, epoch, data_loader, optimizer, dis_metric='ours', print_freq=10, train_iters=400, acc_iters=0):
        print("train both gan and reid")
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            # print(len(inputs))
            data_time.update(time.time() - end)

            # process inputs
            reid_inputs, labels, indexes = self._parse_data(inputs[0])

            # forward
            f_out = self._forward(reid_inputs)
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)
            loss_ori = self.memory(f_out, labels)

            self.gan.set_input(inputs[1])
            fake_image_t, fake_image_s = self.gan.synthesize()
            """
            TODO: do transform here
            """
            fake_image_t = my_resize(fake_image_t, (reid_inputs.shape[2], reid_inputs.shape[3]))
            # fake_image_s = my_resize(fake_image_s, (reid_inputs.shape[2], reid_inputs.shape[3]))
            
            """
            # do gradient matching here

            # TODO: Gradient Matching Loss is too large!!!

            net_parameters = list(params for params in self.encoder.parameters() if params.requires_grad)
            gw_real = torch.autograd.grad(loss, net_parameters, retain_graph=True)
            gw_real = list((_.detach().clone() for _ in gw_real))

            f_syn_out = self._forward(fake_image_s)
            # do not update cluster memory for synthesized inputs
            loss_syn = self.memory(f_syn_out, labels, update=False)
            gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

            gm_loss = match_loss(gw_syn, gw_real, dis_metric)

            self.gan.optimize_parameters_generated(gm_loss)
            """

            # TODO: do id-level contrastive learning
            f_tar = self._forward(fake_image_t)
            q = self.encoder.module.predictor(f_tar) 
            k = f_out.detach().clone()
            q = torch.mean(torch.stack(torch.tensor_split(q, 4, dim=0), dim=0), dim=1)
            k = torch.mean(torch.stack(torch.tensor_split(k, 4, dim=0), dim=0), dim=1)
            loss_cl = self.contrastive_loss(q, k)  

            # all backward for gan
            # self.gan.optimize_parameters_generated()
            # self.gan.optimize_parameters_generated()
            # self.gan.optimize_parameters()         
            
            # TODO: fake target as extended inputs
            # ex_input = fake_image_t.detach().clone()
            # f_ex_out = self._forward(ex_input)
            # loss_ex = self.memory(f_tar, labels)
            
            """
            Finished
            """
            loss = loss_ori + loss_cl

            # GAN dis opt
            self.gan.optimizer_D.zero_grad()
            self.gan.backward_D()
            self.gan.optimizer_D.step()

            # ReID and GAN gen opt
            self.gan.optimizer_G.zero_grad()
            optimizer.zero_grad()
        
            self.gan.backward_G(retain_graph=True)
            loss.backward()
            
            self.gan.optimizer_G.step()
            optimizer.step()

            losses.update(loss.item())

            # add writer
            if self.writer is not None:
                # gan model
                total_steps = acc_iters + i
                gan_losses = self.gan.get_current_errors()
                self.writer.add_scalar('Loss/app_gen_s', gan_losses['app_gen_s'], total_steps)
                self.writer.add_scalar('Loss/content_gen_s', gan_losses['content_gen_s'], total_steps)
                self.writer.add_scalar('Loss/style_gen_s', gan_losses['style_gen_s'], total_steps)
                self.writer.add_scalar('Loss/app_gen_t', gan_losses['app_gen_t'], total_steps)
                self.writer.add_scalar('Loss/ad_gen_t', gan_losses['ad_gen_t'], total_steps)
                self.writer.add_scalar('Loss/dis_img_gen_t', gan_losses['dis_img_gen_t'], total_steps)
                self.writer.add_scalar('Loss/content_gen_t', gan_losses['content_gen_t'], total_steps)
                self.writer.add_scalar('Loss/style_gen_t', gan_losses['style_gen_t'], total_steps)
                # reid model
                self.writer.add_scalar('Loss/reid_loss', losses.val, total_steps)
                # gradient matching
                self.writer.add_scalar('Loss/cl_loss', loss_cl.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'CLLoss: {:.3f}\t '
                      'GANLoss: {:.3f}\n '
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              loss_cl.item(),
                              sum(gan_losses.values()) / len(gan_losses)))

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, acc_iters=0):
        """
        TODO: GAN in test mode:
        """

        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            # print(len(inputs))
            data_time.update(time.time() - end)

            # process inputs
            reid_inputs, labels, indexes = self._parse_data(inputs[0])

            self.gan.set_input(inputs[1])
            fake_image_t, _ = self.gan.synthesize()
            """
            TODO: do transform here
            """
            ex_inputs = my_resize(fake_image_t, (reid_inputs.shape[2], reid_inputs.shape[3]))

            # fake target as extended inputs
            new_inputs = torch.cat([reid_inputs, ex_inputs], dim=0)
            labels = labels.repeat(2)

            # forward
            f_out = self._forward(new_inputs)
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
                # gan model
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
                      'Loss {:.3f} ({:.3f})\n'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)
    
    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # Einstein sum is more intuitive
        logits = torch.einsum('n c, m c -> n m', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = torch.arange(N, dtype=torch.long).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)


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


def match_loss(gw_syn, gw_real, dis_metric):
    """
    TODO: try contrastive loss
    """
    dis = torch.tensor(0.0).cuda()

    if dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

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

    else:
        raise('unknown distance function: %s'%dis_metric)

    return dis