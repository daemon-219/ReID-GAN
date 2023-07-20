from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch


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
    def __init__(self, encoder, GAN=None, writer=None, memory=None):
        super(ClusterContrastWithGANTrainer, self).__init__()
        self.encoder = encoder
        if GAN is None:
            raise('GAN not implemented!')
        self.gan = GAN
        self.memory = memory
        self.writer = writer

    def train(self, epoch, data_loader, optimizer, dis_metric='ours', print_freq=10, train_iters=400, acc_iters=0):
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
            loss = self.memory(f_out, labels)

            """
            TODO: add gan loss and optimize step here 
            """

            self.gan.set_input(inputs[1])
            fake_image_t, fake_image_s = self.gan.synthesize()
            """
            TODO: do transform here!!!
            """

            # # do gradient matching here
            # net_parameters = list(params for params in self.encoder.parameters() if params.requires_grad)
            # gw_real = torch.autograd.grad(loss, net_parameters, retain_graph=True)
            # gw_real = list((_.detach().clone() for _ in gw_real))

            # f_syn_out = self._forward(fake_image_s)
            # # do not update cluster memory for synthesized inputs
            # loss_syn = self.memory(f_syn_out, labels, update=False)
            # gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

            # gm_loss = match_loss(gw_syn, gw_real, dis_metric)

            # self.gan.optimize_parameters_generated(gm_loss)

            self.gan.optimize_parameters_generated()
            # self.gan.optimize_parameters()
            
            # fake target as extended inputs
            ex_input = fake_image_t.detach().clone()
            f_ex_out = self._forward(ex_input)
            loss += self.memory(f_ex_out, labels)

            """
            Finished
            """

            optimizer.zero_grad()
            loss.backward()
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
                # # gradient matching
                # self.writer.add_scalar('Loss/gm_loss', gm_loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                    #   'GMLoss: {:.3f}\t '
                      'GANLoss: {:.3f}\n '
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                            #   gm_loss.item(),
                              sum(gan_losses.values()) / len(gan_losses)))

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