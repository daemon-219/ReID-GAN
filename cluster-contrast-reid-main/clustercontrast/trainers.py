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
    def __init__(self, encoder, GAN=None, writer=None, memory=None, opt=None):
        super(ClusterContrastWithGANTrainer, self).__init__()
        self.encoder = encoder
        if GAN is None:
            raise('GAN not implemented!')
        self.gan = GAN
        self.memory = memory
        self.writer = writer
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
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
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            # print(len(inputs))
            data_time.update(time.time() - end)

            # process inputs
            reid_inputs, labels, indexes = self._parse_data(inputs[0])

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
            self.gan.set_input(inputs[1])
            fake_image_t, fake_image_s = self.gan.synthesize(is_tain=True)

            # (batch, feature_dim) = (b, 2048)
            f_out = self._forward(reid_inputs)
            loss = self.memory(f_out, labels)

            net_parameters = list(params for params in self.encoder.parameters() if params.requires_grad)
            gw_real = torch.autograd.grad(loss, net_parameters, retain_graph=True)
            gw_real = list((_.detach().clone() for _ in gw_real))

            f_syn_out = self._forward(fake_image_s)
            # do not update cluster memory for synthesized inputs
            loss_syn = self.memory(f_syn_out, labels, update=False)
            gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

            gm_loss = self.opt.lambda_nl * match_loss(gw_syn, gw_real)
            
            # backward for adaptor
            self.gan.optimize_parameters_adaptor(gm_loss)

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
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss GM {:.3f}\n'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              gm_loss.item()
                              ))
                
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
            fake_image_n = self.gan.synthesize().detach().clone()

            """
            TODO: do transform here
            """
            postive_pair = my_resize(self.gan.target_image, (reid_inputs.shape[2], reid_inputs.shape[3]))
            fake_image_n = my_resize(fake_image_n, (reid_inputs.shape[2], reid_inputs.shape[3]))
            f_tar_p, f_tar_n = self._forward(postive_pair), self._forward(fake_image_n)
            
            # bp gan loss here
            loss_neg = self.memory(f_tar_n, labels, update=False)

            # TODO: Gradient Matching Loss is too large!!!
            """
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
            f_tar = torch.cat([f_tar_p, f_tar_n], dim=0)
            # q = self.encoder.module.predictor(f_tar) 
            # k = f_out.detach().clone()
            # q = torch.mean(torch.stack(torch.tensor_split(q, 4, dim=0), dim=0), dim=1)
            # k = torch.mean(torch.stack(torch.tensor_split(k, 4, dim=0), dim=0), dim=1)
            # loss_cl = self.contrastive_loss(q, k)  
            loss_cl = self.cl_loss(f_out, self.encoder.module.predictor(f_tar.detach().clone()))  

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

            # ReID and GAN opt
            self.gan.optimize_generated(loss_neg)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

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
                self.writer.add_scalar('Loss/reid_loss', losses.val, total_steps)
                # neg loss bp into gan
                self.writer.add_scalar('Loss/nl_loss', loss_neg.item())
                # cl loss from hard negative samples
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
                      'NLLoss: {:.3f}\t '
                      'GANLoss: G:{:.3f} D:{:.3f}\n '
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              loss_cl.item(),
                              loss_neg.item(),
                              gan_losses['G'], gan_losses['D']))

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
        return nn.CrossEntropyLoss()(logits, labels) * 2 * self.T    
    
    def cl_loss(self, q, k, group_size=16):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # Einstein sum is more intuitive
        logits = torch.einsum('n c, m c -> n m', [q, k]) / self.T
        # group sum 
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


def match_loss(gw_syn, gw_real, dis_metric='ours'):
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