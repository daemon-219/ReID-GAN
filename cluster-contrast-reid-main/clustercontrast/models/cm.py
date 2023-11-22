import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd


class CM(autograd.Function):
    
    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    # @torch.cuda.amp.autocast()
    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

class CM_Conf(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, conf_mask, momentum):
        ctx.features = features
        ctx.conf_mask = conf_mask
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets) 
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y, cfw in zip(inputs, targets, ctx.conf_mask.tolist()):
            # if cfw < 0.0625:
            #     continue            
            if cfw < 1.:
                continue
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()


        # batch_centers = collections.defaultdict(list)

        # for instance_feature, conf, index in zip(inputs, ctx.conf_mask.tolist(), targets.tolist()):
        #     batch_centers[index].append(conf * instance_feature)

        # for index, features in batch_centers.items():
        #     # epo update 
        #     ''' 
        #     rew_feature = F.normalize(torch.mean(torch.stack(features, dim=0), dim=0), dim=0)
        #     '''

        #     # iter update
        #     rew_feature = torch.sum(torch.stack(features, dim=0), dim=0)
        #     # rew_feature = F.normalize(torch.sum(torch.stack(features, dim=0), dim=0), dim=0)

        #     ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * rew_feature
        #     ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None, None


def cm_conf(inputs, indexes, features, conf_mask, momentum=0.5):
    return CM_Conf.apply(inputs, indexes, features, conf_mask, torch.Tensor([momentum]).to(inputs.device))


# class ClusterMemory(nn.Module, ABC):
#     def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False, use_conf=False):
#         super(ClusterMemory, self).__init__()
#         self.num_features = num_features
#         self.num_samples = num_samples

#         self.momentum = momentum
#         self.temp = temp
#         self.use_hard = use_hard
#         self.use_conf = use_conf

#         self.register_buffer('features', torch.zeros(num_samples, num_features))

#     # @torch.cuda.amp.autocast()
#     def forward(self, inputs, targets, update=True, ex_f=None, conf_mask=None, focus_hard=False):

#         # gather    
#         inputs = F.normalize(inputs, dim=1).cuda()
#         if not update:
#             outputs = torch.mm(inputs, self.features.t())
#         else:
#             if self.use_hard:
#                 outputs = cm_hard(inputs, targets, self.features, self.momentum)
#             elif self.use_conf:
#                 outputs = cm_conf(inputs, targets, self.features, conf_mask, self.momentum)
#             else:
#                 outputs = cm(inputs, targets, self.features, self.momentum)
#         # print(self.features.shape)
#         if ex_f is not None: 
#             ex_f = F.normalize(ex_f, dim=1).cuda()
#             # t extend samples, outputs_ex:(n, t)
#             outputs_ex = torch.mm(inputs, ex_f.t())

#             # n_ids = torch.unique(targets).shape[0]
#             # group_size = targets.shape[0] // n_ids

#             # # mask the same id
#             # outputs_ex += (-10000.0 * torch.eye(n_ids)).repeat_interleave(group_size, dim=0).cuda()
            
#             outputs = torch.cat([outputs, outputs_ex], dim=1)
#             # outputs_ex = torch.mm(ex_f, self.features.t())
#             # pred = torch.argmax(F.softmax(outputs_ex, dim=1), dim=1)
#             # print(pred)
#             # print('label:', targets)

#             # outputs_ex /= self.temp
#             # fc_labels = torch.repeat_interleave(torch.arange(ex_f.shape[0], dtype=torch.long), group_size, dim=0).cuda()
#             # loss_fc = F.cross_entropy(outputs_ex, fc_labels)

#         # cl = (q * k) / (q * c) + (q * ex)
#         # cl = (q * k) / ((q * ex))
#         outputs /= self.temp
#         # print(outputs.shape)

#         if focus_hard:
#             """
#             TODO: block those loss with low confidence 
#             """
#             n_ids = torch.unique(targets).shape[0]
#             group_size = targets.shape[0] // n_ids

#             # # set the loss of the last k conf in each group to 0
#             # _, diff_ids = torch.topk(conf_mask.reshape(n_ids, group_size), k=(group_size // 4), dim=-1, largest=False)
#             # loss = F.cross_entropy(outputs, targets, reduction="none").reshape(n_ids, group_size)
#             # # print("original loss:", loss[:group_size])
#             # loss = loss.scatter(-1, diff_ids, 0.).mean()  
#             # # print("blocked loss:", loss[:group_size])

#             """
#             TODO: foucus on those loss with low confidence 
#             """
#             loss = F.cross_entropy(outputs, targets, reduction="none")
#             loss[conf_mask<(1./group_size)] *= 2
#             loss = loss.mean()

#         else:
#             # loss = F.cross_entropy(outputs, targets)

#             # reweighted 

#             loss = F.cross_entropy(outputs, targets, reduction="none")
#             # # # iter update
#             # # '''
#             # # # print(loss, loss.mean())
#             # n_ids = torch.unique(targets).shape[0]
#             # group_size = targets.shape[0] // n_ids

#             # # print(loss[:group_size])

#             # loss *= conf_mask * group_size

#             # barrier = (conf_mask * group_size) < 1.
#             # loss[barrier] *= conf_mask[barrier] * group_size
#             # # print(loss, loss.mean())
#             # loss = loss.mean()
#             # # loss = (conf_mask * loss).sum() / torch.unique(targets).shape[0]
#             # '''

#             # # epo update 
#             # # print(loss.mean())
#             # loss *= conf_mask

#             # loss = loss.mean()

#         return loss
    
    
class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False, use_conf=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # self.register_buffer('features1', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets, ex_f=None):
 
        inputs = F.normalize(inputs, dim=1).cuda()

        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)

        if ex_f is not None: 
            ex_f = F.normalize(ex_f, dim=1).cuda()
            outputs_ex = torch.mm(inputs, ex_f.t())

            n_ids = torch.unique(targets).shape[0]
            group_size = targets.shape[0] // n_ids
            
            outputs_ex += (-10000.0 * torch.eye(n_ids)).repeat_interleave(group_size, dim=0).cuda()
            outputs = torch.cat([outputs, outputs_ex], dim=1)

        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets, reduction="none")

        return loss


class ClusterMemory_Gradient(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05):
        super(ClusterMemory_Gradient, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.temp = temp

        # cluters
        self.normed_clusters = None

    def set_clusters(self, clusters, cluster_lr):
        self.trainable_clusters = clusters.detach().clone().requires_grad_(True)
        self.optimizer_cluster = torch.optim.SGD([self.trainable_clusters], lr=cluster_lr)
        self.normed_clusters = F.normalize(self.trainable_clusters)
        # print(self.trainable_clusters.is_leaf)

    # @torch.cuda.amp.autocast()
    def forward(self, inputs, targets, ex_f=None):

        # gather    
        inputs = F.normalize(inputs, dim=1).cuda()

        outputs = torch.mm(inputs, self.normed_clusters.detach().clone().t())
        
        if ex_f is not None: 
            ex_f = F.normalize(ex_f, dim=1).cuda()
            # t extend samples, outputs_ex:(n, t)
            outputs_ex = torch.mm(inputs, ex_f.t())
            # n == t
            # outputs_ex += (-10000.0 * torch.eye(ex_f.shape[0])).cuda()
            # n = t * group_size
            group_size = outputs_ex.shape[0] // outputs_ex.shape[1]
            # print(group_size)
            outputs_ex += (-10000.0 * torch.eye(ex_f.shape[0])).repeat_interleave(group_size, dim=0).cuda()
            # outputs_ex[::group_size] += (-10000.0 * torch.eye(ex_f.shape[0])).cuda()
            # outputs:(n, m+t)
            outputs = torch.cat([outputs, outputs_ex], dim=1)

        outputs /= self.temp
        # print(outputs.shape)
        loss = F.cross_entropy(outputs, targets)
        return loss
    
    def update_clusters(self, p_ids, eps=1e-16):
        # called for optimizing trainable clusters after GAN Loss G backward
        # gradient clip for stable update
        # nn.utils.clip_grad_norm_(parameters=self.trainable_clusters, max_norm=1, norm_type=2)
        for p_id in p_ids:
            self.trainable_clusters.grad[p_id] /= self.trainable_clusters.grad[p_id].norm() + eps

        self.optimizer_cluster.step()
        self.optimizer_cluster.zero_grad()
        self.normed_clusters = F.normalize(self.trainable_clusters)

