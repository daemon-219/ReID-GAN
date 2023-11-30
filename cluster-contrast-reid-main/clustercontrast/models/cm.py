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

class CM_gan(autograd.Function):
    
    @staticmethod
    def forward(ctx, inputs, gan_inputs, targets, features, gan_features, momentum):
        ctx.features = features
        ctx.gan_features = gan_features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, gan_inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    # @torch.cuda.amp.autocast()
    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, gan_inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, gan_x, y in zip(inputs, gan_inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()            
            ctx.gan_features[y] = ctx.momentum * ctx.gan_features[y] + (1. - ctx.momentum) * gan_x
            ctx.gan_features[y] = F.normalize(ctx.gan_features[y], dim=0)

        return grad_inputs, None, None, None, None, None

def cm_gan(inputs, gan_inputs, indexes, features, gan_features, momentum=0.5):
    return CM_gan.apply(inputs, gan_inputs, indexes, features, gan_features, torch.Tensor([momentum]).to(inputs.device))
    
class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False, use_conf=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('gan_features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets, gan_inputs=None, conf_weight=None):
 
        inputs = F.normalize(inputs, dim=1).cuda()

        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)
            # outputs = cm_gan(inputs, gan_inputs, targets, self.features, self.gan_features, self.momentum)
            # outputs = cm_conf(inputs, targets, self.features, conf_weight, self.momentum)

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