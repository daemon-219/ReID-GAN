# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp
import torch.multiprocessing as mp
import random
import numpy as np
import sys
import copy
import collections
import time
from datetime import timedelta
from collections import OrderedDict

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torch.utils.tensorboard import SummaryWriter

# import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp

# GAN model
# from fdgan.options import Options
# from fdgan.utils.visualizer import Visualizer
# from fdgan.model import FDGANModel

from examples.options.train_options import TrainOptions
from dual_gan.models.models import create_model as create_gan
from dual_gan.gan_visualizer import Visualizer

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.models.cm import ClusterMemory, ClusterMemory_Gradient
from clustercontrast.trainers import ClusterContrastTrainer, ClusterContrastWithGANTrainer
from clustercontrast.evaluators import Evaluator, extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
from clustercontrast.utils.data.preprocessor import Preprocessor
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.infomap_cluster import get_dist_nbr, cluster_by_infomap

import wandb

start_epoch = best_mAP = 0


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(opt, dataset, height, width, batch_size, workers,
                     num_instances, iters, with_gan=False, trainset=None, no_cam=False, rank=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    
    GAN_transform = None
    if with_gan:
        # basic size transform, put data augmentations in trainer 
        # (height, weight): (256, 128)
        # train_transformer = T.Compose([
        #     T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        #     T.ToTensor(),
        #     normalizer,
        # ])

        # prepare for transformation
        # opt.loadSize: (128, 64)

        # DPTN
        # GAN_transform = T.Compose([
        #     T.ToTensor(),
        #     T.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        # ])        
        
        # DEC AE
        # GAN_transform = T.Compose([
        #     # T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        #     # T.RandomHorizontalFlip(p=0.5),
        #     T.ToTensor(),
        #     normalizer
        # ])        
        
        GAN_transform = T.Compose([
            # T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
            # T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # normalizer
        ])
        
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        # T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])    

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
            
    else:
        sampler = None
    
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                pose_file=dataset.train_pose_dir if hasattr(dataset, 'train_pose_dir') else None, 
                                with_gan=with_gan, load_size=opt.loadSize,
                                transform=train_transformer, GAN_transform=GAN_transform),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, with_gan=False, testset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir,
                     pose_file=dataset.test_pose_dir if hasattr(dataset, 'test_pose_dir') else None, 
                     with_gan=with_gan, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def get_gan_loader(opt, dataset, height, width, batch_size, workers, only_gan=True, testset=None):

    GAN_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, pose_file=dataset.train_pose_dir, 
                     only_gan=only_gan, load_size=opt.loadSize, GAN_transform=GAN_transform),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model(opt):
    if opt.arch == "resnet_mp50":
        model = models.create(opt.arch, num_features=opt.features, norm=True, dropout=opt.dropout,
                          num_proj=256, pooling_type=opt.pooling_type, need_predictor=opt.cl_loss)
    else:
        model = models.create(opt.arch, num_features=opt.features, norm=True, dropout=opt.dropout,
                          num_classes=0, pooling_type=opt.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main():
    opt = TrainOptions().parse()

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        cudnn.deterministic = True

    if opt.batch_size % opt.num_instances:
        raise("batch_size should be divided exactly by num_instance for each minibatch!")

    wandb.init(config=opt,
               project="reid",
            #    entity=your_team_name,
            #    notes=socket.gethostname(),
               name=opt.name,
            #    dir=run_dir,
               job_type="training")

    main_worker(opt)


def main_worker(opt):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(opt.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(opt))

    # Create datasets
    iters = opt.iters if (opt.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(opt.dataset, opt.data_dir)
    test_loader = get_test_loader(dataset, opt.height, opt.width, opt.batch_size, opt.workers)
    
    # Create model
    GAN_model = None
    visualizer = None
    if opt.with_gan:
        # '''fdgan'''
        # opt = Options().parse()
        # GAN_model = FDGANModel(opt)
        # visualizer = Visualizer(opt)

        '''GAN_model'''
        iter_path = osp.join(opt.checkpoints_dir, opt.name, 'iter.txt')
        if opt.continue_train:
            try:
                restart_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
            except:
                restart_epoch, epoch_iter = 1, 0
            print('Resuming from epoch %d at iteration %d' % (restart_epoch, epoch_iter))        
        else:    
            restart_epoch, epoch_iter = 1, 0

        opt.iter_start = restart_epoch
    
        if opt.debug:
            opt.display_freq = 1
            opt.print_freq = 1
            opt.niter = 1
            opt.niter_decay = 0
            opt.max_dataset_size = 10

        GAN_model = create_gan(opt)
        visualizer = Visualizer(opt)

    '''reid model'''
    ReID_model = create_model(opt)

    # Evaluator
    evaluator = Evaluator(ReID_model)

    # writer
    writer = SummaryWriter(comment=opt.name)

    # Optimizer
    params = [{"params": [value]} for _, value in ReID_model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=opt.reid_lr, weight_decay=opt.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # Trainer
    if opt.with_gan:
        trainer = ClusterContrastWithGANTrainer(encoder=ReID_model, GAN=GAN_model, writer=writer, opt=opt)
    else: 
        trainer = ClusterContrastTrainer(ReID_model)

    acc_iters = 0

    if opt.continue_train:
        model_path = osp.join(opt.reid_pretrain, "checkpoint.pth.tar")
        print('Loading ReID model from epoch %s' % (model_path))
        ReID_model.load_state_dict(torch.load(model_path)['state_dict'], strict=False)
        

    for epoch in range(opt.epochs):
        # release memory 
        torch.cuda.empty_cache()

        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = get_test_loader(dataset, opt.height, opt.width,
                                             opt.batch_size, opt.workers, testset=sorted(dataset.train))

            features, _ = extract_features(ReID_model, cluster_loader, print_freq=50)
            # all_features, _ = extract_features(ReID_model, cluster_loader, print_freq=50, clustering=True)
            # cluster by features, with multiple groups of clusters
            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            # features = torch.cat([all_features[f][0].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            # features1 = torch.cat([features[f][1].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)

            features_array = F.normalize(features, dim=1).cpu().numpy()
            feat_dists, feat_nbrs = get_dist_nbr(features=features_array, k=opt.k1, knn_method='faiss-gpu')
            del features_array

            s = time.time()
            pseudo_labels = cluster_by_infomap(feat_nbrs, feat_dists, min_sim=opt.eps, cluster_num=opt.k2)
            pseudo_labels = pseudo_labels.astype(np.intp)

            print('cluster cost time: {}'.format(time.time() - s))
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)

            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features = generate_cluster_features(pseudo_labels, features)
        # cluster_features1 = generate_cluster_features(pseudo_labels, features1)

        del cluster_loader, features
        # del cluster_loader, all_features, features, features1

        # Create hybrid memory
        if opt.learnable_memory:
            memory = ClusterMemory_Gradient(ReID_model.module.num_features, num_cluster, temp=opt.temp).cuda()
            cluster_features = F.normalize(cluster_features, dim=1).cuda()
            # cluster_lr = opt.cluster_lr * (0.95 ** epoch)
            # wandb.log({
            #     "cluster_lr": cluster_lr})
            memory.set_clusters(cluster_features, opt.cluster_lr)
        else:
            memory = ClusterMemory(ReID_model.module.num_features, num_cluster, temp=opt.temp,
                               momentum=opt.momentum, use_hard=opt.use_hard, use_conf=opt.use_conf).cuda()
            memory.features = F.normalize(cluster_features, dim=1).cuda()
            # memory.features1 = F.normalize(cluster_features1, dim=1).cuda()
        
        trainer.memory = memory
        if opt.bipath:
            trainer.memoryb = copy.deepcopy(memory)
        pseudo_labeled_dataset = []

        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), cid))

        print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

        @torch.no_grad()
        def get_conf_weight(clusters):
            gan_loader = get_gan_loader(opt, dataset, opt.height, opt.width,
                                             opt.batch_size, opt.workers, testset=pseudo_labeled_dataset)
            s = time.time()

            conf_weight = torch.ones(len(pseudo_labeled_dataset)).cuda()
            print(conf_weight.shape)
            # labels = OrderedDict()
            
            for i, (gan_inputs, pid, index) in enumerate(gan_loader):
                # print(index)
                GAN_model.set_input(gan_inputs)
                GAN_model.synthesize_p(clusters[pid])
                loss_rec = GAN_model.get_L1_loss()
                # print(loss_rec.shape)
                # loss_rec = GAN_model.get_loss_G(need_cm=False)
                # print(loss_rec)
                # conf_weight[index] = torch.cos(loss_rec / 2)
                # conf_weight[index] = 1 - torch.pow(loss_rec, 2)
                # conf_weight[index] = 2 * torch.pow(1 - loss_rec, 2)

                # 2e^{-2x}
                # conf_weight[index] = 2 * torch.exp(-2 * loss_rec)
                # 1e^{-x^{2}}
                # conf_weight[index] = torch.exp(-torch.pow(loss_rec, 2))
                conf_weight[index] = torch.exp(-4 * torch.pow(loss_rec, 4))
                # print(conf_weight[index])

            print('calculate confidence weight cost time: {}'.format(time.time() - s))

            del gan_loader
            
            return conf_weight

        conf_weight = None
        # conf_weight = get_conf_weight(memory.features)

        # train_loader = get_train_loader(opt, dataset, opt.height, opt.width,
        #                                 opt.batch_size, opt.workers, opt.num_instances, iters,
        #                                 with_gan=opt.gan_train, trainset=pseudo_labeled_dataset, no_cam=opt.no_cam)        
        train_loader = get_train_loader(opt, dataset, opt.height, opt.width,
                                        opt.batch_size, opt.workers, opt.num_instances, iters,
                                        with_gan=opt.with_gan, trainset=pseudo_labeled_dataset, no_cam=opt.no_cam)

        train_loader.new_epoch()

        """
        TODO: check data preprocessor, trainer
        """
        if (epoch + 1) > opt.warmup_epo: 
            if opt.gan_train:
                if opt.bipath:
                    trainer.train_all_bip(epoch, train_loader, optimizer, print_freq=opt.print_freq, 
                            train_iters=len(train_loader), acc_iters=acc_iters, conf_weight=conf_weight)
                elif opt.learnable_memory:
                    trainer.train_all_with_memoery(epoch, train_loader, optimizer, print_freq=opt.print_freq, 
                            train_iters=len(train_loader), acc_iters=acc_iters)
                else:
                    trainer.train_all(epoch, train_loader, optimizer, print_freq=opt.print_freq, 
                            train_iters=len(train_loader), acc_iters=acc_iters, conf_weight=conf_weight)             
            else:
                trainer.train(epoch, train_loader, optimizer, print_freq=opt.print_freq, 
                            train_iters=len(train_loader), acc_iters=acc_iters)
        else:
            trainer.train_reid(epoch, train_loader, optimizer, print_freq=opt.print_freq, 
                            train_iters=len(train_loader), acc_iters=acc_iters)        
        
        acc_iters += len(train_loader)

        if (epoch + 1) % opt.eval_step == 0 or (epoch == opt.epochs - 1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            wandb.log({
                "mAP": mAP,
                "best_mAP": best_mAP,
                "reis_lr": optimizer.state_dict()['param_groups'][0]['lr']})

            save_checkpoint({
                'state_dict': ReID_model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(opt.logs_dir, 'checkpoint.pth.tar'))
            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))
        
        if (epoch + 1) > opt.warmup_epo: 
            if opt.with_gan: 
                if opt.gan_train:
                    GAN_model.save_networks('latest')
                    # GAN_model.save_networks(epoch)
                    # np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
                    lr_G, lr_D = GAN_model.get_current_learning_rate()
                    print('Epoch: [{}]\t'
                                'LR/reid {:.7f}\t'
                                'LR/G: {:.7f}\t'
                                'LR/D: {:.7f}\n'
                                .format(epoch,
                                        optimizer.state_dict()['param_groups'][0]['lr'],
                                        lr_G,
                                        lr_D))
                    wandb.log({
                        "G_lr": lr_G,
                        "D_lr": lr_D})
                    GAN_model.update_learning_rate()
                
                if (epoch + 1) % opt.vis_step == 0 or (epoch == opt.epochs - 1):
                    # visualize gan results 
                    # GAN_model.visual_names = ['source_image', 'target_image', 'fake_image', 'fake_image_n', 'mixed_image']
                    # GAN_model.visual_names = ['source_image', 'fake_image', 'mixed_image']
                    GAN_model.visual_names = ['source_image', 'fake_image']
                    # GAN_model.visual_names = []
                    visualizer.display_current_results(GAN_model.get_current_visuals(), epoch)
                    if hasattr(GAN_model, 'distribution'):
                        visualizer.plot_current_distribution(GAN_model.get_current_dis())

        lr_scheduler.step()
        

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(opt.logs_dir, 'model_best.pth.tar'))
    ReID_model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    main()
