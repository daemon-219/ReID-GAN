import argparse
import os
import os.path as osp
import torch
import dual_gan.models as gan_models
from clustercontrast import models
from clustercontrast import datasets

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.DPTN_group = self.parser.add_argument_group(title='DPTN GAN options')
        self.CC_group = self.parser.add_argument_group(title='CC ReID options')
        self.GM_group = self.parser.add_argument_group(title='Gradient Matching options')

        self.parser.add_argument('--name', type=str, default='DPTN_CC_market', help='name of the experiment. It decides where to store samples and models')        
        # self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.DPTN_group.add_argument('--checkpoints_dir', type=str, default='./examples/gan_logs/DPTN', help='models are saved here')
        self.DPTN_group.add_argument('--model', type=str, default='DPTN', help='which model to use')
        self.DPTN_group.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')        
        self.DPTN_group.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.DPTN_group.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
        self.DPTN_group.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        self.DPTN_group.add_argument('--fp16', action='store_true', default=False, help='train with AMP')
        # self.parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

        # input/output sizes
        self.DPTN_group.add_argument('--image_nc', type=int, default=3)
        self.DPTN_group.add_argument('--pose_nc', type=int, default=18)
        # self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.DPTN_group.add_argument('--old_size', type=int, default=(128, 64), help='Scale images to this size. The final image will be cropped to --crop_size.')
        self.DPTN_group.add_argument('--loadSize', type=int, default=128, help='scale images to this size')
        # self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        # self.parser.add_argument('--label_nc', type=int, default=35, help='# of input label channels')
        # self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        # self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        # self.parser.add_argument('--dataset_mode', type=str, default='fashion')
        # self.parser.add_argument('--dataroot', type=str, default='/media/data2/zhangpz/DataSet/Fashion')
        # self.parser.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        # self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')        
        # self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation') 
        # self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')                
        # self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # for displays
        self.DPTN_group.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.DPTN_group.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
        self.DPTN_group.add_argument('--display_id', type=int, default=0, help='display id of the web')  # 1
        self.DPTN_group.add_argument('--display_port', type=int, default=8096, help='visidom port of the web display')
        self.DPTN_group.add_argument('--display_single_pane_ncols', type=int, default=0,
                            help='if positive, display all images in a single visidom web panel')
        self.DPTN_group.add_argument('--display_env', type=str, default=self.parser.parse_known_args()[0].name.replace('_', ''),
                            help='the environment of visidom display')
        # for instance-wise features
        self.DPTN_group.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')        
        self.DPTN_group.add_argument('--instance_feat', action='store_true', help='if specified, add encoded instance features as input')
        self.DPTN_group.add_argument('--label_feat', action='store_true', help='if specified, add encoded label features as input')        
        self.DPTN_group.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')        
        self.DPTN_group.add_argument('--load_features', action='store_true', help='if specified, load precomputed feature maps')
        self.DPTN_group.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder') 
        self.DPTN_group.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')        
        self.DPTN_group.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')

        '''reid args'''
        # data
        self.parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                            choices=datasets.names())
        self.parser.add_argument('-b', '--batch-size', type=int, default=2)
        self.parser.add_argument('-j', '--workers', type=int, default=4)
        self.CC_group.add_argument('--height', type=int, default=256, help="input height")
        self.CC_group.add_argument('--width', type=int, default=128, help="input width")
        self.CC_group.add_argument('--num-instances', type=int, default=4,
                            help="each minibatch consist of "
                                "(batch_size // num_instances) identities, and "
                                "each identity has num_instances instances, "
                                "default: 0 (NOT USE)")
        # cluster
        self.CC_group.add_argument('--eps', type=float, default=0.5,
                            help="max neighbor distance for DBSCAN")
        self.CC_group.add_argument('--eps-gap', type=float, default=0.02,
                            help="multi-scale criterion for measuring cluster reliability")
        self.CC_group.add_argument('--k1', type=int, default=15,
                            help="hyperparameter for KNN")
        self.CC_group.add_argument('--k2', type=int, default=4,
                            help="hyperparameter for outline")
        # model
        self.CC_group.add_argument('-a', '--arch', type=str, default='resnet50',
                            choices=models.names())
        self.CC_group.add_argument('--features', type=int, default=0)
        self.CC_group.add_argument('--dropout', type=float, default=0)
        self.CC_group.add_argument('--momentum', type=float, default=0.2,
                            help="update momentum for the hybrid memory")
        
        # path
        working_dir = './examples'
        self.parser.add_argument('--data-dir', type=str, metavar='PATH',
                            default=osp.join(working_dir, 'data'))
        self.parser.add_argument('--logs-dir', type=str, metavar='PATH',
                            default=osp.join(working_dir, 'gan_logs'))
        self.CC_group.add_argument('--pooling-type', type=str, default='gem')
        self.CC_group.add_argument('--use-hard', action="store_true")
        self.parser.add_argument('--no-cam', action="store_true")
        
        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        opt, _ = self.parser.parse_known_args()
        # modify the options for different models
        model_option_set = gan_models.get_option_setter(opt.model)
        self.parser = model_option_set(self.parser, self.isTrain)

        # data_option_set = data.get_option_setter(opt.dataset_mode)
        # self.parser = data_option_set(self.parser, self.isTrain)

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        if torch.cuda.is_available():
            self.opt.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
        else:
            self.opt.device = torch.device("cpu")

        # str_ids = self.opt.gpu_ids.split(',')
        # self.opt.gpu_ids = []
        # for str_id in str_ids:
        #     id = int(str_id)
        #     if id >= 0:
        #         self.opt.gpu_ids.append(id)
        
        # # set gpu ids
        # if len(self.opt.gpu_ids) > 0:
        #     torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = osp.join(self.opt.checkpoints_dir, self.opt.name)
        if not osp.exists(expr_dir):
            os.makedirs(expr_dir)
        if save and not (self.isTrain and self.opt.continue_train):
            name = 'train' if self.isTrain else 'test'
            file_name = osp.join(expr_dir, name+'_opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
