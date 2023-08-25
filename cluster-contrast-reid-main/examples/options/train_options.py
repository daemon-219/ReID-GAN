# from .base_options import BaseOptions
from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        # self.GAN_group.add_argument('--display_freq', type=int, default=200, help='frequency of showing training results on screen')
        # self.GAN_group.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
        # self.GAN_group.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        # self.GAN_group.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        self.GAN_group.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.GAN_group.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.CC_group.add_argument('--reid_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.GAN_group.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.GAN_group.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        # self.GAN_group.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.GAN_group.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.GAN_group.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.GAN_group.add_argument('--iter_start', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        self.GAN_group.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.GAN_group.add_argument('--gan_lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.GAN_group.add_argument('--gan_lr_policy', type=str, default='lambda', help='learning rate policy[lambda|step|plateau]')
        self.GAN_group.add_argument('--gan_mode', type=str, default='lsgan', choices=['wgan-gp', 'hinge', 'lsgan'])
        # for discriminators        
        self.GAN_group.add_argument('--num_D', type=int, default=1, help='number of discriminators to use')
        self.GAN_group.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.GAN_group.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.GAN_group.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')                
        self.GAN_group.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.GAN_group.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        self.GAN_group.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

        '''reid'''
        # optimizer
        self.CC_group.add_argument('--reid_lr', type=float, default=0.00035,
                            help="learning rate")
        self.CC_group.add_argument('--weight-decay', type=float, default=5e-4)
        self.CC_group.add_argument('--epochs', type=int, default=50)
        self.CC_group.add_argument('--iters', type=int, default=400)
        self.CC_group.add_argument('--lr-step-size', type=int, default=20)

        # training configs
        self.parser.add_argument('--seed', type=int, default=1)
        self.parser.add_argument('--print-freq', type=int, default=10)
        self.GAN_group.add_argument('--vis-step', type=int, default=2)
        self.CC_group.add_argument('--eval-step', type=int, default=10)
        self.CC_group.add_argument('--temp', type=float, default=0.05,
                            help="temperature for scaling contrastive loss")
        self.parser.add_argument('--with_gan', action="store_true")

        '''additional args'''
        self.AL_group.add_argument('--warmup_epo', type=int, default=0, help='warm up epochs for reid')
        self.AL_group.add_argument('--lambda_nl', type=float, default=0.3, help='weight of following loss for gan')
        self.AL_group.add_argument('--lambda_ori', type=float, default=1., help='weight of following cl loss for reid')
        self.AL_group.add_argument('--lambda_cl', type=float, default=1., help='weight of following cl loss for reid')
        self.AL_group.add_argument('--dis_metric', choices=['ours', 'mse', 'cos'], type=str, default='ours', help='loss types of gradient matching')
        self.AL_group.add_argument('--cl_loss', action='store_true', help='if specified, train reid with added cl')
        self.AL_group.add_argument('--cl_temp', type=float, default=1.0, help='temperature of contrastive learning')
