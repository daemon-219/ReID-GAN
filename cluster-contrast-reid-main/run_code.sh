# cd /home/mmc_zhaojiacheng/project/people_image/ReID_GAN/cluster-contrast-reid-main/

export PYTHONPATH=$PYTHONPATH:/home/mmc_zhaojiacheng/project/people_image/ReID_GAN/cluster-contrast-reid-main/ 
export CUDA_VISIBLE_DEVICES=0,1,2,3

python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --gan_train --model AE --model_gen Pose \
    --lambda_nl 1. --eval-step 1 --vis-step 1 \
    -b 128 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 8 

# python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --model AE --model_gen AE \
#     --load_pretrain examples/Logs/AE_warmup1_logs/AE/AE_CC_market_test \
#     --lambda_fus 0.7 --vis-step 1 --eval-step 1 -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 

    # --load_pretrain examples/Logs/AE_warmup_logs/AE/AE_CC_market_test \
    # --lambda_nl 1. --eval-step 1 --vis-step 1 
    # --gan_lr 0.00035 --gan_lr_policy step \
    # --load_pretrain examples/Logs/AE_warmup_logs/AE/AE_CC_market_test 
    # --lambda_rec 4.0 --lambda_g 10.0 --dis_metric cos_m 

# python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --use_adp --model AE \
#     --load_pretrain examples/pretrained/market_AE --eval-step 1  --dis_metric cos \
#     --gan_lr 0.002 --gan_lr_policy step --gan_lr_policy exponent --lr-step-size 20 --lambda_nl 1 -b 64 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 
    
# python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --gan_train --model AE --continue_train \
#     --load_pretrain examples/pretrained/market_AE --reid_pretrain examples/Logs/logs \
#     --warmup_epo 0 --reid_lr 0.0000035 --cl_loss --lambda_ori 1 --lambda_cl 1 -b 128 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 

# python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --gan_train --use_adp --continue_train --model DPTN --load_pretrain examples/pretrained/market -b 32 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 

# python examples/cluster_contrast_train_usl_infomap.py -b 128 --lr 0.0000035 --reid_pretrain examples/Logs/logs \
#  -a resnet50 -d market1501 --epochs 20 --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16

# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16

# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16


# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d dukemtmcreid --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d dukemtmcreid --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16


# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d veri --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16 --height 224 --width 224
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d veri --iters 400 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16 --height 224 --width 224
