# cd /home/mmc_zhaojiacheng/project/people_image/ReID_GAN/cluster-contrast-reid-main/

export PYTHONPATH=$PYTHONPATH:/home/mmc_zhaojiacheng/project/people_image/ReID_GAN/cluster-contrast-reid-main/ 
export CUDA_VISIBLE_DEVICES=2,3,0,1

# GAN train
# python examples/train_gan_warmup.py --name AE_warmup_msmt17 --logs-dir ./examples/Logs/AE_warmup_msmt17 --checkpoints_dir ./examples/Logs/AE_warmup_msmt17 \
#     --gan_train --model AE --model_gen AE \
#     --epochs 50 -b 256 -d msmt17 --vis-step 1 \

# python examples/train_gan_warmup.py --name DEC_warmup --logs-dir ./examples/Logs/DEC_warmup --checkpoints_dir ./examples/Logs/DEC_warmup \
#     --gan_train --warmup_with_reid_enc --model AE --model_gen DEC --layers_g 3 --num_blocks 3 --num_feats 2048 --gan_lr 0.0002 --no_vgg_loss \
#     --bipath --epochs 50 -b 256 -a resnet_bipd50 -d market1501 --vis-step 1 \

# python examples/train_gan_warmup.py --name pose_gan_warmup --logs-dir ./examples/Logs/pose_gan_warmup --checkpoints_dir ./examples/Logs/pose_gan_warmup \
#     --gan_train --model AE --model_gen Pose --layers_g 3 --num_feats 256 --nhead 2 --num_CABs 3 --num_TTBs 3 --gan_lr 0.0002 --no_vgg_loss \
#     --bipath --epochs 50 -b 256 -d market1501 --vis-step 1 \

# train reid and gan
# --no_vgg_loss 
# python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --gan_train --model AE --model_gen Pose \
#     --layers_g 4 --num_feats 256 --gan_lr 0.0002 --gan_lr_policy step --eval-step 1 --vis-step 1 \
#     -b 128 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 8 \
#     --name pose_gan_tg --logs-dir ./examples/Logs/pose_gan_tg --checkpoints_dir ./examples/Logs/pose_gan_tg

# # latest 
# """
# gan_train
# """
    # python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --gan_train --model AE --model_gen DEC \
    # --layers_g 3 --gan_lr 0.0002 --gan_lr_policy step --no_vgg_loss --lambda_nl 1 \
    python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --gan_train --model AE --model_gen Pose \
    --layers_g 3 --gan_lr 0.0002 --gan_lr_policy step --no_vgg_loss --lambda_nl 1 \
    --vis-step 1 --eval-step 1 -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 \
    --name test_pose_rew --logs-dir ./examples/Logs/test_pose_rew --checkpoints_dir ./examples/Logs/test_pose_rew

    # python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --gan_train --model AE --model_gen Pose \
    # --layers_g 3 --num_feats 256 --nhead 2 --num_CABs 2 --num_TTBs 2 --no_vgg_loss --gan_lr 0.0002 --gan_lr_policy step \

    # python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --gan_train --model AE --model_gen DEC \
    # --layers_g 3 --num_blocks 3 --num_feats 1024 --gan_lr 0.0002 --no_vgg_loss --gan_lr_policy step \
    # --reid_lr 0.00035 --lr-step-size 20 --eval-step 1 --vis-step 1 \
    # --lambda_fus 0.5 -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 \
    # --name origin_test_decg1 --logs-dir ./examples/Logs/origin_test_decg1 --checkpoints_dir ./examples/Logs/origin_test_decg1

    # --use_conf

    # python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --gan_train --model AE --model_gen Pose \
    # --layers_g 3 --num_feats 256 --nhead 2 --num_CABs 3 --num_TTBs 3 --no_vgg_loss --gan_lr 0.0002 --gan_lr_policy step \
    # -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 \
    # --name origin_test_poseg --logs-dir ./examples/Logs/origin_test_poseg --checkpoints_dir ./examples/Logs/origin_test_poseg

    # # --use_conf

# python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --gan_train --model AE --model_gen Pose \
#     --layers_g 5 --num_feats 256 --num_CABs 3 --num_TTBs 3 --no_vgg_loss --gan_lr 0.0002 --gan_lr_policy step \
#     --load_pretrain examples/Logs/pose_gan_warmup_u5/pose_gan_warmup_u5 \
#     --lambda_nl 0.5 --reid_lr 0.00035 --lr-step-size 20 --eval-step 1 --vis-step 1 \
#     --cf_temp 1. \
#     -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 \
#     --name origin_test_filter_l1 --logs-dir ./examples/Logs/origin_test_filter_l1 --checkpoints_dir ./examples/Logs/origin_test_filter_l1

#     # --use_conf

# latest_bip
# python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --gan_train --model AE --model_gen Pose \
#     --bipath --layers_g 3 --num_feats 256 --nhead 4 --num_CABs 3 --num_TTBs 3 --no_vgg_loss --gan_lr 0.0002 --gan_lr_policy step \
#     --load_pretrain examples/Logs/pose_gan_warmup_bip_g2/pose_gan_warmup_bip_g2 \
#     --lambda_fus 0.5 --reid_lr 0.00035 --lr-step-size 20 --eval-step 1 --vis-step 1 \
#     -b 256 -a resnet_bipd50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 \
#     --name origin_bi_ex_fgan --logs-dir ./examples/Logs/origin_bi_ex_fgan --checkpoints_dir ./examples/Logs/origin_bi_ex_fgan

    # --load_pretrain examples/Logs/pose_gan_warmup_bip_g2/pose_gan_warmup_bip_g2 \
    # --load_pretrain examples/Logs/pose_gan_warmup_u/pose_gan_warmup_u \
    # --use_conf --bipath_gan --cf_temp 0.2 --lambda_nl 0.5 --lambda_nl 100 --cl_loss 

# python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --gan_train --model AE --model_gen Pose \
#     --bipath --layers_g 5 --num_feats 256 --num_CABs 3 --num_TTBs 3 --no_vgg_loss --gan_lr 0.0002 --gan_lr_policy step \
#     --load_pretrain examples/Logs/pose_gan_warmup_u5/pose_gan_warmup_u5 \
#     --lambda_nl 0.5 --reid_lr 0.00035 --lr-step-size 20 --eval-step 1 --vis-step 1 \
#     --cf_temp 1. \
#     -b 256 -a resnet_bip50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 \
#     --name origin_test_bi_g --logs-dir ./examples/Logs/origin_test_bi_g --checkpoints_dir ./examples/Logs/origin_test_bi_g

    # --load_pretrain examples/Logs/pose_gan_warmup_u/pose_gan_warmup_u \
    # --use_conf --bipath_gan

# python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --gan_train --model AE --model_gen Pose \
#     --load_pretrain examples/Logs/pose_gan_warmup/pose_gan_warmup \
#     --layers_g 4 --num_feats 256 --gan_lr 0.0002 --gan_lr_policy step --use-conf --eval-step 1 --vis-step 1 \
#     -b 256 -a resnet50 -d market1501 --iters 200 --lambda_fus 0.5 --cf_temp 0.2 --momentum 0. --eps 0.6 --num-instances 16 \
#     --name pose_gan_rewp2 --logs-dir ./examples/Logs/pose_gan_rewp2 --checkpoints_dir ./examples/Logs/pose_gan_rewp2

# python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --gan_train --model AE --model_gen Pose \
#     --layers_g 5 --num_feats 256 --num_CABs 3 --num_TTBs 3 --no_vgg_loss --gan_lr 0.0002 --gan_lr_policy step --eval-step 1 --vis-step 1 \
#     --load_pretrain examples/Logs/pose_gan_warmup_u/pose_gan_warmup_u \
#     --learnable_memory --cluster_lr 0.5 --lambda_fus 0.7 --cf_temp 0.2 \
#     -b 256 -a resnet50 -d market1501 --iters 200 --eps 0.6 --num-instances 16 \
#     --name pose_gan_reid_tp --logs-dir ./examples/Logs/pose_gan_reid_tp --checkpoints_dir ./examples/Logs/pose_gan_reid_tp

    # --load_pretrain examples/Logs/pose_gan_warmup/pose_gan_warmup \ --cf_temp 0.2
    # --continue_train --reid_pretrain examples/Logs/pose_gan_reid_rwclr0.5 \
    # --load_pretrain examples/Logs/pose_gan_reid_rwclr0.5/pose_gan_reid_rwclr0.5 \

# train reid
# python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --model AE --model_gen AE \
#     --load_pretrain examples/Logs/AE_warmup4/AE_warmup4 \
#     --lambda_fus 0.7 --vis-step 1 --eval-step 1 -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 \
#     --name test_market_ex --logs-dir ./examples/Logs/test_market_ex --checkpoints_dir ./examples/Logs/test_market_ex
    
# python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --model AE --model_gen AE \
#     --load_pretrain examples/Logs/AE_warmup_msmt17/AE_warmup_msmt17 \
#     --lambda_fus 0.7 --vis-step 1 --eval-step 1 -b 256 -a resnet50 -d msmt17 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 \
#     --name test_msmt17_ex --logs-dir ./examples/Logs/test_msmt17_ex --checkpoints_dir ./examples/Logs/test_msmt17_ex
    
    # --name baseline --logs-dir ./examples/Logs/baseline --checkpoints_dir ./examples/Logs/baseline

    # --load_pretrain examples/Logs/AE_warmup_logs/AE/AE_CC_market_test \
    # --lambda_nl 1. --eval-step 1 --vis-step 1 
    # --gan_lr 0.00035 --gan_lr_policy step \
    # --load_pretrain examples/Logs/AE_warmup_logs/AE/AE_CC_market_test 
    # --lambda_rec 4.0 --lambda_g 10.0 --dis_metric cos_m 

# python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --use_adp --model AE \
#     --load_pretrain examples/pretrained/market_AE --eval-step 1  --dis_metric cos \
#     --gan_lr 0.002 --gan_lr_policy step --gan_lr_policy exponent --lr-step-size 20 --lambda_nl 1 -b 64 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 
    
# python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --model AE \
#     --eval-step 1 --warmup_epo 50 -b 256 -a resnet50 -d market1501 --iters 200 --momentum 1. --eps 0.6 --num-instances 16 

# python examples/cluster_contrast_gan_train_usl_infomap.py --with_gan --gan_train --use_adp --continue_train --model DPTN --load_pretrain examples/pretrained/market -b 32 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 

# python examples/cluster_contrast_gan_train_usl_infomap.py \
#     -b 256 -a resnet50 -d dukemtmcreid --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --eval-step 1 --num-instances 16 \
#     --name baseline_duke --logs-dir ./examples/Logs/baseline_duke --checkpoints_dir ./examples/Logs/baseline_duke

# python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --eval-step 1 --num-instances 16

# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16

# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16


# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d dukemtmcreid --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d dukemtmcreid --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16


# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d veri --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16 --height 224 --width 224
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d veri --iters 400 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16 --height 224 --width 224
