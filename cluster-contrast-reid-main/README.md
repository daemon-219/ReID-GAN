
# Modifications
Do data loading part in clustercontrast/utils/data/preprocessor.py

Modify training pipeline in clustercontrast/trainers.py

GAN model in dual_gan/models/DPTN_model.py

Training options in examples/options

```shell
--with_gan : use gan
--gan_train : trian gan at the same time
--cl_loss : use contrastive loss for reid, add a predictor on original model arch
--use_adp : use adaptor for gan, train it with reid
--continue_train : load pretrained reid and gan to continue train
-a (resnet50) : model arch for reid
--model (AE/DPTN): model arch for gan 
--lambda_fus : ratio for feature fusion in gan
```

