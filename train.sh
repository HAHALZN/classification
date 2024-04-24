# cd /home/lisen/zjj/code/classification
cd /root/autodl-tmp/classification
# python main.py -a vgg16 --lr 0.001 --epochs 80 --suffix vgg16_lr_0.001_epoch_80_bs_32 ./datasets/virus/virus_16_classes
# python main.py -a resnet50 --lr 0.001 --epochs 80 --suffix resnet50_lr_0.001_epoch_80_bs_32 ./datasets/virus/virus_16_classes
# python main.py -a googlenet --lr 0.001 --epochs 80 --suffix googlenet_lr_0.001_epoch_80_bs_32 ./datasets/virus/virus_16_classes
# python main.py -a densenet161 --lr 0.001 --epochs 80 --suffix densenet161_lr_0.001_epoch_80_bs_32 ./datasets/virus/virus_16_classes
# python main.py -a vgg16 --lr 0.005 --epochs 80 --suffix vgg16_lr_0.005_epoch_80_bs_32_wo_centercrop ./datasets/virus/virus_16_classes
# python main.py -a resnet50 --lr 0.005 --epochs 80 --suffix resnet50_lr_0.005_epoch_80_bs_32_wo_centercrop ./datasets/virus/virus_16_classes
# python main.py -a googlenet --lr 0.005 --epochs 80 --suffix googlenet_lr_0.005_epoch_80_bs_32_wo_centercrop ./datasets/virus/virus_16_classes
# python main.py -a densenet161 --lr 0.005 --epochs 80 --suffix densenet161_lr_0.005_epoch_80_bs_32_wo_centercrop ./datasets/virus/virus_16_classes
# python main.py -a vgg16 --lr 0.01 --epochs 80 --suffix vgg16_lr_0.01_epoch_80_bs_32 ./datasets/virus/virus_16_classes
# python main.py -a resnet50 --lr 0.01 --epochs 80 --suffix resnet50_lr_0.01_epoch_80_bs_32 ./datasets/virus/virus_16_classes
# python main.py -a goocd /home/lisen/zjj/code/classification
# python main.py -a vgg16 --lr 0.001 --epochs 80 --suffix vgg16_lr_0.001_epoch_80_bs_32 ./datasets/virus/virus_16_classes
# python main.py -a resnet50 --lr 0.001 --epochs 80 --suffix resnet50_lr_0.001_epoch_80_bs_32 ./datasets/virus/virus_16_classes
# python main.py -a googlenet --lr 0.001 --epochs 80 --suffix googlenet_lr_0.001_epoch_80_bs_32 ./datasets/virus/virus_16_classes
# python main.py -a densenet161 --lr 0.001 --epochs 80 --suffix densenet161_lr_0.001_epoch_80_bs_32 ./datasets/virus/virus_16_classes
# python main.py -a vgg16 --lr 0.005 --epochs 80 --suffix vgg16_lr_0.005_epoch_80_bs_32_resize_48 ./datasets/virus/virus_16_classes
# python main.py -a resnet50 --lr 0.005 --epochs 80 --suffix resnet50_lr_0.005_epoch_80_bs_32_resize_48 ./datasets/virus/virus_16_classes
# python main.py -a googlenet --lr 0.005 --epochs 80 --suffix googlenet_lr_0.005_epoch_80_bs_32_resize_48 ./datasets/virus/virus_16_classes
# python main.py -a densenet161 --lr 0.005 --epochs 80 --suffix densenet161_lr_0.005_epoch_80_bs_32_resize_48 ./datasets/virus/virus_16_classes
# python main.py -a vgg16 --lr 0.01 --epochs 80 --suffix vgg16_lr_0.01_epoch_80_bs_32_resize_48 ./datasets/virus/virus_16_classes
# python main.py -a resnet50 --lr 0.01 --epochs 80 --suffix resnet50_lr_0.01_epoch_80_bs_32_resize_48 ./datasets/virus/virus_16_classes
# python main.py -a googlenet --lr 0.01 --epochs 80 --suffix googlenet_lr_0.01_epoch_80_bs_32_resize_48 ./datasets/virus/virus_16_classes
# python main.py -a densenet161 --lr 0.01 --epochs 80 --suffix densenet161_lr_0.01_epoch_80_bs_32_resize_48 ./datasets/virus/virus_16_classes
# python main.py -a resnet101 --lr 0.005 --epochs 80 --suffix resnet101_lr_0.005_epoch_80_bs_32_wo_centercrop ./datasets/virus/virus_16_classes
# python main.py -a vgg16 --lr 0.005 --epochs 120 --suffix vgg16_lr_0.005_epoch_120_bs_32 ./datasets/virus/virus_16_classes
# python main.py -a resnet50 --lr 0.005 --epochs 120 --suffix resnet50_lr_0.005_epoch_120_bs_32 ./datasets/virus/virus_16_classes
# python main.py -a resnet101 --lr 0.005 --epochs 120 --suffix resnet101_lr_0.005_epoch_120_bs_32 ./datasets/virus/virus_16_classes
# python main.py -a googlenet --lr 0.005 --epochs 120 --suffix googlenet_lr_0.005_epoch_120_bs_32 ./datasets/virus/virus_16_classes
# python main.py -a densenet161 --lr 0.005 --epochs 120 --suffix densenet161_lr_0.005_epoch_120_bs_32 ./datasets/virus/virus_16_classes
# python main.py -a densenet169 --lr 0.005 --epochs 120 --suffix densenet169_lr_0.005_epoch_120_bs_32 ./datasets/virus/virus_16_classes

# python main_source.py -a resnet50 --lr 0.005 --src D --dx GRAD --suffix resnet50_lr_0.005_src_D_size_full --save-dir 
# python main_source.py -a resnet50 --lr 0.005 --src A --dx GRAD --suffix resnet50_lr_0.005_src_A_size_full --save-dir ./checkpoints/train_source/GRAD --draw-cm ./datasets/retinal/DR
# python main_source.py -a resnet50 --lr 0.005 --src I --dx GRAD --suffix resnet50_lr_0.005_src_I_size_full --save-dir ./checkpoints/train_source/GRAD --draw-cm ./datasets/retinal/DR
# python main_source.py -a resnet50 --lr 0.005 --src M2 --dx GRAD --suffix resnet50_lr_0.005_src_M_size_full --save-dir ./checkpoints/train_source/GRAD --draw-cm ./datasets/retinal/DR

# python main_source.py -a resnet50 --lr 0.005 --src D --dx GRAD --suffix resnet50_lr_0.005_src_D_size_3000 --save-dir ./checkpoints/train_source/GRAD --draw-cm ./datasets/retinal_3000/DR
# python main_source.py -a resnet50 --lr 0.005 --src A --dx GRAD --suffix resnet50_lr_0.005_src_A_size_3000 --save-dir ./checkpoints/train_source/GRAD --draw-cm ./datasets/retinal_3000/DR
# python main_source.py -a resnet50 --lr 0.005 --src I --dx GRAD --suffix resnet50_lr_0.005_src_I_size_3000 --save-dir ./checkpoints/train_source/GRAD --draw-cm ./datasets/retinal_3000/DR
# python main_source.py -a resnet50 --lr 0.005 --src M2 --dx GRAD --suffix resnet50_lr_0.005_src_M2_size_3000 --save-dir ./checkpoints/train_source/GRAD --draw-cm ./datasets/retinal_3000/DR

# python main_source.py -a resnet50 --lr 0.005 --src D --dx GRAD --suffix resnet50_lr_0.005_src_D_size_2000 --save-dir ./checkpoints/train_source/GRAD --draw-cm ./datasets/retinal_2000/DR
# python main_source.py -a resnet50 --lr 0.005 --src A --dx GRAD --suffix resnet50_lr_0.005_src_A_size_2000 --save-dir ./checkpoints/train_source/GRAD --draw-cm ./datasets/retinal_2000/DR
# python main_source.py -a resnet50 --lr 0.005 --src I --dx GRAD --suffix resnet50_lr_0.005_src_I_size_2000 --save-dir ./checkpoints/train_source/GRAD --draw-cm ./datasets/retinal_2000/DR
# python main_source.py -a resnet50 --lr 0.005 --src M2 --dx GRAD --suffix resnet50_lr_0.005_src_M2_size_2000 --save-dir ./checkpoints/train_source/GRAD --draw-cm ./datasets/retinal_2000/DR

# python main_source.py -a resnet50 --lr 0.005 --src D --dx RDR --suffix resnet50_lr_0.005_src_D_size_3000 --save-dir ./checkpoints/train_source/RDR_NEW --draw-cm ./datasets/retinal_3000/DR
# python main_source.py -a resnet50 --lr 0.005 --src A --dx RDR --suffix resnet50_lr_0.005_src_A_size_3000 --save-dir ./checkpoints/train_source/RDR_NEW --draw-cm ./datasets/retinal_3000/DR
# python main_source.py -a resnet50 --lr 0.005 --src I --dx RDR --suffix resnet50_lr_0.005_src_I_size_3000 --save-dir ./checkpoints/train_source/RDR_NEW --draw-cm ./datasets/retinal_3000/DR
# python main_source.py -a resnet50 --lr 0.005 --src M --dx RDR --suffix resnet50_lr_0.005_src_M_size_3000 --save-dir ./checkpoints/train_source/RDR_NEW --draw-cm ./datasets/retinal_3000/DR
# python main_source.py -a resnet50 --lr 0.005 --src M2 --dx RDR --suffix resnet50_lr_0.005_src_M2_size_3000 --save-dir ./checkpoints/train_source/RDR_NEW --draw-cm ./datasets/retinal_3000/DR

# python main_source.py -a resnet50 --lr 0.005 --src D --dx PDR --suffix resnet50_lr_0.005_src_D_size_3000 --save-dir ./checkpoints/train_source/PDR --draw-cm ./datasets/retinal_3000/DR
# python main_source.py -a resnet50 --lr 0.005 --src A --dx PDR --suffix resnet50_lr_0.005_src_A_size_3000 --save-dir ./checkpoints/train_source/PDR --draw-cm ./datasets/retinal_3000/DR
# python main_source.py -a resnet50 --lr 0.005 --src I --dx PDR --suffix resnet50_lr_0.005_src_I_size_3000 --save-dir ./checkpoints/train_source/PDR --draw-cm ./datasets/retinal_3000/DR
# python main_source.py -a resnet50 --lr 0.005 --src M2 --dx PDR --suffix resnet50_lr_0.005_src_M2_size_3000 --save-dir ./checkpoints/train_source/PDR --draw-cm ./datasets/retinal_3000/DR

# python main_source.py -a resnet50 --lr 0.005 --src D --dx NORM --suffix resnet50_lr_0.005_src_D_size_3000 --save-dir ./checkpoints/train_source/NORM --draw-cm ./datasets/retinal_3000/DR
# python main_source.py -a resnet50 --lr 0.005 --src A --dx NORM --suffix resnet50_lr_0.005_src_A_size_3000 --save-dir ./checkpoints/train_source/NORM --draw-cm ./datasets/retinal_3000/DR
# python main_source.py -a resnet50 --lr 0.005 --src I --dx NORM --suffix resnet50_lr_0.005_src_I_size_3000 --save-dir ./checkpoints/train_source/NORM --draw-cm ./datasets/retinal_3000/DR
# python main_source.py -a resnet50 --lr 0.005 --src M --dx NORM --suffix resnet50_lr_0.005_src_M_size_3000 --save-dir ./checkpoints/train_source/NORM --draw-cm ./datasets/retinal_3000/DR
# python main_source.py -a resnet50 --lr 0.005 --src M2 --dx NORM --suffix resnet50_lr_0.005_src_M2_size_3000 --save-dir ./checkpoints/train_source/NORM --draw-cm ./datasets/retinal_3000/DR

# python main_source.py -a resnet50 --lr 0.005 --src D --dx RDR --suffix resnet50_lr_0.005_src_D_size_full --save-dir ./checkpoints/train_source/RDR_NEW --draw-cm ./datasets/retinal/DR
# python main_source.py -a resnet50 --lr 0.005 --src A --dx RDR --suffix resnet50_lr_0.005_src_A_size_full --save-dir ./checkpoints/train_source/RDR_NEW --draw-cm ./datasets/retinal/DR
# python main_source.py -a resnet50 --lr 0.005 --src I --dx RDR --suffix resnet50_lr_0.005_src_I_size_full --save-dir ./checkpoints/train_source/RDR_NEW --draw-cm ./datasets/retinal/DR
# python main_source.py -a resnet50 --lr 0.005 --src M --dx RDR --suffix resnet50_lr_0.005_src_M_size_full --save-dir ./checkpoints/train_source/RDR_NEW --draw-cm ./datasets/retinal/DR
# python main_source.py -a resnet50 --lr 0.005 --src M2 --dx RDR --suffix resnet50_lr_0.005_src_M2_size_full --save-dir ./checkpoints/train_source/RDR_NEW --draw-cm ./datasets/retinal/DR

# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --beta 0.3 --suffix resnet50_tgt_A_src_DIMM2_beta_0.3_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --gpu-ids 0 2 --save-dir ./checkpoints/train_source/RDR_TARGET ./datasets/retinal/DR/ 
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --beta 0.2 --suffix resnet50_tgt_A_src_DIMM2_beta_0.2_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --gpu-ids 0 2 --save-dir ./checkpoints/train_source/RDR_TARGET ./datasets/retinal/DR/
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --beta 0.1 --suffix resnet50_tgt_A_src_DIMM2_beta_0.1_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --gpu-ids 0 2 --save-dir ./checkpoints/train_source/RDR_TARGET ./datasets/retinal/DR/
# python main_target.py -a resnet50 --dx RDR --tgt D --src I M M2 A --beta 0.3 --suffix resnet50_tgt_D_src_AIMM2_beta_0.3_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --gpu-ids 0 2 --save-dir ./checkpoints/train_source/RDR_TARGET ./datasets/retinal/DR/
# python main_target.py -a resnet50 --dx RDR --tgt D --src I M M2 A --beta 0.2 --suffix resnet50_tgt_D_src_AIMM2_beta_0.2_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --gpu-ids 0 2 --save-dir ./checkpoints/train_source/RDR_TARGET ./datasets/retinal/DR/
# python main_target.py -a resnet50 --dx RDR --tgt D --src I M M2 A --beta 0.1 --suffix resnet50_tgt_D_src_AIMM2_beta_0.1_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --gpu-ids 0 2 --save-dir ./checkpoints/train_source/RDR_TARGET ./datasets/retinal/DR/
# python main_target.py -a resnet50 --dx RDR --tgt I --src D M M2 A --beta 0.3 --suffix resnet50_tgt_I_src_ADMM2_beta_0.3_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --gpu-ids 0 2 --save-dir ./checkpoints/train_source/RDR_TARGET ./datasets/retinal/DR/
# python main_target.py -a resnet50 --dx RDR --tgt I --src D M M2 A --beta 0.2 --suffix resnet50_tgt_I_src_ADMM2_beta_0.2_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --gpu-ids 0 2 --save-dir ./checkpoints/train_source/RDR_TARGET ./datasets/retinal/DR/
# python main_target.py -a resnet50 --dx RDR --tgt I --src D M M2 A --beta 0.1 --suffix resnet50_tgt_I_src_ADMM2_beta_0.1_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --gpu-ids 0 2 --save-dir ./checkpoints/train_source/RDR_TARGET ./datasets/retinal/DR/

# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005 --batch-size 64 --beta 0.5 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.5_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET ./datasets/retinal/DR/ 
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005 --batch-size 64 --beta 0.4 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.4_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET ./datasets/retinal/DR/ 
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005 --batch-size 64 --beta 0.3 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET ./datasets/retinal/DR/ 
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005 --batch-size 64 --beta 0.2 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.2_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET ./datasets/retinal/DR/
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005 --batch-size 64 --beta 0.1 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.1_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET ./datasets/retinal/DR/

# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.1 --batch-size 64 --beta 0.3 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.1_beta_0.3_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET ./datasets/retinal/DR/ 
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.5 --batch-size 64 --beta 0.3 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.5_beta_0.3_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET ./datasets/retinal/DR/ 
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 1.0 --batch-size 64 --beta 0.3 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_1.0_beta_0.3_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET ./datasets/retinal/DR/ 
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.05 --batch-size 64 --beta 0.3 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.05_beta_0.3_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET ./datasets/retinal/DR/ 

# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005 --batch-size 64 --beta 0.3 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.01_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD ./datasets/retinal/DR/ 
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.01  --batch-size 64 --beta 0.3 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.01_beta_0.3_gamma_0.01_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD ./datasets/retinal/DR/ 
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.001 --batch-size 64 --beta 0.3 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.001_beta_0.3_gamma_0.01_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD ./datasets/retinal/DR/ 

# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0.1 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.1_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD ./datasets/retinal/DR/ 
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0.5 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.5_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD ./datasets/retinal/DR/ 
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0.05 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.05_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD ./datasets/retinal/DR/ 

# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005  --batch-size 64 --beta 0.3 --gamma 0 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD ./datasets/retinal/DR/ 
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005  --batch-size 64 --beta 0.3 --gamma 0.1 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.1_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD ./datasets/retinal/DR/ 
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005  --batch-size 64 --beta 0.3 --gamma 0.5 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.5_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD ./datasets/retinal/DR/ 

# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005 --batch-size 64 --beta 0.3 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_wd_only_opt_wl_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD ./datasets/retinal/DR/ 

# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD ./datasets/retinal/DR/
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0.1 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.1_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD ./datasets/retinal/DR/

# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0    --par 0.3 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0_par_0.3_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD_CE ./datasets/retinal/DR/
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0.01 --par 0.3 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.01_par_0.3_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD_CE ./datasets/retinal/DR/
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0.05 --par 0.3 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.05_par_0.3_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD_CE ./datasets/retinal/DR/

# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0    --par 0.4 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0_par_0.4_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD_CE ./datasets/retinal/DR/
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0.01 --par 0.4 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.01_par_0.4_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD_CE ./datasets/retinal/DR/
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0.05 --par 0.4 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.05_par_0.4_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD_CE ./datasets/retinal/DR/

# python main_target.py -a resnet50 --dx RDR --tgt A --src I --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0 --par 0 --suffix resnet50_tgt_A_src_I_bs_64_lr_0.005_beta_0.3_gamma_0_par_0_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_SINGLE_SRC_WITH_IM ./datasets/retinal/DR/
# python main_target.py -a resnet50 --dx RDR --tgt A --src M --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0 --par 0 --suffix resnet50_tgt_A_src_M_bs_64_lr_0.005_beta_0.3_gamma_0_par_0_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_SINGLE_SRC_WITH_IM ./datasets/retinal/DR/
# python main_target.py -a resnet50 --dx RDR --tgt A --src M2 --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0 --par 0 --suffix resnet50_tgt_A_src_M2_bs_64_lr_0.005_beta_0.3_gamma_0_par_0_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_SINGLE_SRC_WITH_IM ./datasets/retinal/DR/
# python main_target.py -a resnet50 --dx RDR --tgt A --src D --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0 --par 0 --suffix resnet50_tgt_A_src_D_bs_64_lr_0.005_beta_0.3_gamma_0_par_0_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_SINGLE_SRC_WITH_IM ./datasets/retinal/DR/

# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005  --batch-size 64 --beta 0.3 --gamma 0.3 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.3_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD ./datasets/retinal/DR/ 
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005  --batch-size 64 --beta 0.3 --gamma 0.6 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.6_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD ./datasets/retinal/DR/ 

# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005  --batch-size 64 --beta 0.3 --gamma 0.3 --no-wd --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.3_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_MU_wo_WD ./datasets/retinal/DR/ 
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005  --batch-size 64 --beta 0.3 --gamma 0.01 --no-wd --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.01_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_MU_wo_WD ./datasets/retinal/DR/ 
# python main_target.py -a resnet50 --dx RDR --tgt A --src I M M2 D --lr 0.005  --batch-size 64 --beta 0.3 --gamma 0.6 --no-wd --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.6_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_MU_wo_WD ./datasets/retinal/DR/ 

# python main_target.py -a resnet50 --dx RDR --tgt A --src I D M M2 --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0.01 --par 0 --no-wd --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.01_par_0_no_wd_wo_abs_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD_CE ./datasets/retinal/DR/


# python main_target.py -a resnet50 --dx RDR --tgt A --src I D M M2 --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0.01 --par 0.3 --suffix resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.01_par_0.3_no_wd_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --save-dir ./checkpoints/train_source/RDR_TARGET_IM_WD_CE ./datasets/retinal/DR/

# python main_source.py -a vgg16 --lr 0.005 --src D  --dx RDR --suffix vgg16_lr_0.005_src_D_size_full  --save-dir ./checkpoints/train_source/RDR_SOURCE_VGG16 ./datasets/retinal/DR

# python main_source.py -a efficientnet_b5 --lr 0.005 --src D  --dx RDR --suffix efficientnet_b5_lr_0.005_src_D_size_full  --save-dir ./checkpoints/train_source/RDR_SOURCE_EFFICIENTNETB5 ./datasets/retinal/DR

# python main_source.py -a vgg16 --lr 0.005 --src A  --dx RDR --suffix vgg16_lr_0.005_src_A_size_full  --save-dir ./checkpoints/train_source/RDR_SOURCE_VGG16 ./datasets/retinal/DR
# python main_source.py -a vgg16 --lr 0.005 --src I  --dx RDR --suffix vgg16_lr_0.005_src_I_size_full  --save-dir ./checkpoints/train_source/RDR_SOURCE_VGG16 ./datasets/retinal/DR
# python main_source.py -a vgg16 --lr 0.005 --src M  --dx RDR --suffix vgg16_lr_0.005_src_M_size_full  --save-dir ./checkpoints/train_source/RDR_SOURCE_VGG16 ./datasets/retinal/DR
# python main_source.py -a vgg16 --lr 0.005 --src M2 --dx RDR --suffix vgg16_lr_0.005_src_M2_size_full --save-dir ./checkpoints/train_source/RDR_SOURCE_VGG16 ./datasets/retinal/DR
# python main_source.py -a vgg16 --lr 0.005 --src E  --dx RDR --suffix vgg16_lr_0.005_src_E_size_full  --save-dir ./checkpoints/train_source/RDR_SOURCE_VGG16 ./datasets/retinal/DR

# python main_source.py -a efficientnet_b5 --lr 0.005 --src A  --dx RDR --suffix efficientnet_b5_lr_0.005_src_A_size_full  --save-dir ./checkpoints/train_source/RDR_SOURCE_EFFICIENTNETB5 ./datasets/retinal/DR
# python main_source.py -a efficientnet_b5 --lr 0.005 --src I  --dx RDR --suffix efficientnet_b5_lr_0.005_src_I_size_full  --save-dir ./checkpoints/train_source/RDR_SOURCE_EFFICIENTNETB5 ./datasets/retinal/DR
# python main_source.py -a efficientnet_b5 --lr 0.005 --src M  --dx RDR --suffix efficientnet_b5_lr_0.005_src_M_size_full  --save-dir ./checkpoints/train_source/RDR_SOURCE_EFFICIENTNETB5 ./datasets/retinal/DR
# python main_source.py -a efficientnet_b5 --lr 0.005 --src M2 --dx RDR --suffix efficientnet_b5_lr_0.005_src_M2_size_full --save-dir ./checkpoints/train_source/RDR_SOURCE_EFFICIENTNETB5 ./datasets/retinal/DR
# python main_source.py -a efficientnet_b5 --lr 0.005 --src E  --dx RDR --suffix efficientnet_b5_lr_0.005_src_E_size_full  --save-dir ./checkpoints/train_source/RDR_SOURCE_EFFICIENTNETB5 ./datasets/retinal/DR

# python main_target.py -a vgg16 --dx RDR --tgt A --src I D M M2 --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0.01 --par 0.3 --suffix vgg16_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.01_par_0.3_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_SOURCE_VGG16 --save-dir ./checkpoints/train_source/RDR_TARGET_VGG16 ./datasets/retinal/DR/
# python main_target.py -a vgg16 --dx RDR --tgt A --src I D M M2 --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0.01 --par 0 --suffix vgg16_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.01_par_0_no_wd_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_SOURCE_VGG16 --save-dir ./checkpoints/train_source/RDR_TARGET_VGG16 ./datasets/retinal/DR/
# python main_target.py -a vgg16 --dx RDR --tgt A --src I D M M2 --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0.01 --par 0 --no-wd --suffix vgg16_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.01_par_0_no_wd_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_SOURCE_VGG16 --save-dir ./checkpoints/train_source/RDR_TARGET_VGG16 ./datasets/retinal/DR/

# python main_target.py -a efficientnet_b5 --dx RDR --tgt A --src I D M M2 --lr 0.005 --batch-size 32 --beta 0.3 --gamma 0.01 --par 0.3 --suffix efficientnet_b5_tgt_A_src_DIMM2_bs_32_lr_0.005_beta_0.3_gamma_0.01_par_0.3_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_SOURCE_EFFICIENTNETB5 --save-dir ./checkpoints/train_source/RDR_TARGET_EFFICIENTNETB5 ./datasets/retinal/DR/
# python main_target.py -a efficientnet_b5 --dx RDR --tgt A --src I D M M2 --lr 0.005 --batch-size 32 --beta 0.3 --gamma 0.01 --par 0 --suffix efficientnet_b5_tgt_A_src_DIMM2_bs_32_lr_0.005_beta_0.3_gamma_0.01_par_0_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_SOURCE_EFFICIENTNETB5 --save-dir ./checkpoints/train_source/RDR_TARGET_EFFICIENTNETB5 ./datasets/retinal/DR/
# python main_target.py -a efficientnet_b5 --dx RDR --tgt A --src I D M M2 --lr 0.005 --batch-size 32 --beta 0.3 --gamma 0.01 --par 0 --no-wd --suffix efficientnet_b5_tgt_A_src_DIMM2_bs_32_lr_0.005_beta_0.3_gamma_0.01_par_0_no_wd_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_SOURCE_EFFICIENTNETB5 --save-dir ./checkpoints/train_target/RDR_TARGET_EFFICIENTNETB5 ./datasets/retinal/DR/

# python main_source.py -a resnet50 --lr 0.005 --src E --dx RDR --suffix resnet50_lr_0.005_src_E_size_full --save-dir ./checkpoints/train_source/RDR_NEW_full ./datasets/retinal/DR


python -X faulthandler main_target_30.py -a vit --dx RDR --tgt A --epochs 40 --gpu-ids 0 1 --draw-cm --draw-roc --draw-tsne --src D S --lr 0.005 --batch-size 32 --suffix VIT_tgt_A_src_DS_bs_32_lr_0.005_size_full --src-models /root/autodl-tmp/classification/checkpoints/train_source/RDR_SOURCE_VIT --save-dir ./checkpoints/train_target/RDR_TARGET_VIT ./datasets/retinal/DR/


# python -X faulthandler main_target.py -a vit --dx RDR --tgt A --epochs 40 --gpu-ids 0 --draw-cm --draw-roc --draw-tsne --src D --lr 0.005 --batch-size 32 --suffix vit_tgt_A_src_D_bs_32_lr_0.005_size_full --src-models /root/autodl-tmp/classification/checkpoints/train_source/RDR_SOURCE_VIT --save-dir ./checkpoints/train_target/RDR_TARGET_VIT ./datasets/retinal/DR/

# python -X faulthandler main_source_RETFound.py -a vit --epochs 20 --dx RDR --tgt A --gpu-ids 0 1 --draw-cm --draw-roc --draw-tsne --src S --lr 0.005 --batch-size 32 --suffix vit_tgt_A_src_S_bs_32_lr_0.005_size_full --save-dir ./checkpoints/train_source/RDR_SOURCE_VIT ./datasets/retinal/DR/

# python -X faulthandler main_source_RETFound.py -a vit --epochs 40 --dx RDR --tgt A --draw-cm --draw-roc --draw-tsne --src M --lr 0.005 --batch-size 32 --suffix vit_tgt_A_src_M_bs_32_lr_0.005_size_full --save-dir ./checkpoints/train_source/RDR_SOURCE_VIT ./datasets/retinal/DR/

# python -X faulthandler main_source_RETFound.py -a vit --epochs 40 --dx RDR --tgt A --draw-cm --draw-roc --draw-tsne --src M2 --lr 0.005 --batch-size 32 --suffix vit_tgt_A_src_M2_bs_32_lr_0.005_size_full --save-dir ./checkpoints/train_source/RDR_SOURCE_VIT ./datasets/retinal/DR/


