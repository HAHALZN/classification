# cd /home/lisen/zjj/code/classification
# python main.py -a vgg16 --lr 0.001 --epochs 80 --suffix vgg16_lr_0.001_epoch_80_bs_32 ./datasets/virus/virus_16_classes
# python main.py -a resnet50 --lr 0.001 --epochs 80 --suffix resnet50_lr_0.001_epoch_80_bs_32 ./datasets/virus/virus_16_classes
# python main.py -a googlenet --lr 0.001 --epochs 80 --suffix googlenet_lr_0.001_epoch_80_bs_32 ./datasets/virus/virus_16_classes
# python main.py -a densenet161 --lr 0.001 --epochs 80 --suffix densenet161_lr_0.001_epoch_80_bs_32 ./datasets/virus/virus_16_classes
# python main.py -a vgg16 --lr 0.005 --epochs 80 --suffix vgg16_lr_0.005_epoch_80_bs_32_wo_centercrop ./datasets/virus/virus_16_classes
# python main.py -a resnet50 --lr 0.005 --epochs 80 --suffix resnet50_lr_0.005_epoch_80_bs_32_wo_centercrop ./datasets/virus/virus_16_classes
# python main.py -a resnet50 --lr 0.005 --epochs 80 --suffix resnet101_lr_0.005_epoch_80_bs_32_wo_centercrop ./datasets/virus/virus_16_classes
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
# python main.py -a vgg16 -e --resume ./checkpoints/20220415005534_vgg16_lr_0.005_epoch_80_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a resnet50 -e --resume ./checkpoints/20220415010431_resnet50_lr_0.005_epoch_80_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a googlenet -e --resume ./checkpoints/20220415011112_googlenet_lr_0.005_epoch_80_bs_32/model_best.pth.tar ./datasets/virus/virus_16_classes
# python main.py -a densenet161 -e --resume ./checkpoints/20220415011759_densenet161_lr_0.005_epoch_80_bs_32/model_best.pth.tar ./datasets/virus/virus_16_classes
# python main.py -a vgg16 -e --resume ./checkpoints/20220414182403_vgg16_lr_0.01_epoch_40_bs_16/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a vgg16 -e --resume ./checkpoints/20220415082538_vgg16_lr_0.01_epoch_80_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# # python main.py -a vgg16 -e --resume ./checkpoints/20220415105001_vgg16_lr_0.01_epoch_80_bs_32_resize_48/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a vgg16 -e --resume ./checkpoints/20220415005534_vgg16_lr_0.005_epoch_80_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# # python main.py -a vgg16 -e --resume ./checkpoints/20220415101708_vgg16_lr_0.005_epoch_80_bs_32_resize_48/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a vgg16 -e --resume ./checkpoints/20220415130719_vgg16_lr_0.005_epoch_120_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a vgg16 -e --resume ./checkpoints/20220414233428_vgg16_lr_0.001_epoch_80_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a resnet50 -e --resume ./checkpoints/20220414175130_resnet50_lr_0.01_epoch_40_bs_16/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a resnet50 -e --resume ./checkpoints/20220415083326_resnet50_lr_0.01_epoch_80_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# # python main.py -a resnet50 -e --resume ./checkpoints/20220415105713_resnet50_lr_0.01_epoch_80_bs_32_resize_48/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a resnet50 -e --resume ./checkpoints/20220415010431_resnet50_lr_0.005_epoch_80_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# # python main.py -a resnet50 -e --resume ./checkpoints/20220415102404_resnet50_lr_0.005_epoch_80_bs_32_resize_48/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a resnet50 -e --resume ./checkpoints/20220415132053_resnet50_lr_0.005_epoch_120_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a resnet50 -e --resume ./checkpoints/20220414234427_resnet50_lr_0.001_epoch_80_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a resnet101 -e --resume ./checkpoints/20220415124724_resnet101_lr_0.005_epoch_80_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a resnet101 -e --resume ./checkpoints/20220415133054_resnet101_lr_0.005_epoch_120_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a googlenet -e --resume ./checkpoints/20220414174259_googlenet_lr_0.01_epoch_40_bs_16/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a googlenet -e --resume ./checkpoints/20220415084006_googlenet_lr_0.01_epoch_80_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# # python main.py -a googlenet -e --resume ./checkpoints/20220415105729_googlenet_lr_0.01_epoch_80_bs_32_resize_48/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a googlenet -e --resume ./checkpoints/20220415011112_googlenet_lr_0.005_epoch_80_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# # python main.py -a googlenet -e --resume ./checkpoints/20220415102420_googlenet_lr_0.005_epoch_80_bs_32_resize_48/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a googlenet -e --resume ./checkpoints/20220415134734_googlenet_lr_0.005_epoch_120_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a googlenet -e --resume ./checkpoints/20220414235123_googlenet_lr_0.001_epoch_80_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a densenet121 -e --resume ./checkpoints/20220414180758_densenet121_lr_0.01_epoch_40_bs_16/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a densenet161 -e --resume ./checkpoints/20220415084654_densenet161_lr_0.01_epoch_80_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# # python main.py -a densenet161 -e --resume ./checkpoints/20220415110400_densenet161_lr_0.01_epoch_80_bs_32_resize_48/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a densenet161 -e --resume ./checkpoints/20220415011759_densenet161_lr_0.005_epoch_80_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# # python main.py -a densenet161 -e --resume ./checkpoints/20220415103048_densenet161_lr_0.005_epoch_80_bs_32_resize_48/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a densenet161 -e --resume ./checkpoints/20220415135740_densenet161_lr_0.005_epoch_120_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a densenet161 -e --resume ./checkpoints/20220415003602_densenet161_lr_0.001_epoch_80_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes
# python main.py -a densenet169 -e --resume ./checkpoints/20220414-20220415/20220415142633_densenet169_lr_0.005_epoch_120_bs_32/model_best.pth.tar --draw-cm ./datasets/virus/virus_16_classes

# python main_target.py -a resnet50 --dx RDR --tgt A --src D --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0 --par 0 --suffix resnet50_tgt_A_src_D_bs_64_lr_0.005_beta_0.3_gamma_0_par_0_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --resume /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_TARGET_SINGLE_SRC_WITH_IM/20220503231608_resnet50_tgt_A_src_D_bs_64_lr_0.005_beta_0.3_gamma_0_par_0_size_full/checkpoint.pth.tar -e --draw-roc --draw-cm --draw-tsne ./datasets/retinal/DR/

# python main_target.py -a resnet50 --dx RDR --tgt A --src D --lr 0.005 --batch-size 64 --beta 0.3 --gamma 0.01 --par 0.4 --suffix resnet50_tgt_A_src_D_bs_64_lr_0.005_beta_0.3_gamma_0_par_0_size_full --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --resume /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_TARGET_IM_WD_CE/20220504222020_resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.01_par_0.4_size_full/checkpoint.pth.tar -e --draw-roc --draw-cm --draw-tsne ./datasets/retinal/DR

python main_target_30.py -a vit --dx RDR --tgt A --src D --lr 0.005 --batch-size 32 --beta 0.3 --gamma 0.01 --par 0 --src-models /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_NEW_full --resume /home/lisen/zjj/code/classification/checkpoints/train_source/RDR_TARGET_IM_WD_CE/20220505154629_resnet50_tgt_A_src_DIMM2_bs_64_lr_0.005_beta_0.3_gamma_0.01_par_0_no_wd_wo_abs_size_full/checkpoint.pth.tar -e --draw-tsne ./datasets/retinal/DR

