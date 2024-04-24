import argparse
import os
import random
import shutil
import warnings
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

from objects.train import train_target
from objects.evaluate import evaluate_target
from objects.dataloader import get_target_loader
from objects.networks import backbone, bottleneck, classifier, weight
from objects.target_selection import target_selection
from objects.losses import Entropy, EntropyDiv

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-d', '--delimiter', default=' ', type=str,
                    help='delimiter of the image list between path and label')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', dest='weight_decay',
                    help='weight decay (default: 1e-4)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', default=True,
                    help='use pre-trained model')
# tricks
parser.add_argument('-i', '--imbalanced-sampler', action='store_true',
                    dest='imbalanced_sampler',
                    help='use imbalanced dataset sampler')
parser.add_argument('--label-smoothing', type=float, default=0.0,
                    dest='label_smoothing',
                    help='the amount of smoothing')
# evaluate
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--draw-cm', dest='draw_cm', action='store_true',
                    help='draw confusion matrix')
parser.add_argument('--draw-roc', dest='draw_roc', action='store_true',
                    help='draw roc curve')
parser.add_argument('--draw-tsne', dest='draw_tsne', action='store_true',
                    help='draw t-SNE plot')
# checkpoints and tensorboard
parser.add_argument('-s', '--save', dest='save', type=bool, default=True,
                    help='save checkpoint')
parser.add_argument('--save-dir', default='./checkpoints', type=str, metavar='PATH',
                    dest='save_dir',
                    help='checkpoints directory (default: ./checkpoints)')
parser.add_argument('--evaluate-freq', default=5, type=int, dest="evaluate_freq",
                    help='evaluate frequency (default: 5)')
parser.add_argument('-t', '--tensorboard', dest='tensorboard', default=True,
                    help='use tensorboard to record results')
parser.add_argument('--suffix', default='', type=str,
                    help='suffix of the checkpoint directory and tensorboard files')
parser.add_argument('--seed', default=2022, type=int,
                    help='seed for initializing training')
parser.add_argument('--gpu-ids', dest='gpu_ids', nargs='*', type=str, default=None,
                    help='GPU ids to use')
# DR-SFUDA
parser.add_argument('--src', default=None, nargs='*', type=str, choices=['A', 'I', 'M', 'D', 'E', 'M2'],
                    help='source domains, if empty, use all the datasets except target domain,'
                         ' A: APTOS2019, I: IDRiD, M: Messidor, D: DDR,'
                         ' E: EyePACS(Kaggle), M2: Messidor-2')
parser.add_argument('--tgt', default='ALL', type=str, choices=['A', 'I', 'M', 'D', 'E', 'M2'],
                    help='target domain, A: APTOS2019, I: IDRiD, M: Messidor, D: DDR,'
                         ' E: EyePACS(Kaggle), M2: Messidor-2')
parser.add_argument('--dx', default='GRAD', type=str, choices=['GRAD', 'RDR', 'PDR', 'NORM'],
                    help='diagnosis type, GRAD: DR grading, RDR: referable/non-referable DR,'
                         'PDR: proliferative/non-proliferative DR, NORM: normal/abnormal')
parser.add_argument('--src-models', dest="src_models", default='./checkpoints', type=str, metavar='PATH',
                    help='path to source models (default: none)')
parser.add_argument('--beta', type=float, default=0.3, dest='beta',
                    help='the hyper-parameter beta')
parser.add_argument('--gamma', type=float, default=0.01, dest='gamma',
                    help='the hyper-parameter gamma')
parser.add_argument('--par', type=float, default=0.3, dest='par',
                    help='the hyper-parameter par')
parser.add_argument('--no-wd', dest='no_wd', action='store_true',
                    help='no weight determination loss')

best_acc1 = 0


def main():
    args = parser.parse_args()
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu_ids)
        warnings.warn('You have chosen specific GPUs.')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args)


def main_worker(args):
    global best_acc1

    if args.gpu_ids is not None:
        print("Use GPU: {} for training".format(args.gpu_ids))

    # DR dataset
    names = {'A': 'APTOS2019', 'I': 'IDRiD', 'M': 'Messidor', 'D': 'DDR', 'M2': 'Messidor-2', 'E': 'EyePACS'}

    args.target = names[args.tgt]
    if not args.src:
        args.source = list(names.values())
    else:
        args.source = [names[src] for src in args.src]

    # data loading
    target_loader = get_target_loader(args)

    # load source models
    checkpoints = dict()
    if os.path.isdir(args.src_models):
        if len(os.listdir(args.src_models)) < len(args.source) + 1:
            raise RuntimeError('too little checkpoints in {}, please check'.format(args.src_models))
        for d in os.listdir(args.src_models):
            src_model_path = os.path.join(args.src_models, d)
            if os.path.isdir(src_model_path):
                model_checkpoints = os.path.join(src_model_path, 'checkpoint.pth.tar')
                if os.path.exists(model_checkpoints):
                    checkpoint = torch.load(model_checkpoints)
                    source = checkpoint['source']
                    if source != args.target and source in args.source:
                        print("=> found source ({}) model from '{}'".format(checkpoint['source'], model_checkpoints))
                        checkpoints[source] = checkpoint
                else:
                    print("=> no checkpoint found at '{}'".format(model_checkpoints))

    # create model
    print("=> loading source models from checkpoints")
    nets_f, nets_b, nets_c = dict(), dict(), dict()
    param_group = []
    learning_rate = args.lr
    for source in args.source:
        if source != args.target:
            net_f = backbone(arch=args.arch, pretrained=args.pretrained)
            net_b = bottleneck(net_f.out_features, type="bn")
            net_c = classifier(args.num_classes, type="wn")

            net_f, net_b, net_c = torch.nn.DataParallel(net_f).cuda(), \
                                  torch.nn.DataParallel(net_b).cuda(), \
                                  torch.nn.DataParallel(net_c).cuda()

            net_f.load_state_dict(checkpoints[source]['state_dict_net_f'])
            net_b.load_state_dict(checkpoints[source]['state_dict_net_b'])
            net_c.load_state_dict(checkpoints[source]['state_dict_net_c'])
            net_f.eval()
            net_b.eval()
            net_c.eval()
            for k, v in net_f.named_parameters():
                v.requires_grad = False
            for k, v in net_b.named_parameters():
                v.requires_grad = False
            for k, v in net_c.named_parameters():
                v.requires_grad = False
            nets_f[source] = net_f
            nets_b[source] = net_b
            nets_c[source] = net_c

    # init target model
    tgt_net_f = backbone(arch=args.arch, pretrained=args.pretrained)
    tgt_net_b = bottleneck(tgt_net_f.out_features, type="bn")
    weight_layer = weight(tgt_net_b.bottleneck_dim)

    tgt_net_f, tgt_net_b, weight_layer = torch.nn.DataParallel(tgt_net_f).cuda(), \
                                         torch.nn.DataParallel(tgt_net_b).cuda(), \
                                         weight_layer.cuda()
    if not args.resume:
        tgt_net_f, tgt_net_b = target_selection(target_loader, tgt_net_f, tgt_net_b,
                                               nets_f, nets_b, nets_c, checkpoints, args)
    for source in args.source:
        del checkpoints[source]
    tgt_net_f.eval()
    tgt_net_b.eval()
    weight_layer.eval()

    for k, v in tgt_net_f.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in tgt_net_b.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate * 10}]
    for k, v in weight_layer.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate * 10}]


    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda()
    criterion_ent = Entropy().cuda()
    criterion_div = EntropyDiv().cuda()

    optimizer = torch.optim.SGD(param_group, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    epoch = 0

    # load checkpoint
    if args.resume:
        print("=> loading model from the target checkpoints: {}".format(args.resume))
        tgt_checkpoint = torch.load(args.resume)
        tgt_net_f.load_state_dict(tgt_checkpoint['state_dict_net_f'])
        tgt_net_b.load_state_dict(tgt_checkpoint['state_dict_net_b'])
        weight_layer.load_state_dict(tgt_checkpoint['state_dict_weight_layer'])
        args.start_epoch = tgt_checkpoint['epoch']
        best_acc1 = tgt_checkpoint['best_acc1']
        optimizer.load_state_dict(tgt_checkpoint['optimizer'])
        scheduler.load_state_dict(tgt_checkpoint['scheduler'])
        epoch = tgt_checkpoint['epoch']
        del tgt_checkpoint

    cudnn.benchmark = True

    if args.evaluate:
        evaluate_target(target_loader, nets_f, nets_b, nets_c, tgt_net_f, tgt_net_b,
                        weight_layer, criterion_ce, epoch - 1, args, writer=None)
        return

    # output directory
    now_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if not args.resume:
        args.output_dir = os.path.join(args.save_dir,
                                       now_time + (("_" + args.suffix) if args.suffix else ''))
    else:
        args.output_dir = os.path.basename(args.resume)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # create tensorboard writer
    writer = None
    if args.tensorboard:
        writer = SummaryWriter(log_dir=args.output_dir)

    pl_init_c_all = None
    pl_src_features_all = None
    for epoch in range(args.start_epoch, args.epochs):
        # evaluate before training
        if epoch == 0:
            _, pl_init_c_all, pl_src_features_all = evaluate_target(target_loader, nets_f, nets_b, nets_c,
                                                                    tgt_net_f, tgt_net_b, weight_layer,
                                                                    criterion_ce, epoch - 1, args, writer=writer)
        # train for one epoch
        train_target(target_loader, nets_f, nets_b, nets_c, tgt_net_f, tgt_net_b, weight_layer,
                     criterion_ce, criterion_ent, criterion_div, pl_init_c_all, pl_src_features_all,
                     optimizer, epoch, args=args, writer=writer)

        if args.save and ((epoch + 1) % args.evaluate_freq == 0 or (epoch + 1) == args.epochs):
            acc1, pl_init_c_all, pl_src_features_all = evaluate_target(target_loader, nets_f, nets_b, nets_c, tgt_net_f, tgt_net_b,
                                   weight_layer, criterion_ce, epoch, args, writer=writer)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'source': args.source,
                'state_dict_net_f': tgt_net_f.state_dict(),
                'state_dict_net_b': tgt_net_b.state_dict(),
                'state_dict_weight_layer': weight_layer.state_dict(),
                # Todo: add source classifiers' state_dict
                # 'state_dict_net_c': ,
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, args.output_dir)

        scheduler.step()
    writer.flush()
    writer.close()


def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth.tar'):
    filename = os.path.join(output_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(output_dir, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
