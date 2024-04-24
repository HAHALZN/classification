import time
import torch
from utils.metrics import topk_accuracy
from utils.meters import AverageMeter, ProgressMeter


def train_target(target_loader, src_nets_f, src_nets_b, src_nets_c, tgt_net_f, tgt_net_b,
                 weight_layer, criterion_ce, criterion_ent, criterion_div, pl_init_c_all, pl_src_features_all,
                 optimizer, epoch, args=None, writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_ent = AverageMeter('LossEnt', ':.4e', writer=writer, tag="Loss/Entropy")
    losses_div = AverageMeter('LossDiv', ':.4e', writer=writer, tag="Loss/Diversity")
    losses_im = AverageMeter('LossIM', ':.4e', writer=writer, tag="Loss/InformationMaximization")
    losses_wd = AverageMeter('LossWD', ':.4e', writer=writer, tag="Loss/WeightDetermination")
    losses_ce = AverageMeter('LossCE', ':.4e', writer=writer, tag="Loss/CrossEntropy")
    losses = AverageMeter('Loss', ':.4e', writer=writer, tag="Loss/Overall")
    top1 = AverageMeter('Acc@1', ':6.2f', writer=writer, tag="Accuracy/top-1")
    # top5 = AverageMeter('Acc@5', ':6.2f', writer=writer, tag="Accuracy/top-5")
    progress = ProgressMeter(
        len(target_loader),
        # [batch_time, data_time, losses, top1, top5],
        [batch_time, data_time, losses_ent, losses_div, losses_im, losses_wd, losses_ce, losses, top1],
        prefix="Epoch: [{}/{}]".format(epoch + 1, args.epochs)
    )

    # switch to train mode
    tgt_net_f.train()
    tgt_net_b.train()
    weight_layer.train()
    for source in args.source:
        src_nets_f[source].eval()
        src_nets_b[source].eval()
        src_nets_c[source].eval()

    end = time.time()
    for i, (images, target, indices) in enumerate(target_loader):
        iter_num = epoch * len(target_loader) + i

        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)

        tgt_features = tgt_net_b(tgt_net_f(images))
        logits_all = torch.zeros(len(args.source), images.shape[0], args.num_classes).cuda()

        # weight determination loss
        if args.gamma > 0:
            src_features_all = torch.zeros(len(args.source), images.shape[0], tgt_features.shape[1]).cuda()
            for s, source in enumerate(args.source):
                logits_all[s] = src_nets_c[source](tgt_features)
                src_features_all[s] = src_nets_b[source](src_nets_f[source](images))

            taus = torch.mean(src_features_all - tgt_features, dim=1)
            mu = weight_layer(taus)
            if not args.no_wd:
                # 用于计算源域和目标域的Feature之间的Wasserstein距离
                loss_wd = torch.mean(mu.T * torch.sum((src_features_all - tgt_features) ** 2, dim=2), dim=1).sum()
            else:
                loss_wd = torch.tensor(0).cuda()
        else:
            for s, source in enumerate(args.source):
                logits_all[s] = src_nets_c[source](tgt_features)
            mu = torch.ones([1, len(args.source)]).cuda() / len(args.source)
            loss_wd = torch.tensor(0).cuda()

        logits_weighted = torch.sum(logits_all * mu.T.unsqueeze(dim=2), dim=0)

        # 信息熵损失
        loss_ent = criterion_ent(logits_weighted)
        # 熵分解
        loss_div = criterion_div(logits_weighted)
        loss_im = loss_ent + args.beta * loss_div

        # pseudo labeling and cross entropy loss
        if args.par > 0:
            # calculate pseudo labels and cross entropy loss
            initc = pl_init_c_all
            all_feas = pl_src_features_all

            initc_ = torch.zeros(initc[0].size()).cuda()
            temp = all_feas[0]
            all_feas_ = torch.zeros(temp[indices, :].size()).cuda()
            for s in range(len(args.source)):
                initc_ = initc_ + mu.t()[s] * initc[s].float()
                src_fea = all_feas[s]
                all_feas_ = all_feas_ + mu.t()[s] * src_fea[indices, :]
            dd = torch.cdist(all_feas_.float(), initc_.float(), p=2)
            pred_label = dd.argmin(dim=1)
            pred_label = pred_label.int()
            pred = pred_label.long()
            loss_ce = criterion_ce(logits_weighted, pred.cuda())
        else:
            loss_ce = torch.tensor(0).cuda()

        loss = loss_im + args.gamma * loss_wd + args.par * loss_ce
        # measure accuracy and record loss
        # acc1, acc5 = topk_accuracy(output, target, topk=(1, 5))
        acc1 = topk_accuracy(logits_weighted, target, topk=(1,))
        top1.update(acc1[0].item(), images.size(0), iter_num)


        losses_ent.update(loss_ent.item(), images.size(0), iter_num)
        losses_div.update(loss_div.item(), images.size(0), iter_num)
        losses_im.update(loss_im.item(), images.size(0), iter_num)
        losses_wd.update(loss_wd.item(), images.size(0), iter_num)
        losses_ce.update(loss_ce.item(), images.size(0), iter_num)
        losses.update(loss.item(), images.size(0), iter_num)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)







def train_label(target_loader, src_nets_f, src_nets_b, src_nets_c, tgt_net_f, tgt_net_b,
                 weight_layer, criterion_ce, criterion_ent, criterion_div, pl_init_c_all, pl_src_features_all,
                 optimizer, epoch, args=None, writer=None,label_flag=True):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_ent = AverageMeter('LossEnt', ':.4e', writer=writer, tag="Loss/Entropy")
    losses_div = AverageMeter('LossDiv', ':.4e', writer=writer, tag="Loss/Diversity")
    losses_im = AverageMeter('LossIM', ':.4e', writer=writer, tag="Loss/InformationMaximization")
    losses_wd = AverageMeter('LossWD', ':.4e', writer=writer, tag="Loss/WeightDetermination")
    losses_ce = AverageMeter('LossCE', ':.4e', writer=writer, tag="Loss/CrossEntropy")
    losses = AverageMeter('Loss', ':.4e', writer=writer, tag="Loss/Overall")
    top1 = AverageMeter('Acc@1', ':6.2f', writer=writer, tag="Accuracy/top-1")
    # top5 = AverageMeter('Acc@5', ':6.2f', writer=writer, tag="Accuracy/top-5")
    progress = ProgressMeter(
        len(target_loader),
        # [batch_time, data_time, losses, top1, top5],
        [batch_time, data_time, losses_ent, losses_div, losses_im, losses_wd, losses_ce, losses, top1],
        prefix="Epoch: [{}/{}]".format(epoch + 1, args.epochs)
    )

    # switch to train mode
    tgt_net_f.train()
    tgt_net_b.train()
    weight_layer.train()
    for source in args.source:
        src_nets_f[source].eval()
        src_nets_b[source].eval()
        src_nets_c[source].eval()

    end = time.time()
    for i, (images, target, indices) in enumerate(target_loader):
        iter_num = epoch * len(target_loader) + i

        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)

        tgt_features = tgt_net_b(tgt_net_f(images))
        logits_all = torch.zeros(len(args.source), images.shape[0], args.num_classes).cuda()

        # weight determination loss
        if args.gamma > 0:
            src_features_all = torch.zeros(len(args.source), images.shape[0], tgt_features.shape[1]).cuda()
            for s, source in enumerate(args.source):
                logits_all[s] = src_nets_c[source](tgt_features)
                src_features_all[s] = src_nets_b[source](src_nets_f[source](images))

            taus = torch.mean(src_features_all - tgt_features, dim=1)
            mu = weight_layer(taus)
            if not args.no_wd:
                loss_wd = torch.mean(mu.T * torch.sum((src_features_all - tgt_features) ** 2, dim=2), dim=1).sum()
            else:
                loss_wd = torch.tensor(0).cuda()
        else:
            for s, source in enumerate(args.source):
                logits_all[s] = src_nets_c[source](tgt_features)
            mu = torch.ones([1, len(args.source)]).cuda() / len(args.source)
            loss_wd = torch.tensor(0).cuda()

        logits_weighted = torch.sum(logits_all * mu.T.unsqueeze(dim=2), dim=0)

        loss_ent = criterion_ent(logits_weighted)
        loss_div = criterion_div(logits_weighted)
        loss_im = loss_ent + args.beta * loss_div

        # pseudo labeling and cross entropy loss
        if args.par > 0:
            # calculate pseudo labels and cross entropy loss
            initc = pl_init_c_all
            all_feas = pl_src_features_all

            initc_ = torch.zeros(initc[0].size()).cuda()
            temp = all_feas[0]
            all_feas_ = torch.zeros(temp[indices, :].size()).cuda()
            for s in range(len(args.src)):
                initc_ = initc_ + mu.t()[s] * initc[s].float()
                src_fea = all_feas[s]
                all_feas_ = all_feas_ + mu.t()[s] * src_fea[indices, :]
            dd = torch.cdist(all_feas_.float(), initc_.float(), p=2)
            pred_label = dd.argmin(dim=1)
            pred_label = pred_label.int()
            pred = pred_label.long()
            if label_flag:
                loss_ce = criterion_ce(logits_weighted, target)
            else:
                loss_ce = criterion_ce(logits_weighted, pred.cuda())
        else:
            loss_ce = torch.tensor(0).cuda()

        loss = loss_im + args.gamma * loss_wd + args.par * loss_ce
        # measure accuracy and record loss
        # acc1, acc5 = topk_accuracy(output, target, topk=(1, 5))
        acc1 = topk_accuracy(logits_weighted, target, topk=(1,))
        top1.update(acc1[0].item(), images.size(0), iter_num)


        losses_ent.update(loss_ent.item(), images.size(0), iter_num)
        losses_div.update(loss_div.item(), images.size(0), iter_num)
        losses_im.update(loss_im.item(), images.size(0), iter_num)
        losses_wd.update(loss_wd.item(), images.size(0), iter_num)
        losses_ce.update(loss_ce.item(), images.size(0), iter_num)
        losses.update(loss.item(), images.size(0), iter_num)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def train_source(train_loader, net_f, net_b, net_c, criterion, optimizer, epoch, args, writer=None, label_flag=True):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e', writer=writer, tag="Loss/train")
    top1 = AverageMeter('Acc@1', ':6.2f', writer=writer, tag="Accuracy/top-1")
    # top5 = AverageMeter('Acc@5', ':6.2f', writer=writer, tag="Accuracy/top-5")
    progress = ProgressMeter(
        len(train_loader),
        # [batch_time, data_time, losses, top1, top5],
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}/{}]".format(epoch + 1, args.epochs)
    )

    # switch to train mode
    net_f.train()
    net_b.train()
    net_c.train()

    end = time.time()
    for i, (images, target, _) in enumerate(train_loader):
        iter_num = epoch * len(train_loader) + i

        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)

        # compute output
        if label_flag:
            output = net_c(net_b(net_f(images)))
        # else:

        loss = criterion(output, target)

        # measure accuracy and record loss
        # acc1, acc5 = topk_accuracy(output, target, topk=(1, 5))
        acc1 = topk_accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0), iter_num)
        top1.update(acc1[0].item(), images.size(0), iter_num)
        # top5.update(acc5[0], images.size(0), iter_num)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

