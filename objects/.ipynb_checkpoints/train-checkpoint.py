import time
import torch
from utils.metrics import topk_accuracy
from utils.meters import AverageMeter, ProgressMeter


def train_target(target_loader, tgt_net_f, tgt_net_b, src_nets_c, criterion_ent, criterion_div,
                 optimizer, epoch, args, writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_ent = AverageMeter('LossEnt', ':.4e', writer=writer, tag="Loss/Entropy")
    losses_div = AverageMeter('LossDiv', ':.4e', writer=writer, tag="Loss/Diversity")
    losses_im = AverageMeter('LossIM', ':.4e', writer=writer, tag="Loss/InformationMaximization")
    top1 = AverageMeter('Acc@1', ':6.2f', writer=writer, tag="Accuracy/top-1")
    # top5 = AverageMeter('Acc@5', ':6.2f', writer=writer, tag="Accuracy/top-5")
    progress = ProgressMeter(
        len(target_loader),
        # [batch_time, data_time, losses, top1, top5],
        [batch_time, data_time, losses_ent, losses_div, losses_im, top1],
        prefix="Epoch: [{}/{}]".format(epoch + 1, args.epochs)
    )

    # switch to train mode
    tgt_net_f.train()
    tgt_net_b.train()
    for source in args.source:
        src_nets_c[source].train()

    end = time.time()
    for i, (images, target, _) in enumerate(target_loader):
        iter_num = epoch * len(target_loader) + i

        # measure data loading time
        data_time.update(time.time() - end)

        # compute h_t
        features = tgt_net_b(tgt_net_f(images))
        logits = torch.zeros_like(features)
        for source in args.source:
            logits += src_nets_c[source](features)
        logits_mean = logits / len(args.source)

        loss_ent = criterion_ent(logits_mean)
        loss_div = criterion_div(logits_mean)
        loss_im = loss_ent + args.beta * loss_div

        # measure accuracy and record loss
        # acc1, acc5 = topk_accuracy(output, target, topk=(1, 5))
        acc1 = topk_accuracy(logits_mean, target, topk=(1,))
        losses_ent.update(loss_ent.item(), images.size(0), iter_num)
        losses_div.update(loss_div.item(), images.size(0), iter_num)
        losses_im.update(loss_im.item(), images.size(0), iter_num)
        top1.update(acc1[0].item(), images.size(0), iter_num)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_im.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def train_source(train_loader, net_f, net_b, net_c, criterion, optimizer, epoch, args, writer=None):
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
        output = net_c(net_b(net_f(images)))
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
