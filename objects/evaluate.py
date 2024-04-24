import os
import time
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist

from utils.metrics import topk_accuracy, precision_recall_f1score,\
    sensitivity_specificity, quadratic_weighted_kappa_score
from utils.meters import Summary, AverageMeter, ProgressMeter
from utils.draw import draw_pretty_confusion_matrix, draw_ROC, draw_TSNE


def summary(y_true, y_pred, outputs, features, top1, tag, writer, epoch, args):
    # calculate confusion matrix
    labels = torch.unique(torch.concat((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print_str = ''
    # calculate metrics
    if args.num_classes == 2:
        sensitivity, specificity = sensitivity_specificity(cm)
        precision, recall, f1score = precision_recall_f1score(y_true, y_pred, average="binary")
        if writer:
            writer.add_scalar(tag + "/Sensitivity", sensitivity, epoch + 1)
            writer.add_scalar(tag + "/Specificity", specificity, epoch + 1)
        print_str += " *   Sensitivity: {:.4f} Specificity: {:4f}\n".format(sensitivity, specificity)
    else:
        qwk = quadratic_weighted_kappa_score(y_true, y_pred)
        precision, recall, f1score = precision_recall_f1score(y_true, y_pred, average="macro")
        if writer:
            writer.add_scalar(tag + "/QuadraticWeightedKappa", qwk, epoch + 1)
        print_str += " *   Quadratic Weighted Kappa Score: {:.4f}\n".format(qwk)
    print_str += " *   Precision: {:.4f} Recall: {:.4f} F1-score: {:.4f}".format(precision, recall, f1score)

    # if binary classification, calculate sensitivity and specificity
    label_names = ['Non-referable DR', 'Referable DR']
    output_dir = os.path.dirname(args.resume) if args.evaluate else args.output_dir
    if args.draw_cm:
        if args.evaluate:
            draw_pretty_confusion_matrix(cm, output_fname=os.path.join(output_dir, "cm.pdf"),
                                         font_size=9, fig_size=[7, 7])
        else:
            img = draw_pretty_confusion_matrix(cm, font_size=9, fig_size=[7, 7])
            if writer:
                writer.add_image(tag + "/ConfusionMatrix", img, epoch + 1, dataformats='HWC')
    if args.draw_roc:
        if args.evaluate:
            draw_ROC(outputs, y_true, label_names, output_dir=output_dir)
    if args.draw_tsne:
        if args.evaluate:
            draw_TSNE(features, y_true, label_names, output_dir=output_dir)

    if writer:
        writer.add_scalar(tag + "/Accuracy", top1.avg, epoch + 1)
        writer.add_scalar(tag + "/Precision", precision, epoch + 1)
        writer.add_scalar(tag + "/Recall", recall, epoch + 1)
        writer.add_scalar(tag + "/F1score", f1score, epoch + 1)

    print(print_str)

def evaluate_source(source_loaders, target_loaders, net_f, net_b, net_c, criterion, epoch, writer, args):
    print("Evaluating: source ({})".format(args.source))
    tag = "SourceTestOn_" + args.src
    acc1 = evaluate(source_loaders['test'], net_f, net_b, net_c, criterion, epoch, writer, tag, args)

    for name, loader in target_loaders.items():
        print("Evaluating: source ({}) to target ({})".format(args.source, name))
        tgt = name[0] if name != "Messidor-2" else "M2"
        tag = "SourceTestOnTarget_" + tgt
        evaluate(loader, net_f, net_b, net_c, criterion, epoch, writer, tag, args)
    return acc1


def evaluate_target(target_loader, src_nets_f, src_nets_b, src_nets_c, tgt_net_f, tgt_net_b,
                    weight_layer, criterion, epoch, args, writer=None):
    batch_time = AverageMeter('Time', ':6.3f', summary_type=Summary.NONE)
    losses_ce = AverageMeter('LossCE', ':.4e', summary_type=Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', summary_type=Summary.AVERAGE)
    # top5 = AverageMeter('Acc@5', ':6.2f', summary_type=Summary.AVERAGE)
    progress = ProgressMeter(
        len(target_loader),
        # [batch_time, losses, top1, top5],
        [batch_time, top1],
        prefix='Test: '
    )

    # switch to evaluate mode
    tgt_net_f.eval()
    tgt_net_b.eval()
    weight_layer.eval()
    for source in args.source:
        src_nets_f[source].eval()
        src_nets_b[source].eval()
        src_nets_c[source].eval()

    pl_src_features_all = dict()
    pl_src_logits_all = dict()
    pl_init_c_all = dict()
    with torch.no_grad():
        end = time.time()
        for i, (images, target, _) in enumerate(target_loader):
        # for i, tmp in enumerate(target_loader):
        #     print(tmp)

        #     print(len(tmp))
        #     exit(0)
            if torch.cuda.is_available():
                target = target.cuda(non_blocking=True)

            # compute output,使用了目标域的模型参数
            tgt_features = tgt_net_b(tgt_net_f(images))
            logits_all = torch.zeros(len(args.source), images.shape[0], args.num_classes).cuda()

            if args.gamma > 0 or args.par > 0:
                src_features_all = torch.zeros(len(args.source), images.shape[0], tgt_features.shape[1]).cuda()
                for s, source in enumerate(args.source):
                    logits_all[s] = src_nets_c[source](tgt_features)
                    # 使用了不同源域的参数，输入了目标域图像
                    src_features_all[s] = src_nets_b[source](src_nets_f[source](images))

                    if args.par > 0:
                        src_logits = src_nets_c[source](src_features_all[s])
                        if i == 0:
                            pl_src_features_all[source] = src_features_all[s].cpu()
                            pl_src_logits_all[source] = src_logits.cpu()
                        else:
                            pl_src_features_all[source] = torch.cat((pl_src_features_all[source], src_features_all[s].cpu()), 0)
                            pl_src_logits_all[source] = torch.cat((pl_src_logits_all[source], src_logits.cpu()), 0)
                # 计算所有源获取的特征与目标源的差距，取平均值
                taus = torch.mean(src_features_all - tgt_features, dim=1)
                mu = weight_layer(taus)
            else:
                for s, source in enumerate(args.source):
                    logits_all[s] = src_nets_c[source](tgt_features)
                mu = torch.ones([1, len(args.source)]).cuda() / len(args.source)
            
            
            logits_weighted = torch.sum(logits_all * mu.T.unsqueeze(dim=2), dim=0)
            loss = criterion(logits_weighted, target)

            # acc1, acc5 = topk_accuracy(output, target, topk=(1, 5))
            acc1 = topk_accuracy(logits_weighted, target, topk=(1,))
            losses_ce.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            # top5.update(acc5[0], images.size(0))

            # record to calculate metrics
            if i == 0:
                y_true = target.cpu()
                outputs = logits_weighted.cpu()
                features = tgt_features.cpu()
            else:
                y_true = torch.cat((y_true, target.cpu()), 0)
                outputs = torch.cat((outputs, logits_weighted.cpu()), 0)
                features = torch.cat((features, tgt_features.cpu()), 0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        _, y_pred = torch.max(outputs, 1)
        tag = 'MultiSourceTrainOnTarget_' + args.target
        summary(y_true, y_pred, outputs, features, top1, tag, writer, epoch, args)

        # print the evaluate results
        progress.display_summary()

        # calculate mean features
        if args.par > 0 and not args.evaluate:
            print("=> calculating the mean features and the accuracy of pseudo labels")
            for source in args.source:
                all_fea = pl_src_features_all[source]
                all_output = pl_src_logits_all[source]
                all_output = torch.nn.Softmax(dim=1)(all_output)
                # _, predict = torch.max(all_output, 1)
                # accuracy = torch.sum(torch.squeeze(predict).float() == y_true).item() / float(y_true.size()[0])

                all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1).to(all_fea.device)), 1)
                all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
                all_fea = all_fea.float().cpu().numpy()

                K = all_output.size(1)
                aff = all_output.float().cpu().numpy()
                initc = aff.transpose().dot(all_fea)
                initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

                dd = cdist(all_fea, initc, 'cosine')
                pred_label = dd.argmin(axis=1)
                # acc = np.sum(pred_label == y_true.float().numpy()) / len(all_fea)

                for round in range(1):
                    aff = np.eye(K)[pred_label]
                    initc = aff.transpose().dot(all_fea)
                    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
                    dd = cdist(all_fea, initc, 'cosine')
                    pred_label = dd.argmin(axis=1)
                    acc = np.sum(pred_label == y_true.float().numpy()) / len(all_fea)

                pl_init_c_all[source] = torch.from_numpy(initc).cuda()
                pl_src_features_all[source] = torch.from_numpy(all_fea).cuda()

                print("the accuracy of pseudo labels on source model {} is {:.2f}%".format(source, acc * 100))
                if writer:
                    writer.add_scalar(tag + "/PseudoLabelAccOn{}".format(source), acc, epoch + 1)

    return top1.avg, list(pl_init_c_all.values()), list(pl_src_features_all.values())


def evaluate(val_loader, net_f, net_b, net_c, criterion, epoch, writer, tag, args):
    batch_time = AverageMeter('Time', ':6.3f', summary_type=Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', summary_type=Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', summary_type=Summary.AVERAGE)
    # top5 = AverageMeter('Acc@5', ':6.2f', summary_type=Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        # [batch_time, losses, top1, top5],
        [batch_time, losses, top1],
        prefix='Test: '
    )

    # switch to evaluate mode
    net_f.eval()
    net_b.eval()
    net_c.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target, _) in enumerate(val_loader):

            if torch.cuda.is_available():
                target = target.cuda(non_blocking=True)

            # compute output
            features = net_b(net_f(images))
            output = net_c(features)
            loss = criterion(output, target)

            # measure accuracy and record loss

            # acc1, acc5 = topk_accuracy(output, target, topk=(1, 5))
            acc1 = topk_accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            # top5.update(acc5[0], images.size(0))

            # record to calculate metrics
            if i == 0:
                y_true = target.cpu()
                outputs = output.cpu()
                features_all = features.cpu()
            else:
                y_true = torch.cat((y_true, target.cpu()), 0)
                outputs = torch.cat((outputs, output.cpu()), 0)
                features_all = torch.cat((features_all, features.cpu()), 0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        _, y_pred = torch.max(outputs, 1)
        summary(y_true, y_pred, outputs, features_all, top1, tag, writer, epoch, args)

        # print the evaluate results
        progress.display_summary()

    return top1.avg
