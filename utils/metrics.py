import torch
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score


def topk_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def precision_recall_f1score(y_true, y_pred, average="macro"):
    precision, recall, f1score, _ = precision_recall_fscore_support(y_true, y_pred, beta=1.0, average=average)
    return precision, recall, f1score


def sensitivity_specificity(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity


def quadratic_weighted_kappa_score(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')
