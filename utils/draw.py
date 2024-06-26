import os
import numpy
import torch
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import interp
from sklearn.metrics import roc_curve, auc
from tsnecuda import TSNE
from sklearn.preprocessing import label_binarize
from .pretty_confusion_matrix import pp_matrix
from itertools import cycle

# COLOR_PALETTES = ["#0173B2", "#D55E00", "#029E73", "#CC78BC", "#ECE133", "#56B4E9", "#DE8F05"]
COLOR_PALETTES = ["#303872", "#852427", "#029E73", "#CC78BC", "#ECE133", "#56B4E9", "#DE8F05"]
def draw_pretty_confusion_matrix(confusion_matrix, output_fname=None, font_size=8, fig_size=[8, 8]):
    """
    print pretty confusion matrix
    @param fig_size: figure size
    @param font_size: font size
    @param confusion_matrix: confusion matrix
    @param output_fname: the absolute output path with file name, e.g., '/path/to/save/cm.png'
    @return: None
    """
    cm = pd.DataFrame(confusion_matrix)
    return pp_matrix(cm, output_fname=output_fname, fz=font_size, figsize=fig_size)


def draw_TSNE(features, y_true, label_names, output_dir, title=None, tsne_plot_count=1200):
    if features.shape[0] != len(y_true):
        raise RuntimeError('the number of features ({}) and y_true ({}) is unequal.'
                           .format(features.shape[0], len(y_true)))
    plt.figure(figsize=(10, 6))
    classes = np.unique(y_true)
    if tsne_plot_count > len(y_true):
        tsne_plot_count = len(y_true)
    # sample n = tsne_plot_count features
    sample_list = [i for i in range(len(y_true))]
    sample_list = random.sample(sample_list, tsne_plot_count)
    feature = numpy.asfarray([features[i].cpu().detach().numpy() for i in sample_list])
    y_labels = [y_true[i] for i in sample_list]
    # Dimensionality Reduction
    tsne_result = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=350, learning_rate=300).fit_transform(feature)
    data = {'x': np.array(tsne_result[:, 0]), 'y': np.array(tsne_result[:, 1]), 'label': np.array(y_labels)}
    for c in classes:
        plt.scatter(data['x'][data['label'] == c], data['y'][data['label'] == c],
                    c=COLOR_PALETTES[int(c)], marker='o', s=40)
    plt.legend(labels=label_names, loc="lower right")
    plt.axis('off')
    if title:
        plt.title(title)
    plt.savefig(os.path.join(output_dir, 'tsne.pdf'))


def draw_ROC(logits, y_true, label_names, output_dir, title=None):
    classes = np.unique(y_true)
    n_classes = len(np.unique(y_true))
    y_score = torch.nn.Softmax(dim=1)(logits.cpu()) if n_classes > 2 else torch.nn.Softmax(dim=1)(logits.cpu())[:, 1]
    y_true = label_binarize(y_true, classes=classes)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i]) if n_classes > 2 else roc_curve(y_true, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    if n_classes != 2:
        # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.detach().numpy().ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        #
        # fpr["macro"] = all_fpr
        # tpr["macro"] = mean_tpr
        # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        # plt.plot(fpr["micro"], tpr["micro"],
        #          label='micro-average ROC curve (area = {0:0.2f})'
        #                ''.format(roc_auc["micro"]),
        #          color='deeppink', linestyle=':', linewidth=4)
        #
        # plt.plot(fpr["macro"], tpr["macro"],
        #          label='macro-average ROC curve (AUC = {0:0.2f})'
        #                ''.format(roc_auc["macro"]),
        #          color='navy', linestyle=':', linewidth=4)
        lw = 2
        colors = cycle(sns.palettes.SEABORN_PALETTES['colorblind6'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (AUC = {1:0.2f})'
                     ''.format(label_names[i], roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)

    else:
        lw = 2
        plt.plot(fpr[0], tpr[0], color='#852427',
                 lw=lw, label='AUC = %0.2f' % roc_auc[0])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        fpr_tpr = np.concatenate((np.expand_dims(fpr[0], axis=1), np.expand_dims(tpr[0], axis=1)), axis=1)
        pd.DataFrame(fpr_tpr, columns=['fpr', 'tpr']).to_csv(os.path.join(output_dir, 'fpr_tpr_data.csv'), index=False)


    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if title:
        plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc.pdf'))