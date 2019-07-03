import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import exposure
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


"""
Plotting utilities
Author:     Nicolas EKICIER
Release:    V1.1    07/2019
                - Add plot_confusion_matrix function
            V1.O    06/2019
                - Initialization
"""

def imadjust(input):
    '''
    Adjust intensity image
    :param input:   image (numpy array)
    :return:        rescale image
    '''
    p3, p97 = np.percentile(input, (3, 97))
    if p3 == 0:
        ip3 = 4
        while p3 == 0:
            p3 = np.percentile(input, ip3)
            ip3 = ip3 + 1
    img_rescale = exposure.rescale_intensity(input, in_range=(p3, p97))
    return img_rescale


def plot_confusion_matrix(y_true, y_pred,
                          labels=None,
                          normalize=False,
                          title=None,
                          cmap='YlGn'):
    """
    This function prints and plots the confusion matrix.
    :param y_true:      y_true
    :param y_pred:      y_pred
    :param labels:      labels of classes [label num_class] (optional)
                        type : pandas dataframe
    :param normalize:   normalize data (default = False)
    :param title:       title of figure (optional)
    :param cmap:        cmap
                        type = string (default = YlGn)
    :return:
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if isinstance(labels, pd.DataFrame):
        id = []
        for i in classes:
            test = np.flatnonzero(labels.iloc[:,-1] == i)
            if test:
                id.append(int(test))
        labels = list(labels.iloc[id, 0])
    else:
        labels = classes

    fig = plt.figure(figsize=(12,9))
    ax = fig.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    ax.figure.colorbar(im, ax=ax)
    ax.xaxis.tick_top()

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax