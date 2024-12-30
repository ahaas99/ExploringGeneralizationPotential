import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score
from medmnist.evaluator import getACC, getAUC

def get_acc(y_true: np.ndarray, y_pred: np.ndarray, task: str):
    """
    Calculates the accuracy of the prediction.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param task: Type of classification task.
    :return: Accuracy
    """

    return getACC(y_true, y_pred, task)

def get_auc(y_true: np.ndarray, y_pred: np.ndarray, task: str):
    """
    Calculates the Area-under-the-ROC curve of the prediction.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param task: Type of classification task.
    :return: AUC score.
    """
    return getAUC(y_true, y_pred, task)

def get_balanced_acc(y_true: np.ndarray, y_pred: np.ndarray, task: str, threshold=0.5):
    """
    Calculates the accuracy of the prediction adapted for the kNN approach.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param task: Type of classification task.
    :return: Accuracy
    """
    if task == "multi-label, binary-class":
        y_pre = y_pred > threshold
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = balanced_accuracy_score(y_true[:, label], y_pre[:, label])
            acc += label_acc
        ret = acc / y_true.shape[1]
    elif task == "binary-class":
        if y_pred.ndim == 2:
            y_pred = y_pred[:, -1]
        else:
            assert y_pred.ndim == 1
        ret = balanced_accuracy_score(y_true, y_pred > threshold)
    else:
        ret = balanced_accuracy_score(y_true, np.argmax(y_pred, axis=-1))

    return ret

def get_cohen(y_true: np.ndarray, y_pred: np.ndarray, task: str, threshold=0.5):
    if task == "multi-label, binary-class":
        y_pre = y_pred > threshold
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = cohen_kappa_score(y_true[:, label], y_pre[:, label])
            acc += label_acc
        ret = acc / y_true.shape[1]
    elif task == "binary-class":
        if y_pred.ndim == 2:
            y_pred = y_pred[:, -1]
        else:
            assert y_pred.ndim == 1
        ret = cohen_kappa_score(y_true, y_pred > threshold)
    else:
        ret = cohen_kappa_score(y_true, np.argmax(y_pred, axis=-1))

    return ret

def get_precision(y_true: np.ndarray, y_pred: np.ndarray, task: str, threshold=0.5):
    if task == "multi-label, binary-class":
        y_pre = y_pred > threshold
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = precision_score(y_true[:, label], y_pre[:, label], zero_division=np.nan)
            acc += label_acc
        ret = acc / y_true.shape[1]
    elif task == "binary-class":
        if y_pred.ndim == 2:
            y_pred = y_pred[:, -1]
        else:
            assert y_pred.ndim == 1
        ret = precision_score(y_true, y_pred > threshold, pos_label=1, zero_division=np.nan)
    else:
        ret = precision_score(y_true, np.argmax(y_pred, axis=-1), average='macro', zero_division=np.nan)

    return ret
