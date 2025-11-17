import numpy as np


def accuracy ( y_true:np.array, y_pred:np.array) -> float:
    return (y_pred == y_true).sum() / len(y_true)