import numpy as np


def euclidean_error(y_pred, y_true):
    return np.sqrt(((np.asarray(y_pred) - np.asarray(y_true)) ** 2).sum(axis=1))


def coverage(errors, radii):
    return float(np.mean(np.asarray(errors) <= np.asarray(radii)))


def breach_rate(errors, radii):
    return 1.0 - coverage(errors, radii)
