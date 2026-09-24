import math
import numpy as np


def conformal_quantile(scores, alpha):
    scores = np.asarray(scores).reshape(-1)
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - alpha))
    rank = min(max(rank, 1), n)
    return float(np.sort(scores)[rank - 1])
