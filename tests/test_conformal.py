import numpy as np
from trustworthy_uq.calibration.conformal import conformal_quantile

def test_conformal_quantile():
    q = conformal_quantile(np.arange(10), 0.1)
    assert q >= 0
