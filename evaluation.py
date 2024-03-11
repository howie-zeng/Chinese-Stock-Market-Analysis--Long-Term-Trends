
import numpy as np

def calculate_r2_oos(y, y_hat):
    assert len(y_hat) == len(y)
    n = len(y_hat)
    mean_hat = np.mean(y_hat)
    SSR = np.sum((y - mean_hat)**2)
    SST = np.sum(y)
    res = 1 - SSR/SST
    return res