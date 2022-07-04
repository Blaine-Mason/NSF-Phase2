import numpy as np

def sharpen(v, d):
    M = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            M[i, j] = v[(i-1)*d + j]
    return M
            