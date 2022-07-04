import numpy as np

def flatten(matrix):
    d = matrix.shape
    v = np.zeros((1, (d * d)))
    for i in range(d):
        for j in range(d):
            v[(i - 1) * d + j] = matrix[i, j]
    return v