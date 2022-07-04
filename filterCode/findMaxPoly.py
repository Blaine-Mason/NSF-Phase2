from flatten import flatten
from sharpen import sharpen
import numpy as np
def findMaxPoly(data):
    N, d = data.shape
    v = flatten(np.eye(d))

    dataKron = np.zeros((N, d*d))
    for i in range(N):
        dataKron[i, :] = np.kron(data[i, :], data[i, :])
    
    empFourth = dataKron.T @ dataKron/(N - v.T@v)
    print(empFourth)
    U, S, _ = np.linalg.svd(empFourth)
    lamb = S[1,1] / 2
    Mflat = U[:, 1]

    M = sharpen(Mflat, d) / np.sqrt(2)
    x = np.linalg.trace(M) / np.sqrt(2)

    return c, M, lamb
