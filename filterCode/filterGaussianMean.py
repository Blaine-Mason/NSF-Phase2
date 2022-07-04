import numpy as np
import scipy.special
from svdsecon import svdsecon
def filterGaussianMean(data, eps, tau, cher):
    
    N, d = data.shape
    empiricalMean = np.mean(data)
    threshold = eps*np.log(1/eps)
    centeredData = (data - empiricalMean) / np.sqrt(N)
    U, S, _ = svdsecon(centeredData.T, 1, 3)

    lamba = S[0]**2
    V = U[:,0]
    
    if lamba < (1 + 3 * threshold):
        estMean = empiricalMean
    else:
        delta = 2*eps
        projectedData1 = data * V
        med = np.median(projectedData1)

        projectedData = (np.hstack((np.abs(data*V - med), data)))

        sortedProjectedData = np.sort(projectedData, axis = 0)
        for i in range(N):
            T = sortedProjectedData[i, 0] - delta
            if (N - i) > (cher * N * scipy.special.erfc(T / np.sqrt(2)) / (2 + eps / (d * np.log(d * eps / tau)))):
                break
        
        if i == 0 or i == N-1:
            estMean = empiricalMean
        else:
            estMean = filterGaussianMean(sortedProjectedData[0:i, 1:-1], eps, tau, cher)
            print(estMean)
    return estMean