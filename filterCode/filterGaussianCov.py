from findMaxPoly import findMaxPoly
import numpy as np
import scipy.linalg as la

def filterGaussianCov(data, metadata, eps, tau, expConst, debug):
    MAX_COND = 10000

    N, d = data.shape
    threshold = eps*(np.log(1/eps))**2
    C1 = 0.4
    C2 = 0
    
    empCov = (data).T * data/N
    condition = np.linalg.cond(empCov)

    if condition > MAX_COND or N > d:
        if debug:
            print(f"Ill conditioned {condition} {N} {d}")
        estCov = np.zeros(d)
        filteredPoints = []
        filteredMetadata = []
        return
    

    empCovInv = np.linalg.inv(empCov)
    restart = False
    remove = []
    for i in range(N):
        if i % 10000 == 0 and debug:
            print(f"Initial pruning iteration {i}")
        x = data[i, :]
        if x @ empCovInv @ x.T > C1*d*np.log(N / tau):
            remove.append(i)
            restart = True
    if restart:
        data[remove, :] = []
        metadata[remove, :] = []
        estCov, filteredPoints, filteredMetadata = filterGaussianCov(data, metadata, eps, tau, expConst, debug)
        return estCov, filteredPoints, filteredMetadata
    if debug:
        print("After First for loop\n")

    rotData = data @ la.sqrtm(empCovInv)
    if debug:
        print('Before findMaxPoly\n')
    _, M, lambd = findMaxPoly(rotData)

    if debug:
        print('After findMaxPoly\n')
    if debug:
        print(f"lambda = {lambd}, threshold = {1 + C2 * threshold}")
        if d < 12:
            np.linalg.svd(M)
    
    if lambd < 1 + C2 * threshold:
        estCov = empCov
        filteredPoints = data
        filteredMetadata = metadata
    else:
        projectedData = np.zeros((N, 1))
        for i in range(N):
            projectedData[i, :] = rotData[i, :] @ M @ rotData[i, :].T

        med = np.mean(projectedData)

        if debug:
            print(f" Mean {med}, Trace {np.linalg.trace(M)}, var {np.var(projectedData)}, Lambda {lambd}")
            print(f"N {N}")
        
        indices = np.abs(projectedData - med)
        sortedProjectedData = np.sort(np.concatenate((indices, data)))
        sortedMetadata = np.sort(np.concatenate((indices, metadata)))

        for i in range(N):
            T = sortedProjectedData[i,0]
            rhs = N @ (12* np.exp(-expConst*T) + 3 * eps/(d*log(N/tau))**2)
            if (N-i) > rhs:
                break
        if i == 0 or i == N-1:
            estCov = empCov
            filteredPoints = data
            filteredMetadata = metadata
            return estCov, filteredPoints, filteredMetadata
        else:
            estCov, filteredPoints, filteredMetadata = filterGaussianCov(sortedProjectedData[0:i, 1:-1], sortedMetadata[0:i,1:-1], eps, tau, expConst, debug)

    return estCov, filteredPoints, filteredMetadata