import numpy as np
import sys
sys.path.append('../')
import mahalanobis, filteredGuassianCovTuned #might not be right?
import matplotlib.pyplot as plt

eps = 0.05
tau = 0.1

sampErr = []
noisyEmpErr = []
filterErr = []
ds = [x for x in np.arrange(10, 50 , 10)]

spikedCovariance = 1

if spikedCovariance:
    spike = 100
else:
    spike = 1

for d in ds:
    N = 0.5 * d / eps**2
    round = np.round(N, 0)
    print('Training with dimension = {d}, number of samples = {round}')
    sumEmpErr = 0
    sumNoisyEmpErr = 0
    sumFilterErr = 0

    covar = np.eye(d)
    if spikedCovariance:
        covar[0, 0] = spike
    X = np.random.multivariate_normal(np.zeros(0, d), covar, np.round((1 - eps) * N))
    if spikedCovariance:
        U1 = np.linalg.qr(np.random.normal(0, 1, (d, d)))
        Y = [ 0.5 * np.random.uniform(-1, 1, np.round(eps * N), d / 2), 0.8 * np.random.uniform(-2, 2, np.round(eps * N), d / 2 - 1), np.random.uniform(-spike, spike, (np.round(eps * N), 1))]
    else:
        Y = np.zeros(np.round(eps * N), d)

    Z = np.array([X, Y])

    print('Sampling error w/o noise...')
    empCov = np.cov(X)
    sumEmpErr = sumEmpErr + np.linalg.norm(mahalanobis(empCov, covar) - np.eye(d), 'fro')
    print('done')

    print('Sampling error with noise...')
    empCov = np.cov(Z)
    sumNoisyEmpErr = sumNoisyEmpErr + np.linalg.norm(mahalanobis(empCov, covar) - np.eye(d), 'fro')
    print('done')

    print('Filter...')
    ourCov, filterPoints, _ = filterGuassianCovTuned(Z, np.zeros(np.shape(Z)), eps, tau, False)
    sumFilterErr = sumFilterErr + np.linalg.norm(mahalanobis(ourCov, covar) - np.eye(d), 'fro')
    print('done')

    sampErr.append(sumEmpErr)
    noisyEmpErr.append(sumNoisyEmpErr)
    filterErr.append(sumFilterErr)

noisyEmpErr -= sampErr
filterErr -= sampErr

plt.plot(ds, noisyEmpErr, "--b")
plt.plot(ds, filterErr, "--b")
plt.show()