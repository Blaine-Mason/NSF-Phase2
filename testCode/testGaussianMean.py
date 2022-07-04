import numpy as np

import sys
sys.path.append('../filterCode/')
from filterGaussianMean import filterGaussianMean
import matplotlib.pyplot as plt
eps = .1
tau = .1
cher = 2.5

filterErr = []
sampErr = []
noisySampErr = []
ds = [x for x in np.arange(100, 450, 50)]

for d in ds:
    N = 10*np.floor(d/((eps)**2))
    print(f"Training with dimension = {d}, Samples = {round(N)}")
    sumFilterErr = 0
    sumSampErr = 0
    sumNoisySampErr = 0

    X = np.random.multivariate_normal(np.zeros(d), np.identity(d), round((1-eps)*N)) + np.ones((round((1-eps)*N),d))
    print("Sampling Error without noise")
    print(np.mean(X))
    sumSampErr = sumSampErr + np.linalg.norm(np.mean(X) - np.ones((1,d)))
    print("Done")
    
    Y1 = np.random.uniform(0, 1, (round(.5*eps*N), d))
    a = np.array(12* np.ones(round(.5*eps*N)))
    b = np.array(-2*np.ones(round(.5*eps*N)))
    c =  np.array(np.zeros((round(.5*eps*N), d-2)))
    Y2 = np.hstack((np.vstack((a,b)).T, c))
    new_Y = np.vstack((Y1, Y2))
    X =np.vstack((X, new_Y))
    print('Sampling Error with noise')
    print(np.mean(X))
    sumNoisySampErr = sumNoisySampErr +  np.linalg.norm(np.mean(X) - np.ones((1,d)))
    print(np.linalg.norm(np.mean(X) - np.ones((1,d))))
    print('...done\n')

    print('Filter')
    test = filterGaussianMean(X, eps, tau, cher)
    print(test)
    sumFilterErr = sumFilterErr + np.linalg.norm(test - np.ones((1, d)))
    print('...done\n')

    filterErr.append(sumFilterErr)
    sampErr.append(sumSampErr)
    noisySampErr.append(sumNoisySampErr)

noisySampErr = np.array(noisySampErr) - np.array(sampErr)
filterErr = np.array(filterErr) - np.array(sampErr)

plt.plot(ds, noisySampErr, "--r", linewidth=2)
plt.plot(ds, filterErr, "--b")
plt.show()
