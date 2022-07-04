import scipy.sparse.linalg as sla
import numpy as np
def svdsecon(X,k, num):

    m,n = X.shape
    
    if  m <= n:
        C = X@X.T
        D, U = sla.eigs(C,k)
        C = 0
        if num > 2:
            V = X.T@U
            s = np.sqrt(abs(np.diag(D)))
            V = V/s.T
            S = np.diag(s)
            return U, S, V
    else:
        C = X.T@X; 
        D, V = sla.eigs(C,k)
        C = 0
        U = X@V; 
        s = np.sqrt(abs(np.diag(D)))
        U = U/s.T
        S = np.diag(s)
    return U, S, V