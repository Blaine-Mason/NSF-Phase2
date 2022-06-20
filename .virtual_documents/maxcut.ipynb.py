get_ipython().run_line_magic("matplotlib", " widget")
edges = [
    (0,1),
    (0,2),
    (1,3),
    (1,4),
    (2,3),
    (3,4)
]
import scipy.linalg as la
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
X = cp.Variable((5,5), symmetric=True)

constraints = [X >> 0]
constraints += [
    X[i, i] == 1 for i in range(5)
]

objective = sum( .5*(1-X[i, j]) for (i, j) in edges)
print(f'objective = {objective}')
prob = cp.Problem(cp.Maximize(objective), constraints)
print(prob.solve())

evals, evects = np.linalg.eigh(X.value)
sdp_vectors = evects.T[evals > float(1.0E-6)].T
print(sdp_vectors)



