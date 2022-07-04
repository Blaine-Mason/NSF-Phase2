import numpy as np
from math import sqrt

def mahalanobis(Shat, S):
    Sscaled = np.linalg.inv(sqrt(S)) * Shat * np.linalg.inv(sqrt(S))
    return Sscaled