import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd


def ell_1(n):
    U = np.random.uniform(0,1,n)
    tmp = np.array([0,1])
    Final = np.zeros(n)
    U = np.hstack((U,tmp))
    U = np.sort(U)
    for i in range(1,n+1):
        Final[i-1] = U[i] - U[i-1]
    Final = np.reshape(Final, (n,1))
    for j in range(n):
        binom = np.random.binomial(1,.5)
        if(binom == 1):
            Final[j] = Final[j]*(-1)
    return Final


n = 2
unit_vectors = np.array([[0,1,0,-1,0,-1,0,1], [0,1,0,1,0,-1,0,-1]])
e1 = np.array([[0],[1]])
plt.plot(unit_vectors[0], unit_vectors[1])
t = 30
A_left = np.radians(45+t/2)
A_right = np.radians(45-t/2)
B_left = np.radians(135+t/2)
B_right = np.radians(135-t/2)
C_left = np.radians(225+t/2)
C_right = np.radians(225-t/2)
D_left = np.radians(315+t/2)
D_right = np.radians(315-t/2)
A = (A_left, A_right)
B = (B_left, B_right)
C = (C_left, C_right)
D = (D_left, D_right)
sets = list((A, B, C, D))
for letter in sets:
            u_left = max(np.abs(np.cos(letter[0])),np.abs(np.sin(letter[0])))
            u_right = max(np.abs(np.cos(letter[1])),np.abs(np.sin(letter[1])))

            plt.plot([0,np.cos(letter[0])*1/u_left],[0,np.sin(letter[0])*1/u_left], 'r')
            plt.plot([0,np.cos(letter[1])*1/u_right],[0,np.sin(letter[1])*1/u_right], 'r')
samples = 1
S = np.zeros((n, 1))
for i in range(samples):
    S = np.hstack((S, ell_1(n)))
S = np.delete(S, 0,1)
theta = t
print(np.dot(S[:,0], e1)/la.norm(S[:,0]))
print(np.degrees(np.arccos(np.dot(S[:,0], e1)/la.norm(S[:,0]))))
print(theta/2)
if np.degrees(np.arccos(np.dot(S[:,0], e1)/la.norm(S[:,0]))) < theta/2:
    plt.plot(S[0,:], S[1,:],"ro")
else:
    plt.plot(S[0,:], S[1,:],"bo")


np.dot(S[:,0], e1)


la.norm([0,1])


la.norm(S[:,0])


print(np.arccos(np.dot(S[:,0], [0,1])/la.norm(S[:,0])))


np.radians(90)/2



