import numpy as np
import matplotlib.pyplot as plt
import math
from shapely.geometry import Polygon
from tqdm import tqdm


def support(X, val, q): # Set q to be the qth quantile
    return np.quantile((X.T).dot(val), q)

def compute_fb(X, q, n):
    polar_body = np.ones((n,0))
    for i in range(X.shape[1]):
        if support(X, X[:,i] / la.norm(X[:,i]), q) > np.dot(X[:,i], X[:,i]/la.norm(X[:,i])): 
            polar_body = np.hstack((polar_body, np.array(X[:,i]).reshape(n,1)))
    return polar_body


def support_theta(X, val, q): # Set q to be the qth quantile
    return np.quantile(np.dot(X,val), q)

def compute_fb_theta(X, q, n, eps):
    polar_body = np.ones((n,0))
    thetas = [0.02, 90, 180, 270]
    thetas = [thetas[i] + eps for i in range(4)]
    directions = [[np.cos(np.radians(thetas[i])), np.sin(np.radians(thetas[i]))] for i in range(4)]
    for j in range(4):
        for i in range(X.shape[1]):
            if support_theta(directions[j], X[:,i] / la.norm(X[:,i]), q) > np.dot(X[:,i], X[:,i]/la.norm(X[:,i])): 
                polar_body = np.hstack((polar_body, np.array(X[:,i]).reshape(n,1)))
    return polar_body



def ret_supp(point, ret_vals):
    
    if point[0] == 0:
        m = 0
    else:
        m = -math.pow(point[1]/point[0],-1)
    #print(m)
    b = point[1] + -m*point[0]
   # print(b)
    #print(f'y = {m}x + {b}')
    x = np.arange(-3,3,.01)
    y = m*x + b
    if ret_vals:
        return m, b
    else:
        return x, y
def ret_int(m1, b1, m2, b2):
    xi = (b1-b2) / (m2-m1)
    yi = m1 * xi + b1

    return xi, yi
def isInside(p1, p2, p3, p):
    area = triangleArea (p1, p2, p3)
    area1 = triangleArea (p, p2, p3)
    area2 = triangleArea (p1, p, p3)
    area3 = triangleArea (p1, p2, p)
    area_sum = area1 + area2 + area3
    if (area_sum - area < 1e-3):
        return True
def triangleArea(p1, p2, p3):
    return abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1])+ p3[0]*(p1[1]-p2[1]))/2.0);


def square_approx(theta):
    print(f"Theta: {theta}")
    A_left = np.radians(90+theta/2)
    A_right = np.radians(90-theta/2)
    B_left = np.radians(180+theta/2)
    B_right = np.radians(180-theta/2)
    C_left = np.radians(270+theta/2)
    C_right = np.radians(270-theta/2)
    D_left = np.radians(theta/2)
    D_right = np.radians(360-theta/2)
    A = (A_left, A_right)
    B = (B_left, B_right)
    C = (C_left, C_right)
    D = (D_left, D_right)
    sets = list((A, B, C, D))
    U = np.random.uniform(-1,1,(2,3))
    trials = 0
    for thresh in range(1000):
        trials += 1
        U = np.hstack((U, np.random.uniform(-1,1,(2,1))))
        count = 0
        js = []
        for letter in sets:
            u_left = max(np.abs(np.cos(letter[0])),np.abs(np.sin(letter[0])))
            u_right = max(np.abs(np.cos(letter[1])),np.abs(np.sin(letter[1])))

            tri = np.array([[0,0], 
                        [np.cos(letter[0])*1/u_left,np.sin(letter[0])/u_left], 
                        [np.cos(letter[1])*1/u_right,np.sin(letter[1])/u_right]])
            for j in range(0,3+trials):
                if isInside(tri[1,:], tri[2, :], tri[0,:], (U[0,j],U[1,j])):
                    js.append(j)
                    count +=1
                    break


        if count == 4:
            break
    plt.plot(U[0,:], U[1,:], 'ko')
    plt.plot([U[0,i] for i in js], [U[1,i] for i in js], 'ro')

    for letter in sets:
            u_left = max(np.abs(np.cos(letter[0])),np.abs(np.sin(letter[0])))
            u_right = max(np.abs(np.cos(letter[1])),np.abs(np.sin(letter[1])))

            plt.plot([0,np.cos(letter[0])*1/u_left],[0,np.sin(letter[0])*1/u_left], 'r')
            plt.plot([0,np.cos(letter[1])*1/u_right],[0,np.sin(letter[1])*1/u_right], 'r')
    plt.show()
    print("="*20)
    print("="*20)
    print(f"Theta: {theta}")
    print("="*20)
    print("="*20)
    print(f"Current pts: {U.shape[1]}")
    print("="*20)
    print("="*20)
    print(f"Gamma_theta: {(np.tan(np.radians(theta/2)))}")
    p = (np.tan(np.radians(theta/2)))
    N = 4
    print(f"N = {N}")
    print(f"E[T_theta]: {(N+1)/p*(np.log(N))}")
    print("="*20)
    print(f"Error = 2*sin(90+theta)sin(theta) = {2*np.sin(90+(np.radians(theta/2)))*np.sin(np.radians(theta/2))}")
    print("="*20)
    n = U.shape[1]
    plt.show()
    error = []
    eps = theta
    thetas = [0.02, 90, 180, 270]
    thetas = [thetas[i] + eps for i in range(4)]
    directions = [[np.cos(np.radians(thetas[i])), np.sin(np.radians(thetas[i]))] for i in range(4)]
    colors = np.array([[np.dot(U[:,i], directions[j]) for i in range(n)] for j in range(4)])
    scales = [np.dot(U[:,np.argmax(colors[i])], directions[i]) for i in range(4)]
    #print(scales)
    supports = np.array([ret_supp([directions[i][0]*scales[i], directions[i][1]*scales[i]], False) for i in range(4)])
    #m_b = np.array([ret_supp([directions[i][0]*scales[i], directions[i][1]*scales[i]], True) for i in range(4)])
    m_b = np.array([ret_supp([directions[i][0]*scales[i], directions[i][1]*scales[i]], True) for i in range(4)])
    print([U[:,np.argmax(colors[i])] for i in range(4)])
    print(m_b)

    points_of_i = []
    for i in range(4):
        if i < 3:
            points_of_i.append(ret_int(m_b[i][0], m_b[i][1], m_b[i+1][0], m_b[i+1][1]))
        else:
            points_of_i.append(ret_int(m_b[i][0], m_b[i][1], m_b[0][0], m_b[0][1]))
    #print(points_of_i)
    pgon = Polygon(points_of_i) # Assuming the OP's x,y coordinates
    error.append(pgon.area)

    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.scatter(U[0,:], U[1,:], c=colors[0], cmap="RdYlGn", s=40, edgecolors="black");
    #plt.plot([0,directions[0][0]*(scales[0]+2)], [0, directions[0][1]*(scales[0]+2)], "k", linewidth=4)

    for i in range(4):
        plt.plot(supports[i][0], supports[i][1], "k", linewidth=4)
    for i in range(len(points_of_i)):
        plt.plot(points_of_i[i][0],points_of_i[i][1], "rs")
    plt.colorbar()
    plt.show()
    return U, js


theta = 5
U, inxs = square_approx(theta)



