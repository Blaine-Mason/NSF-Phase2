import numpy as np
import matplotlib.pyplot as plt
import math
from shapely.geometry import Polygon
from tqdm import tqdm


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


n = 103
error = []
eps = 1
#print(eps)
U_x = np.random.uniform(-1,1,(1,n))
U_y = np.random.uniform(-1,1,(1,n))
U = np.vstack((U_x,U_y))
thetas = [0.02, 90, 180, 270]
thetas = [thetas[i] + eps for i in range(4)]
directions = [[np.cos(np.radians(thetas[i])), np.sin(np.radians(thetas[i]))] for i in range(4)]
plt.plot(directions, "bo")
plt.show()
colors = np.array([[np.dot(U[:,i], directions[j]) for i in range(n)] for j in range(4)])
print(colors)
scales = [np.dot(U[:,np.argmax(colors[i])], directions[i]) for i in range(4)]
#print(scales)
supports = np.array([ret_supp([directions[i][0]*scales[i], directions[i][1]*scales[i]], False) for i in range(4)])
#m_b = np.array([ret_supp([directions[i][0]*scales[i], directions[i][1]*scales[i]], True) for i in range(4)])
m_b = np.array([ret_supp([directions[i][0]*scales[i], directions[i][1]*scales[i]], True) for i in range(4)])
print(supports)
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

# plt.xlim([-2,2])
# plt.ylim([-2,2])
# plt.scatter(U[0,:], U[1,:], c=colors[0], cmap="RdYlGn", s=40, edgecolors="black");
# #plt.plot([0,directions[0][0]*(scales[0]+2)], [0, directions[0][1]*(scales[0]+2)], "k", linewidth=4)

# for i in range(4):
#     plt.plot(supports[i][0], supports[i][1], "k", linewidth=4)
# for i in range(len(points_of_i)):
#     plt.plot(points_of_i[i][0],points_of_i[i][1], "rs")
# plt.colorbar()
# plt.show()



def support(X, val, q): # Set q to be the qth quantile
    return np.quantile((X.T).dot(val), q)

def compute_fb(X, q, n):
    polar_body = np.ones((n,0))
    for i in range(X.shape[1]):
        if support(X, X[:,i] / la.norm(X[:,i]), q) > np.dot(X[:,i], X[:,i]/la.norm(X[:,i])): 
            polar_body = np.hstack((polar_body, np.array(X[:,i]).reshape(n,1)))
    return polar_body


test = np.random.uniform(-1,1,(2, 103))
test_fb = compute_fb(test, .95, 2)
plt.plot(test[0,:], test[1,:], "bo")
plt.plot(test_fb[0,:], test_fb[1,:], "ro")
plt.show()


plt.plot([x for x in np.arange(0, 45, .01)], [8*np.sin(np.radians(90+x))*np.sin(np.radians(x))+4 for x in np.arange(0, 45, .01)])
plt.plot([x for x in np.arange(0, 46, 1)], error, 'r--')
plt.xlabel("Theta")
plt.ylabel("Error")
plt.title("Error Function Approx vs Theoretical")
plt.savefig("ErrorApprox.png")


n = 800
U_x = np.random.uniform(-1,1,(1,n))
U_y = np.random.uniform(-1,1,(1,n))
U = np.vstack((U_x,U_y))
support_x = []
support_y = []
for theta in range(0, 361):
    direction = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
    colors = [np.dot(U[:,i], direction) for i in range(n)]
    scale = np.dot(U[:,np.argmax(colors)], direction)
    support_x.append(direction[0]*scale)
    support_y.append(direction[1]*scale)
support = np.vstack((support_x, support_y))
plt.title(f"Support")
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.scatter(U[0,:], U[1,:], c="red");
plt.plot(support[0,:], support[1,:], "--", linewidth=3)
plt.show()


import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as la


pi0 = folder['pi0']
mu0 = folder['mu0']
sigma0 = folder['sigma0']

print(f'Check the shape of pi0 = ({pi0.shape}), mu0 = ({mu0.shape}), sigma0 = ({sigma0.shape})')
print(f'pi0: \n{pi0} \n mu0: \n{mu0}\n sigma0: \n{sigma0}')


from scipy.stats import multivariate_normal

def E_step(data, pi, mu, sigma):
    N = data.shape[0]
    K = pi.shape[0]
    d = mu.shape[0]
    gamma = np.zeros((N,K))
    for nk in range(K):
        gamma[:, nk] = pi[nk] * multivariate_normal.pdf(data, mean=mu[nk], cov=sigma[nk])
    gamma = gamma/np.sum(gamma, axis=1, keepdims=True)
    
    return gamma


def M_step(data, gamma):
    N, D = data.shape
    K = gamma.shape[1]
    pi = np.zeros(K)
    mu = np.zeros((K,D))
    sigma = np.zeros((K, D, D))
    
    for ik in range(K):
        n_k = gamma[:, ik].sum()
        pi[ik] = n_k/N
        elements = np.reshape(gamma[:, ik], (gamma.shape[0], 1))
        mu[ik, :] = (np.multiply(elements, data)).sum(axis=0) / n_k
        sigma_sum = 0
        for i in range(N):
            var = data[i] - mu[ik]
            sigma_sum = sigma_sum + gamma[i, ik] * np.outer(var, var)
        sigma[ik, :] = sigma_sum/n_k
    return pi, mu, sigma


def elbo(data, gamma, pi, mu, sigma):
    N = data.shape[0]
    K = gamma.shape[1]
    d = data.shape[1]
    
    loss = 0
    for i in range(N):
        x = data[i]
        for k in range(K):
            pos_dist = gamma[i, k]
            log_lik = np.log(multivariate_normal.pdf(x, mean=mu[k, :], cov=sigma[k, :, :] + 1e-20))
            log_q = np.log(gamma[i, k] + 1e-20)
            log_pz = np.log(pi[k] + 1e-20)
            loss = (loss + np.multiply(pos_dist, log_pz) + np.multiply(pos_dist, log_lik) + np.multiply(pos_dist, -log_q))
    return loss


def train_loop(data, K, tolerance=1e-9, max_iter=50,restart=10):
    N, d = data.shape
    elbo_best = -np.inf
    pi_best = None
    mu_best = None
    sigma_best = None
    gamma_f = None
    for _ in range(restart):
        pi = np.ones(K) / K
        mu = np.random.rand(K, d)
        sigma = np.tile(np.eye(d), (K, 1, 1))
        last_iter_loss = None
        all_losses = []
        try: 
            
            for i in range(max_iter):
                gamma = E_step(data, pi, mu, sigma)
                pi, mu, sigma = M_step(data, gamma)
                loss = elbo(data, gamma, pi, mu, sigma)
                if loss > elbo_best:
                    elbo_best = loss
                    pi_best = pi
                    mu_best = mu
                    sigma_best = sigma
                    gamma_f = gamma
                if last_iter_loss and abs((loss - last_iter_loss)/last_iter_loss) < tolerance:
                    break
                last_iter_loss = loss
                all_losses.append(loss)
        except np.linalg.LinAlgError:
            pass
    return elbo_best, pi_best, mu_best, sigma_best, all_losses, gamma_f


best_loss, pi_best, mu_best, sigma_best, ls_lst, final_posterior = train_loop(data, 3)


print ("starting loss, best_loss: ", ls_lst[0], ',',  best_loss)
print ("best pi", pi_best) 
print ("best mu: ", mu_best)

print ("best sigma: ", sigma_best)


folder = np.load("samples.npz")
data = folder['data']
fig = plt.figure(figsize=(6,4))
plt.plot(data[:, 0], data[:, 1], linestyle='None', marker='o', markersize=5, color='navy')
plt.xlabel('data[:, 0]', fontsize=12)
plt.ylabel('data[:, 1]', fontsize=12)
plt.show()

noisy = data.copy()
for i in range(100):
    scale = np.random.standard_cauchy()
    random_point = np.random.randint(0, data.shape[0])
    noisy[random_point, 0] += scale
    noisy[random_point, 1] += scale
plt.plot(noisy[:, 0], noisy[:, 1], linestyle='None', marker='o', markersize=5, color='red')
plt.xlabel('data[:, 0]', fontsize=12)
plt.ylabel('data[:, 1]', fontsize=12)
plt.show()

data_fb = compute_fb((noisy - np.mean(0)).T, .95, 2)
plt.plot(data_fb[0, :], data_fb[1, :], linestyle='None', marker='o', markersize=5, color='red')
plt.xlabel('data[:, 0]', fontsize=12)
plt.ylabel('data[:, 1]', fontsize=12)
plt.show()


def plot_contours(data, means, covs):
    """visualize the gaussian components over the data"""
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], 'ko')

    delta = 0.025
    k = means.shape[0]
    x = np.arange(-2.5, 10.0, delta)
    y = np.arange(-2.5, 10.0, delta)
    x_grid, y_grid = np.meshgrid(x, y)
    coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T

    col = ['green', 'red', 'indigo']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        z_grid = multivariate_normal(mean, cov).pdf(coordinates).reshape(x_grid.shape)
        plt.contour(x_grid, y_grid, z_grid, colors = col[i])
    plt.tight_layout()


from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, n_init=10)
gmm.fit(data)
fig= plt.figure(figsize=(14, 12))
plot_contours(data, gmm.means_, gmm.covariances_)
plt.xlabel("data[:, 0]", fontsize=12)
plt.ylabel("data[:, 1]", fontsize=12)
plt.show()
print(f"Original Mean: {gmm.means_}")
print(f"Original Cov: {gmm.covariances_}")

plt.plot(noisy[:, 0], noisy[:, 1], linestyle='None', marker='o', markersize=5, color='red')
plt.xlabel('data[:, 0]', fontsize=12)
plt.ylabel('data[:, 1]', fontsize=12)
plt.show()

gmm.fit(data_fb.T)
fig= plt.figure(figsize=(14, 12))
plot_contours(data, gmm.means_, gmm.covariances_)
plt.xlabel("data[:, 0]", fontsize=12)
plt.ylabel("data[:, 1]", fontsize=12)
plt.show()
print(f"New Mean: {gmm.means_}")
print(f"New Cov: {gmm.covariances_}")


n = 103
U_x = np.random.uniform(-1,1,(1,n))
U_y = np.random.uniform(-1,1,(1,n))
U = np.vstack((U_x,U_y))
theta = 50
direction = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
colors = [np.dot(U[:,i], direction) for i in range(n)]
scale = np.dot(U[:,np.argmax(colors)], direction)
print(U[:,np.argmax(colors)])
plt.title(f"Theta = {theta}")
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.scatter(U[0,:], U[1,:], c=colors, cmap="RdYlGn", s=40, edgecolors="black");
plt.plot([0,direction[0]*(scale+.5)], [0, direction[1]*(scale+.5)], "k", linewidth=4)
plt.plot(direction[0]*scale, direction[1]*scale, "bs")
plt.colorbar()

    


def support(X, val, q): # Set q to be the qth quantile
    return np.quantile((X.T).dot(val), q)

def compute_fb(X, q, n):
    polar_body = np.ones((n,0))
    for i in range(X.shape[1]):
        if support(X, X[:,i] / la.norm(X[:,i]), q) > np.dot(X[:,i], X[:,i]/la.norm(X[:,i])): 
            polar_body = np.hstack((polar_body, np.array(X[:,i]).reshape(n,1)))
    return polar_body


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


inside = 0

scale = .012
points = []
coupons = []
for theta in range(20, 90):
    print(f"Theta: {theta}")
    thetas_left = np.radians(90+theta/2)
    thetas_right = np.radians(90-theta/2)
    points_sum  = 0
    for pts in range(10000):
        U = np.random.uniform(-1,1,(2,pts))
        #plt.plot(U[0,:], U[1,:], "ko")
        #plt.plot([1,-1,-1,1,1], [1,1,-1,-1,1])
        u_left = max(np.abs(np.cos(thetas_left)),np.abs(np.sin(thetas_left)))
        u_right = max(np.abs(np.cos(thetas_right)),np.abs(np.sin(thetas_right)))
        tri = np.array([[0,0], 
                      [np.cos(thetas_left)*1/u_left,np.sin(thetas_left)/u_left], 
                      [np.cos(thetas_right)*1/u_right,np.sin(thetas_right)/u_right]])
        for j in range(0,pts):
            if isInside(tri[1,:], tri[2, :], tri[0,:], (U[0,j],U[1,j])):
                inside = inside + 1
                #plt.plot(U[0,j], U[1,j], 'ro', markersize=3)
        #plt.plot([0,np.cos(thetas_left)*1/u_left],[0,np.sin(thetas_left)*1/u_left], 'r')
        #plt.plot([0,np.cos(thetas_right)*1/u_right],[0,np.sin(thetas_right)*1/u_right], 'r')
        if(inside == math.ceil(1/(np.tan(np.radians(theta/2))/4)*np.log(1/(np.tan(np.radians(theta/2))/4)))):
            print(f"Samples: {pts}")
            points_sum += pts
            print(f"inside: {inside}")
            inside = 0
            break
        else:
            inside = 0
    points.append(points_sum)  
    coupons.append(math.ceil(1/(np.tan(np.radians(theta/2))/4)*np.log(1/(np.tan(np.radians(theta/2))/4))))



plt.plot(coupons, points)


import scipy.integrate as integrate


inside = 0

scale = .012
points = []
coupons = []
for theta in range(1, 91):
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
    for pts in range(10000):
        U = np.random.uniform(-1,1,(2,pts))
        count = 0
        js = []
        for letter in sets:
            u_left = max(np.abs(np.cos(letter[0])),np.abs(np.sin(letter[0])))
            u_right = max(np.abs(np.cos(letter[1])),np.abs(np.sin(letter[1])))
            
            tri = np.array([[0,0], 
                          [np.cos(letter[0])*1/u_left,np.sin(letter[0])/u_left], 
                          [np.cos(letter[1])*1/u_right,np.sin(letter[1])/u_right]])
            for j in range(0,pts):
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
    points.append(pts)  
    coupons.append(math.ceil(1/(np.tan(np.radians(theta/2))/4)*np.log(1/(np.tan(np.radians(theta/2))/4))))
    print("="*20)
    print("="*20)
    print(f"Theta: {theta}")
    print("="*20)
    print("="*20)
    print(f"Current pts: {pts}")
    print("="*20)
    print("="*20)
    print(f"tan(theta/2): {(np.tan(np.radians(theta/2)))}")
    p = (np.tan(np.radians(theta/2)))
    N = 4
    print(f"simp = {(np.log(4)*(np.log(1/.4)))/p}")
    print(f"N = {N}")
    EX = ((N)*np.log(N))
    print(f"EX = nlog(n): {EX}")
    print(f"Probability: {p}")
    print(f"Samples: {((N)*np.log(N))/(p)}")
    print("="*20)
    print("="*20)


inside = 0
num_tests = 100
scale = .012
points = []
coupons = []
for theta in range(1, 91):
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
    sum_of_pts = 0
    for test in range(num_tests):
        for thresh in range(100000):
            U = np.random.uniform(-1,1,(2,4))
            count = 0
            js = []
            for letter in sets:
                u_left = max(np.abs(np.cos(letter[0])),np.abs(np.sin(letter[0])))
                u_right = max(np.abs(np.cos(letter[1])),np.abs(np.sin(letter[1])))

                tri = np.array([[0,0], 
                              [np.cos(letter[0])*1/u_left,np.sin(letter[0])/u_left], 
                              [np.cos(letter[1])*1/u_right,np.sin(letter[1])/u_right]])
                for j in range(0,pts):
                    if isInside(tri[1,:], tri[2, :], tri[0,:], (U[0,j],U[1,j])):
                        js.append(j)
                        count +=1
                        break


            if count == 4:
                break
        sum_of_pts += pts
        #plt.plot(U[0,:], U[1,:], 'ko')
        #plt.plot([U[0,i] for i in js], [U[1,i] for i in js], 'ro')
        #for letter in sets:
                #u_left = max(np.abs(np.cos(letter[0])),np.abs(np.sin(letter[0])))
                #u_right = max(np.abs(np.cos(letter[1])),np.abs(np.sin(letter[1])))

                #plt.plot([0,np.cos(letter[0])*1/u_left],[0,np.sin(letter[0])*1/u_left], 'r')
                #plt.plot([0,np.cos(letter[1])*1/u_right],[0,np.sin(letter[1])*1/u_right], 'r')
        #plt.show()
    points.append(sum_of_pts/num_tests)  
    coupons.append(math.ceil(1/(np.tan(np.radians(theta/2)))*np.log(1/(np.tan(np.radians(theta/2))))))
    print("="*20)
    print("="*20)
    print(f"Theta: {theta}")
    print("="*20)
    print("="*20)
    print(f"Current pts: {pts}")
    print("="*20)
    print("="*20)
    print(f"2*tan(theta/2): {(np.tan(np.radians(theta/2)))}")
    p = (np.tan(np.radians(theta/2)))
    N = 4
    print(f"N = {N}")
    print(f"Probability: {p}")
    print(f"Samples: {(N+1)/p*(np.log(N))}")
    print("="*20)
    print("="*20)


plt.plot([i for i in range(1,91)], points)
plt.plot([i for i in range(1,91)], [(N+1)/p*(np.log(N)) for p in ps])


ps = [(np.tan(np.radians(theta/2))) for theta in range(1,91)]


ps


import random
no_simulation=10000
total_coupon_numbers=4
theta_results = []
for theta in range(1,91):
    print(theta)
    coupon_number_set=set()
    flag=False
    boxes_no=0
    total_attempts = 0
    for i in range(no_simulation):
        counter=0
        coupon_number_set=set()
        flag=False
        num_of_attempts = 0
        while not flag:
            num_of_attempts += 1
            if (random.randint(1,100) <= round((np.tan(np.radians(theta/2))), 2)*100):
                coupon=random.randint(1,total_coupon_numbers)     
                counter+=1
                if len(coupon_number_set)==total_coupon_numbers:
                    total_attempts += num_of_attempts
                    boxes_no+=counter
                    flag=True;

                    #print boxes_no

                else:
                    #print coupon_number_set
                    coupon_number_set.add(coupon)
        
    theta_results.append(total_attempts/no_simulation)
print(boxes_no)           
avg_boxes= (boxes_no * 1.0)/no_simulation
print(f'avg number of boxes to make a complete set: {avg_boxes}')


def bound(n, delta, theta):
    ep = 2*np.sin(np.radians(90-theta/2))*np.sin(np.radians(theta/2))
    return (math.pi * n * n)/(6*delta*ep*ep)


inside = 0
num_tests = 100
scale = .012
attempts = []
for theta in range(1, 91):
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
    sum_of_pts = 0
    for test in range(num_tests):
        trials = 0
        U = np.random.uniform(-1,1,(2,3))
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
        sum_of_pts += trials
    attempts.append(sum_of_pts/num_tests) 
    #plt.plot(U[0,:], U[1,:], 'ko')
    #plt.plot([U[0,i] for i in js], [U[1,i] for i in js], 'ro')
    #for letter in sets:
            #u_left = max(np.abs(np.cos(letter[0])),np.abs(np.sin(letter[0])))
            #u_right = max(np.abs(np.cos(letter[1])),np.abs(np.sin(letter[1])))

            #plt.plot([0,np.cos(letter[0])*1/u_left],[0,np.sin(letter[0])*1/u_left], 'r')
            #plt.plot([0,np.cos(letter[1])*1/u_right],[0,np.sin(letter[1])*1/u_right], 'r')
    #plt.show()
    print("="*20)
    print("="*20)
    print(f"Theta: {trials}")
    print("="*20)
    print("="*20)
    print(f"Current pts: {pts}")
    print("="*20)
    print("="*20)
    print(f"2*tan(theta/2): {(np.tan(np.radians(theta/2)))}")
    p = (np.tan(np.radians(theta/2)))
    N = 4
    print(f"N = {N}")
    print(f"Probability: {p}")
    print(f"Samples: {(N+1)/p*(np.log(N))}")
    print("="*20)
    print("="*20)


delta = .0001
N=4
plt.plot([i for i in range(1,91)], attempts, "r", label="emp trials")
plt.plot([i for i in range(1,91)], [np.ceil((1/(p)))*N*(np.log(N)) for p in ps], label="1/p*N*log(N)")
#plt.plot([i for i in range(1,91)], [(1/p)*(np.log(1/p)) for p in ps], label="1/plog(1/p)")
plt.plot([i for i in range(1,91)], theta_results, "g", label="cpp trials")
plt.plot([i for i in range(1, 91)], [(np.log(1/p) + np.log(1/delta))/p for p in ps], "y", label="(log(n) + log(1/delta))/p")
#plt.plot([i for i in range(1, 91)], [((1/p)**2*math.pi**2)/(6*(.99**2)*.999) for p in ps], "y", label="(log(n) + log(1/delta))/p")
plt.legend()


[((1/p)**2*math.pi**2)/(6*(.9**2)*.9) for p in ps]



