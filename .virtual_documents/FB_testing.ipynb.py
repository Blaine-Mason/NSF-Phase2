import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import scipy.stats
pi = np.pi


plt.plot([0,np.cos(pi/2)],[0,np.sin(pi/2)])
plt.plot([0,np.cos(pi)],[0,np.sin(pi)])
plt.plot([0,np.cos(3*pi/2)],[0,np.sin(3*pi/2)])
plt.plot([0,np.cos(0)],[0,np.sin(0)])


def triangleArea(p1, p2, p3):
    return abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1])+ p3[0]*(p1[1]-p2[1]))/2.0);


def isInside(p1, p2, p3, p):
    area = triangleArea (p1, p2, p3)
    area1 = triangleArea (p, p2, p3)
    area2 = triangleArea (p1, p, p3)
    area3 = triangleArea (p1, p2, p)
    area_sum = area1 + area2 + area3
    if (area_sum - area < 1e-3):
        return True


def theta_iteration(u_samples, a, b, letters):
    colors = list(("ro", "bo", "go", "mo"))
    inside_theta = list()
    inside_sets = list()
    for i in range(a,b):
        plt.clf()
        k = 0
        inside = 0
        plt.plot(U[0,:], U[1,:], "ko", )
        plt.plot([1,-1,-1,1,1], [1,1,-1,-1,1])
        for letter in sets:
            u_left = max(np.abs(np.cos(letter[0][i])),np.abs(np.sin(letter[0][i])))
            u_right = max(np.abs(np.cos(letter[1][i])),np.abs(np.sin(letter[1][i])))
            tri = np.array([[0,0], 
                          [np.cos(letter[0][i])*1/u_left,np.sin(letter[0][i])*1/u_left], 
                          [np.cos(letter[1][i])*1/u_right,np.sin(letter[1][i])*1/u_right]])
            for j in range(0,pts):
                if isInside(tri[1,:], tri[2, :], tri[0,:], (U[0,j],U[1,j])):
                    inside = inside + 1
                    plt.plot(U[0,j], U[1,j], colors[k], markersize=3)
            plt.plot([0,np.cos(letter[0][i])*1/u_left],[0,np.sin(letter[0][i])*1/u_left],colors[k][0])
            plt.plot([0,np.cos(letter[1][i])*1/u_right],[0,np.sin(letter[1][i])*1/u_right], colors[k][0])
            k = k + 1
            inside_sets.append(inside)
        inside_theta.append(inside)
    return inside_sets


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return math.degrees(phi)
thetas = []
error = []
a = 4
b = 1000

for pts in np.arange(a, b, 4):
    sum_ = 0
    sum_error = 0
    for itr in range(0, 40):
        working = True
        while(working):
            try:
                U = np.random.uniform(-1,1,(2,pts))
                U_ = np.zeros(pts)
                for i in range(0,pts):
                    U_[i] = cart2pol(U[0,i], U[1,i])
                    if(U_[i] < 0):
                        U_[i] = U_[i] + 360
                U_ = np.sort(U_)
                U_ = [[U_[i] for i in range(0,pts) if U_[i] > 45 and U_[i] < 135], [U_[i] for i in range(0,pts) if U_[i] > 135 and U_[i] < 225], 
                        [U_[i] for i in range(0,pts) if U_[i] > 225 and U_[i] < 315], [U_[i] for i in range(0,pts) if U_[i] > 315 or U_[i] < 45]]
                U_A = [abs(number - 90) for number in U_[0]]
                U_B = [abs(number - 180) for number in U_[1]]
                U_C = [abs(number - 270) for number in U_[2]]
                U_D = min([abs(number - 360) for number in U_[3] if number < 360], [number for number in U_[3] if number < 45])
                max_theta = max(min(U_A), min(U_B), min(U_C), min(U_D))
            except ValueError:
                continue
            else:
                working = False
        sum_ += max_theta
        sum_error += (8*np.sin(np.radians(90+max_theta/2))*np.sin(np.radians(max_theta/2)) + 4)
    if(pts % 100 == 0):
        print(pts)
    thetas.append(sum_/40)
    error.append(sum_error/40)
print(sum(thetas)/len(thetas))


plt.plot(np.arange(a, b, 4), thetas)
#plt.savefig("dist.png")


from distfit import distfit
thetas = np.array(thetas)
dist = distfit(alpha=0.05, smooth=10)
dist.fit_transform(thetas)
best_distr = dist.model
print(best_distr)
# Ranking distributions
dist.summary

# Plot the summary of fitted distributions
dist.plot()
dist.plot_summary()


def iterate_single_theta(u_samples, a, b, inc_set, co, c):
    colors = list(("ro", "bo", "go", "mo"))
    inside_theta = list()
    inside_sets = list()
    for i in range(a,b):
        plt.clf()
        inside = 0
        plt.plot(U[0,:], U[1,:], "ko")
        plt.plot([1,-1,-1,1,1], [1,1,-1,-1,1])
        u_left = max(np.abs(np.cos(inc_set[0][i])),np.abs(np.sin(inc_set[0][i])))
        u_right = max(np.abs(np.cos(inc_set[1][i])),np.abs(np.sin(inc_set[1][i])))
        tri = np.array([[0,0], 
                      [np.cos(inc_set[0][i])*1/u_left,np.sin(inc_set[0][i])*1/u_left], 
                      [np.cos(inc_set[1][i])*1/u_right,np.sin(inc_set[1][i])*1/u_right]])
        for j in range(0,pts):
            if isInside(tri[1,:], tri[2, :], tri[0,:], (U[0,j],U[1,j])):
                inside = inside + 1
                plt.plot(U[0,j], U[1,j], co, markersize=3)
        plt.plot([0,np.cos(inc_set[0][i])*1/u_left],[0,np.sin(inc_set[0][i])*1/u_left],c)
        plt.plot([0,np.cos(inc_set[1][i])*1/u_right],[0,np.sin(inc_set[1][i])*1/u_right], c)
    return inside


def tuning_theta(u_samples, a, b, sets):
    colors = list(("ro", "bo", "go", "mo"))
    k = 0
    plt.plot(U[0,:], U[1,:], "ko", )
    plt.plot([1,-1,-1,1,1], [1,1,-1,-1,1], "k")
    for letter in sets:
        for i in range(a[k],b[k]):
            inside = 0

            u_left = max(np.abs(np.cos(letter[0][i])),np.abs(np.sin(letter[0][i])))
            u_right = max(np.abs(np.cos(letter[1][i])),np.abs(np.sin(letter[1][i])))
            tri = np.array([[0,0], 
                          [np.cos(letter[0][i])*1/u_left,np.sin(letter[0][i])*1/u_left], 
                          [np.cos(letter[1][i])*1/u_right,np.sin(letter[1][i])*1/u_right]])
            
            for j in range(0,pts):
                if isInside(tri[1,:], tri[2, :], tri[0,:], (U[0,j],U[1,j])):
                    inside = inside + 1
                    plt.plot(U[0,j], U[1,j], colors[k], markersize=3)
            plt.plot([0,np.cos(letter[0][i])*1/u_left],[0,np.sin(letter[0][i])*1/u_left],colors[k][0])
            plt.plot([0,np.cos(letter[1][i])*1/u_right],[0,np.sin(letter[1][i])*1/u_right], colors[k][0])
        k = k + 1


theta_set = list()
pts_set = list()
pts = 30
for runs in range(0, 40):
    sum_ = 0
    for avg in range(0,20):
        scale = .012
        thetas = np.arange(0,pi/2,scale)
        A_left = np.arange(pi/2,pi/4,-scale)
        A_right = np.arange(pi/2,3*pi/4,scale)
        B_left = np.arange(pi,3*pi/4,-scale)
        B_right = np.arange(pi,5*pi/4,scale)
        C_left = np.arange(3*pi/2,5*pi/4,-scale)
        C_right = np.arange(3*pi/2,7*pi/4,scale)
        D_left = np.arange(0,pi/4,scale)
        D_right = np.arange(2*pi,7*pi/4,-scale)
        A = (A_left, A_right)
        B = (B_left, B_right)
        C = (C_left, C_right)
        D = (D_left, D_right)
        sets = list((A, B, C, D))
        U = np.random.uniform(-1,1,(2,pts))
        max_iters = 15
        iteration = 0
        limit = 1
        b_set = [2, 2, 2, 2]
        in_set = [0, 0, 0, 0]
        not_done = True
        while(not_done):
            while(iteration < max_iters):
                if(in_set[0] < limit):
                    b_set[0] += 1
                    in_set[0] = iterate_single_theta(U, 1, b_set[0], sets[0], "ro", "r")
                if(in_set[1] < limit):
                    b_set[1] += 1
                    in_set[1] = iterate_single_theta(U, 1, b_set[1], sets[1], "bo", "b")
                if(in_set[2] < limit):
                    b_set[2] += 1
                    in_set[2] = iterate_single_theta(U, 1, b_set[2], sets[2], "go", "g")
                if(in_set[3] < limit):
                    b_set[3] += 1
                    in_set[3] = iterate_single_theta(U, 1, b_set[3], sets[3], "mo", "m")
                iteration += 1
            if(in_set.count(limit) == len(in_set)):
                not_done = False
            elif(sum(in_set) > 4):
                iteration = 0
                scale = scale - .001
                thetas = np.arange(0,pi/2,scale)
                A_left = np.arange(pi/2,pi/4,-scale)
                A_right = np.arange(pi/2,3*pi/4,scale)
                B_left = np.arange(pi,3*pi/4,-scale)
                B_right = np.arange(pi,5*pi/4,scale)
                C_left = np.arange(3*pi/2,5*pi/4,-scale)
                C_right = np.arange(3*pi/2,7*pi/4,scale)
                D_left = np.arange(0,pi/4,scale)
                D_right = np.arange(2*pi,7*pi/4,-scale)
                A = (A_left, A_right)
                B = (B_left, B_right)
                C = (C_left, C_right)
                D = (D_left, D_right)
                sets = list((A, B, C, D))
                max_iters += 5
                b_set = [2, 2, 2, 2]
                in_set = [0, 0, 0, 0]
                print(f"Max_iter {max_iters}")
                if(max_iters > 30):
                    U = np.random.uniform(-1,1,(2,pts))
            else:
                max_iters += 5
        max_theta = np.degrees(max(thetas[b_set[0]],thetas[b_set[1]],thetas[b_set[2]],thetas[b_set[3]]))
        sum_ = sum_ + max_theta
        print("="*20)
        print(f"Trial: {avg} Max Theta: {max_theta}")
    print("="*20)
    print("="*20)
    print(f"Current pts: {pts}")
    print("="*20)
    print("="*20)
    theta_set.append(sum_/20)
    pts_set.append(pts)
    pts += 5


print(in_set)
b = b_set
a = [b[0]-1, b[1]-1, b[2]-1, b[3]-1]
tuning_theta(U, a, b, sets)
max_theta = np.degrees(max(thetas[b_set[0]],thetas[b_set[1]],thetas[b_set[2]],thetas[b_set[3]]))
theta.append(max_theta)
print(f"Largest theta: {max_theta} Volume of error: {8*np.sin(np.radians(90-max_theta))*np.sin(np.radians(max_theta))}")


thetas = np.arange(0.05,pi/2,.00001)
def summation(thetas):
    return 4/((4-0)*np.tan(thetas)) + 4/((4-1)*np.tan(thetas)) + 4/((4-2)*np.tan(thetas)) + 4/((4-3)*np.tan(thetas))


plt.plot(thetas, [summation(theta) for theta in thetas])
plt.show()



