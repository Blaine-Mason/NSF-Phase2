import numpy as np
import matplotlib.pyplot as plt


n = 500
U_x = np.random.uniform(-7,7,(1,n))
U_y = np.random.uniform(-3,3,(1,n))
U = np.vstack((U_x,U_y))
theta = 220
direction = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
colors = [np.dot(U[:,i], direction) for i in range(n)]
scale = np.dot(U[:,np.argmax(colors)], direction)
plt.title(f"Theta = {theta}")
plt.xlim([-10,10])
plt.ylim([-10,10])
plt.scatter(U[0,:], U[1,:], c=colors, cmap="RdYlGn", s=40, edgecolors="black");
plt.plot([0,direction[0]*(scale+2)], [0, direction[1]*(scale+2)], "k", linewidth=4)
plt.plot(direction[0]*scale, direction[1]*scale, "bs")
plt.colorbar()
plt.show()


n = 800
U_x = np.random.uniform(-7,7,(1,n))
U_y = np.random.uniform(-3,3,(1,n))
U = np.vstack((U_x,U_y))
for theta in range(0, 361):
    direction = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
    colors = [np.dot(U[:,i], direction) for i in range(n)]
    scale = np.dot(U[:,np.argmax(colors)], direction)
    plt.title(f"Theta = {theta}")
    plt.xlim([-10,10])
    plt.ylim([-10,10])
    plt.scatter(U[0,:], U[1,:], c=colors, cmap="RdYlGn", s=40, edgecolors="black");
    plt.plot([0,direction[0]*(scale+2)], [0, direction[1]*(scale+2)], "k", linewidth=4)
    plt.plot(direction[0]*scale, direction[1]*scale, "bs")
    plt.colorbar()
    plt.savefig(f"images/theta_{theta}")
    plt.clf()
    


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



