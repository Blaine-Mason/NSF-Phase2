import numpy as np
import matplotlib.pyplot as plt

# Project a point onto a direction
def project(point, direction):
    return point - np.dot(point, direction) * direction 

def main():
    # Create a point
    point = np.random.uniform(0,1,(2, 100))
    # Create a direction
    direction = np.array([1, 1])
    # Project the point onto the direction
    plt.plot(point[0,:], point[1,:], 'o')
    plt.plot([0, direction[0]], [0, direction[1]], 'r-')
    plt.show()
    for i in range(100):
        projected = project(point[:,i], direction)
        plt.plot(projected[0], projected[1], 'bx')
    plt.show()  

if __name__ == '__main__':
    main()
