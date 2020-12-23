

from network import Network
import numpy as np
from matplotlib import pyplot as plt
from time import time

# read the points from the csv
data = np.genfromtxt('data.csv', delimiter=',')
for x, y in data:
    y *= 0.1
    y += -100

# init the network
eta = 0.0003
testnet = Network(0, 0, eta)

# the initial error
print("The initial error, when all parameters are set to 0, is " + str(testnet.average_cost(data)))
input("Press [Enter] to begin training.")

# initial time
t_initial = time()

# train the network
testnet.train(10, data, True, 0.5)

# final time
t_final = time()

# tell the user how long it took
print("Training took " + str((t_final-t_initial)/60)+" minutes.")

# display the weight and bias after training
print()
print("y = "+str(testnet.weight)+" x + "+str(testnet.bias))
print("The finishing error is " + str(testnet.average_cost(data)))

# make an x and y axis for the points
x_list = [a[0] for a in data]
y_list = [a[1] for a in data]

# make an x and y axis for the line
xa = np.linspace(0, 100, 100)
ya = xa * testnet.weight + testnet.bias

# make a black line for the y axis and a black line for the x axis to force the window to be a certain size
xx = np.linspace(-50, 150, 100)
yx = xa * 0

yy = np.linspace(-50, 150, 100)
xy = yy * 0

# plot it all
plt.scatter(x_list, y_list, color="blue")
plt.plot(xa, ya, color="red")
plt.plot(xx, yx, color="black", linestyle="--")
plt.plot(xy, yy, color="black", linestyle="--")
plt.show()
