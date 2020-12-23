

from network import Network
import numpy as np
from matplotlib import pyplot as plt
from time import time

# read the points from the csv
data = np.genfromtxt('data.csv', delimiter=',')
for point in data:
    point[0] -= 50

# init the network
eta = 0.0003
testnet = Network(0, 0, eta)

# the initial error
print("The initial error, when all parameters are set to 0, is " + str(testnet.average_cost(data)))
input("Press [Enter] to begin training.")

# initial time
t_initial = time()

# train the network
testnet.train(1000, data, True, 0.1)

# final time
t_final = time()

# tell the user how long it took
print("Training took " + str((t_final-t_initial)/60)+" minutes.")

# display the weight and bias after training
print()
print("y = "+str(testnet.weight)+" x + "+str(testnet.bias))
print("The finishing error is " + str(testnet.average_cost(data)))

# make an x and y axis for the points
x_list = [x for x,y in data]
y_list = [y for x,y in data]

# make an x and y axis for the line
xa = np.linspace(min(x_list), max(x_list), max(x_list)-min(x_list))
ya = xa * testnet.weight + testnet.bias
print(xa)

# plot it all
plt.scatter(x_list, y_list, color="blue")
plt.plot(xa, ya, color="red")
plt.show()
