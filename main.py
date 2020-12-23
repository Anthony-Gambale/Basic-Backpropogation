

from network import Network
import numpy as np
from matplotlib import pyplot as plt

# read the points from the csv
data = np.genfromtxt('data.csv', delimiter=',')
for x in data:
	x[1] += 500

# init the network
eta = 0.0003
testnet = Network(0, 0, eta)

# the initial error
print("The initial error, when all parameters are set to 0, is " + str(testnet.average_cost(data)))
input("Press [Enter] to begin training.")

# train the network
testnet.train(100000, data)

# display the weight and bias after training
print()
print("y = "+str(testnet.weight)+" x + "+str(testnet.bias))
print("The finishing error is " + str(testnet.average_cost(data)))

# make an x and y axis for the points
x_list = [a[0] for a in data]
y_list = [a[1] for a in data]

# make an x and y axis for the line
xa = np.linspace(0, 100, len(data))
ya = xa * testnet.weight + testnet.bias

# plot it all
plt.scatter(x_list, y_list, color="blue")
plt.plot(xa, ya, color="red")
plt.plot(0, 0, color="black")
plt.show()
