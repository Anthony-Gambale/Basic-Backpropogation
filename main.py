

from network import *
import numpy as np
from matplotlib import pyplot as plt

# read the points from the csv
data = np.genfromtxt('data.csv', delimiter=',')
for x in data:
	x[1] += 500

# init the network
eta = 0.000361
testnet = Network(0, 0, eta)

# train the network
testnet.train(100000, data)

# display the weight and bias after training
print("y = "+str(testnet.weight)+" x + "+str(testnet.bias))

# make an x and y axis for the points
x = [a[0] for a in data]
y = [a[1] for a in data]

# make an x and y axis for the line
xa = np.linspace(0, 100, len(data))
ya = xa * testnet.weight + testnet.bias

# plot it all
plt.scatter(x, y, color="blue")
plt.plot(xa, ya, color="red")
plt.show()
