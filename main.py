

from network import Network
import numpy as np
from matplotlib import pyplot as plt

# read the points from the csv
data = np.genfromtxt('data.csv', delimiter=',')
for x in data:
	x[1] -= 100

# init the network
eta = 0.00035
testnet = Network(0, 0, eta)

# train the network
testnet.train(100000, data)

# display the weight and bias after training
print("y = "+str(testnet.weight)+" x + "+str(testnet.bias))

# make an x and y axis for the points
x_list = [a[0] for a in data]
y_list = [a[1] for a in data]

# centre values
x_c = [(max(x_list) + min(x_list)) / 2]
y_c = [(max(y_list) + min(y_list)) / 2]

# make an x and y axis for the line
xa = np.linspace(0, 100, len(data))
ya = xa * testnet.weight + testnet.bias

# plot it all
plt.scatter(x_list, y_list, color="blue")
plt.plot(xa, ya, color="red")
plt.scatter(x_c, y_c, color="black", marker="o")
plt.scatter(x_c, y_c, color="yellow", marker="X")
plt.plot(0, 0, color="black")
plt.show()
