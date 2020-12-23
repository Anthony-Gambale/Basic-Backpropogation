

from network import Network
import numpy as np
from matplotlib import pyplot as plt
from time import time

# read the points from the csv
data = np.genfromtxt('data.csv', delimiter=',')
for point in data:
    point[1] += 500

# init the network
eta = 0.0003
testnet = Network(0, 0, eta)

# initial time
t_initial = time()

# train the network
testnet.train(100000, data, True, 0.1) # I have found that 0.1 is the sweet spot. lower percentages give worse estimates with similar time, and higher percentages give similar estimates with worse time.

# final time
t_final = time()

# tell the user how long it took
print("Training took " + str( round( (t_final-t_initial)/60, 3) )+" minutes.")

# display the weight and bias after training
print()
print("y = "+str(testnet.weight)+" x + "+str(testnet.bias))

# make an x and y axis for the points
x_list = [x for x,y in data]
y_list = [y for x,y in data]

# make an x and y axis for the line
xa = np.linspace(0, len(data), 100)
ya = xa * testnet.weight + testnet.bias

# find the farthese in the x and y direction that the line goes and make sure the whole plot stretches to fit
f = max(xa[-1], ya[-1])
plt.plot(f, f)

# make sure to display the 0,0 origin
plt.plot(0, 0)

# plot it all
plt.scatter(x_list, y_list, color="blue")
plt.plot(xa, ya, color="red")
plt.show()
