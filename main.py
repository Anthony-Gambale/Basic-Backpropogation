

"""
Basic-Backpropogation

Anthony Gambale December 2020

Uses gradient descent methods to perform backpropogation on the simplest, 2-parameter neural network (i.e. a linear function).
"""

from network import Network
import numpy as np
from matplotlib import pyplot as plt
from time import time

# read the points from the csv
data = np.genfromtxt('data.csv', delimiter=',')
for point in data:
    point[1] *= 1.5

# init the network
eta = 0.0003
testnet = Network(0, 0, eta)

# initial time
t_initial = time()

# train the network
testnet.train(10000, data, True, 0.1) # I have found that 0.1 is the sweet spot for SGD. lower percentages give worse estimates with similar time, and higher percentages give similar estimates with worse time.

# final time
t_final = time()

# tell the user how long it took
print("Training took " + str( round( (t_final-t_initial)/60, 3) )+" minutes.")

# display the weight and bias after training
print()
print("y = "+str(round(testnet.weight,3))+" x + "+str(round(testnet.bias,3)))

# make an x and y axis for the points
x_list = [x for x,y in data]
y_list = [y for x,y in data]
x_list_abs = [abs(x) for x,y in data]
y_list_abs = [abs(y) for x,y in data]

# find the farthest in the x and y direction that the line goes and make sure the whole plot stretches to fit
f_x = int(max(x_list_abs))*1.15
#f_y = testnet.weight * f_x + testnet.bias
f_y = int(max(y_list_abs))*1.15
f = max(f_x, f_y)

# now, draw the line itself, making sure it stays within the [-f, f] [-f, f] boundaries.
if testnet.weight > 0:
    # if the gradient is positive, make the line touch the bottom edge
    smallest_x = max(-f, (f+testnet.bias)/(-testnet.weight))
else:
    # if the gradient is negative, make the line touch the top edge
    smallest_x = max(-f, (f-testnet.bias)/(testnet.weight))

# make an x and y axis for the line
xa = np.linspace(smallest_x, f_x, 100)
ya = xa * testnet.weight + testnet.bias

# generate the lists
incrementing = np.linspace(0, f)
constant = incrementing * 0
negative_incrementing = [-k for k in incrementing]

# positive axis
plt.plot(constant, incrementing, color="black", linestyle="--")
plt.plot(incrementing, constant, color="black", linestyle="--")
# negative axis
plt.plot(constant, negative_incrementing, color="black", linestyle="--")
plt.plot(negative_incrementing, constant, color="black", linestyle="--")

# make sure to display the 0,0 origin
plt.plot(0, 0)

# plot it all
plt.scatter(x_list, y_list, color="blue")
plt.plot(xa, ya, color="red")
plt.show()
