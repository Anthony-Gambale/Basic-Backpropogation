

from network import *

data = [[1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,8]]

testnet = Network(0, 0, 0.00001)

testnet.train(1000000, data)

print("y = "+str(testnet.weight)+" x + "+str(testnet.bias))
