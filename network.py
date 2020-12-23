

from math import exp


class Network():

	'''
	A very basic neural network with two parameters: one weight and one bias.
	The network structure will be very bad at predicting, but it will be very easy for implementing backpropogation.
	The input will be a single number, and the output will be a single number.
	'''


	def __init__(self, w, b):
		# initialise the weight and bias
		self.weight = w
		self.bias = b


	def guess(self, x):
		# x is the input, return the output
		return self.weight * x + self.bias


	def cost(self, x, y):
		# return the cost related to an x input value and y expected output value
		# square it to force positive
		return (y - self.guess(x))**2


	def cost_derivatives(self, x, y):
		# find the derivative of the cost for the weight and the bias, and return both
		# dCost / dw = dCost / dGuess * dGuess / dw
		# dCost / dGuess = 2 * (guess - y)
		# dGuess / dw = x
		# dCost / db = dCost / dGuess * dGuess / db
		# dCost / dGuess = 2 * (guess - y)
		# dGuess / dw = 1
		dC_dg = 2*(y - self.guess(x)) # cost / guess
		dg_dw = x # guess / weight
		dg_db = 1 # guess / bias
		dC_dw = dC_dg * dg_dw # cost / weight
		dC_db = dC_dg * dg_db # cost / bias
		return dC_dw, dC_db


