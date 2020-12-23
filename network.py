

from math import exp
import numpy as np
from matplotlib import pyplot as plt


class Network():

	'''
	A very basic neural network with two parameters: one weight and one bias.
	The network structure will be very bad at predicting, but it will be very easy for implementing backpropogation.
	The input will be a single number, and the output will be a single number.
	'''

	def __init__(self, w, b, e):
		# initialise the weight and bias
		self.weight = w
		self.bias = b
		self.eta = e


	def guess(self, x):
		''' x is the input, return the output '''
		return self.weight * x + self.bias


	def cost(self, x, y):
		''' return the cost related to an x input value and y expected output value
		square it to force positive '''
		return (y - self.guess(x))**4


	def cost_derivatives(self, x, y):
		''' find the derivative of the cost for the weight and the bias, and return both
		dCost / dw = dCost / dGuess * dGuess / dw
		dCost / dGuess = 2 * (guess - y)
		dGuess / dw = x
		dCost / db = dCost / dGuess * dGuess / db
		dCost / dGuess = 2 * (guess - y)
		dGuess / dw = 1'''
		dC_dg = 2*(self.guess(x) - y) # cost / guess
		dg_dw = x # guess / weight
		dg_db = 1 # guess / bias
		dC_dw = dC_dg * dg_dw # cost / weight
		dC_db = dC_dg * dg_db # cost / bias
		return dC_dw, dC_db


	def backprop_step(self, data):
		''' increment my gradient based on all the x and y values in the data
		data looks like [[x,y], [x,y], [x,y], [x,y] ... ] '''
		N = len(data)
		average_dC_dw = 0
		average_dC_db = 0

		for pair in data:
			[x, y] = pair
			dC_dw, dC_db = self.cost_derivatives(x, y)
			average_dC_dw -= dC_dw / N
			average_dC_db -= dC_db / N

		self.weight += self.eta * average_dC_dw
		self.bias += self.eta * average_dC_db


	def train(self, iterations, data):
		
		# pick a random x and y value to use for calculating a rough estimate of error
		x_list = [a[0] for a in data]
		y_list = [a[1] for a in data]
		
		x_c = (max(x_list) + min(x_list)) / 2
		y_c = (max(y_list) + min(y_list)) / 2
		print(x_c, y_c)
		
		# first, do a minimum number of iterations equal to 'iterations'
		for i in range(iterations):
			self.backprop_step(data)
			print(self.cost(x_c, y_c))
		# now, keep iterating till the error (of the mean point) is below 1000
		while self.cost(x_c, y_c) > 1000:
			self.backprop_step(data)
			print(self.cost(x_c, y_c))
