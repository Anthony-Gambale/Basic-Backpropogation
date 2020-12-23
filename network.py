

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


	def guess(x):
		# x is the input, return the output
		return x * self.weight + self.bias