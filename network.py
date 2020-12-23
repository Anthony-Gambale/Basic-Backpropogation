

from math import exp, ceil
import numpy as np
from matplotlib import pyplot as plt
from random import choice


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


    def average_cost(self, data):
        ''' return the average of the costs of each datapoint '''
        average = 0
        number = len(data)
        for x, y in data:
            average += self.cost(x, y) / number
        return average


    def cost_derivatives(self, x, y):
        '''
        find the derivative of the cost for the weight and the bias, and return both
        dCost / dw = dCost / dGuess * dGuess / dw
        dCost / dGuess = 2 * (guess - y)
        dGuess / dw = x
        dCost / db = dCost / dGuess * dGuess / db
        dCost / dGuess = 2 * (guess - y)
        dGuess / dw = 1
        '''
        dC_dg = 2*(self.guess(x) - y) # cost / guess
        dg_dw = x # guess / weight
        dg_db = 1 # guess / bias
        dC_dw = dC_dg * dg_dw # cost / weight
        dC_db = dC_dg * dg_db # cost / bias
        return dC_dw, dC_db


    def backprop_step(self, points):
        '''
        increment my gradients based on all the x and y values in the data
        data looks like [[x,y], [x,y], [x,y], [x,y] ... ]
        '''
        N = len(points)
        average_dC_dw = 0
        average_dC_db = 0
        
        for x, y in points:
            dC_dw, dC_db = self.cost_derivatives(x, y)
            average_dC_dw -= dC_dw / N
            average_dC_db -= dC_db / N

        self.weight += self.eta * average_dC_dw
        self.bias += self.eta * average_dC_db


    def backprop_step_stochastic(self, data, stochastic_percentage):
        '''
        increment the gradients based on all the x and y values in the data
        data looks like [[x,y], [x,y], [x,y], ... ]
        but not all the points, only take a percentage of them - 'stochastic percentage'
        '''
        M = ceil(len(data) * stochastic_percentage)
        M_points = []
        for i in range(M):
            M_points.append(choice(data))
        
        self.backprop_step(M_points)


    def train(self, iterations, data, stochastic, stochastic_percentage):
        '''
        train the network on a whole dataset
        '''
        # non stochastic - detailed
        if not stochastic:

            for i in range(iterations):
                self.backprop_step(data)
                print(i)

        # stochastic - messy and fast
        else:

            for i in range(iterations):
                self.backprop_step_stochastic(data, stochastic_percentage)
                print(i)
