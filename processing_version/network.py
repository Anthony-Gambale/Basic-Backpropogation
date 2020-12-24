
'''
This is very similar to the original network.py, but with less detail.
'''

class Network:
    
    
    def __init__(self, e):
        '''initialise variables'''
        self.w = 0 # the two parameters
        self.b = 0
        self.eta = e # learning rate
    
    
    def guess(self, x):
        '''x is used for inputs, and y is used for expected outputs. inputs and outputs to this network
        are both single numbers.'''
        return self.w * x + self.b
    
    
    # cost() and average_cost() functions are not needed
    # see the original version with both of these functions
    
    
    def cost_derivatives(self, x, y):
        '''based on the specific given input and output, x and y, calculate the derivatives for both
        of the parameters.
        
        Use the chain rule:
        dC/dw = dC/dguess * dguess/dw
        dC/db = dC/dguess * dguess/db
        
        Functions:
        C = (guess - y)^2
        guess = wx + b
        
        Derivatives:
        dC/dguess = 2(guess - y)
        dguess/dw = x
        dguess/db = 1
        '''
        dC_dguess = 2 * (self.guess(x) - y)
        dguess_dw = x
        dguess_db = 1
        dC_dw = dC_dguess * dguess_dw
        dC_db = dC_dguess * dguess_db
        return dC_dw, dC_db
    
    
    def backprop_step(self, points, stochastic_percentage):
        '''the 'points' array looks like [[x,y], [x,y], [x,y], [x,y], [x,y] ... ]
        go through each of these, find the derivatives dC/dw and dC/db for each of these
        points, and average them'''
        
        N = 
        
        
