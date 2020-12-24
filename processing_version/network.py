
'''
This is very similar to the original network.py, but with less detail.
'''

def choose(l):
    '''return a randomly chosen element from the array l'''
    index = int(random(0, int(len(l))))
    return l[index]
    

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
        
        N = len(points) # number of datapoints
        M = int(ceil(N * stochastic_percentage)) # number of stochastic datapoints
        
        M_points = [] # add M points into an array
        for i in range(int(M)): M_points.append(choose(points))
        
        average_dC_dw = 0
        average_dC_db = 0
        
        for x,y in M_points:
            dC_dw, dC_db = self.cost_derivatives(x,y)
            average_dC_dw += dC_dw / M
            average_dC_db += dC_db / M
        
        self.w -= self.eta * average_dC_dw
        self.b -= self.eta * average_dC_db
    
    
    def display(self):
        '''draw a line with my weight and bias in red'''
        x1 = -300
        x2 = 300
        y1 = self.guess(x1)
        y2 = self.guess(x2)
        stroke(0, 255, 255)
        strokeWeight(2)
        line(x1, y1, x2, y2)
        
