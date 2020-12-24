
'''

The file extension .pyde is for the program 'Processing,' a text editor for code that also supplies libraries
for drawing shapes and using colours and images.
To open and run this file, you need both Processing and the python extension (as processing is typically for Java).
After downloading processing, python mode can be installed from within the program.

https://processing.org/

'''

# ====================================================================================================================
# NEURAL NETWORK CLASS #
# ====================================================================================================================


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
        return (self.w * x) + self.b
    
    
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
        
        self.w -= self.eta * average_dC_dw * float(mouseY)/float(height)
        self.b -= self.eta * average_dC_db * 10000 * float(mouseY)/float(height) # larger learning rate works better for constant
        #print(self.w, self.b)
    
    
    def display(self):
        '''draw a line with my weight and bias in red'''
        x1 = -300
        x2 = 300
        y1 = self.guess(x1)
        y2 = self.guess(x2)
        r = 255 * float(mouseY)/float(height)
        g = 255-r
        stroke(r, g, 127)
        strokeWeight(2)
        line(x1, y1, x2, y2)
    
    
# ====================================================================================================================
# SETUP AND DRAW FUNCTION #
# ====================================================================================================================

# make the data array
data = []
for x in range(10, 200, 5):
    y = x * random(0, 0.4) + 200
    data.append([y, x])
    
# network
global eta
eta = 0.00001
global n
n = Network(eta)

# percentage for SGD
global stochastic_percentage
stochastic_percentage = 1.1

# number of iterations
global font_size
font_size = 13

# display text
global text_display
text_display = True


def display():
    '''display the points and the axes, before drawing the line'''
    # draw axes
    stroke(255)
    strokeWeight(1)
    line(-300, 0, 300, 0)
    line(0, -300, 0, 300)
    # draw points
    for x, y in data:
        circle(x, y, 3)


def setup():
    size(600, 600)


def draw():
    background(0)
    translate(width//2, height//2)
    display()
    n.backprop_step(data, stochastic_percentage)
    n.display()
    if text_display:
        textSize(font_size)
        text("Mouse click the screen to place a datapoint.", 0-300, font_size+2-300)
        text("Place your mouse pointer closer to the:", 0-300, 2*(2+font_size)-300)
        text(" - top of the screen to decrease learning rate", 0-300, 3*(2+font_size)-300)
        text(" - bottom of the screen to increase learning rate", 0-300, 4*(2+font_size)-300)
        text("Press any key to display/remove this text.", 0-300, 5*(2+font_size)-300)


def mousePressed():
    data.append([mouseX-300, mouseY-300])


def keyPressed():
    global text_display
    text_display = not text_display
