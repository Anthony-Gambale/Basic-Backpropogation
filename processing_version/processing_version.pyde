
'''

The file extension .pyde is for the program 'Processing,' a text editor for code that also supplies libraries
for drawing shapes and using colours and images.
To open and run this file, you need both Processing and the python extension (as processing is typically for Java).
After downloading processing, python mode can be installed from within the program.

https://processing.org/

'''

from network import Network

# make the data array
data = []
for x in range(10, 200, 10):
    y = x * 3 + random(-30, 30)
    data.append([x, y])
# make the network
n = Network(0.0000001)
# make the percentage
stochastic_percentage = 1


def display():
    '''display the points and the axes, before drawing the line'''
    # draw axes
    stroke(255)
    strokeWeight(1)
    line(-300, 0, 300, 0)
    line(0, -300, 0, 300)
    # draw points
    for x, y in data:
        circle(x, y, 2)

def setup():
    size(600, 600)
    
def draw():
    background(0)
    translate(width//2, height//2)
    display()
    n.backprop_step(data, stochastic_percentage)
    n.display()
