
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
stochastic_percentage = 0.6

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
