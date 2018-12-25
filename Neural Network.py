import random
import math

class Node:
    def __init__(self):
        self.value = 0
        self.iconnections = []
        self.oconnections = []
        self.err = 0
        self.dir = 0
        self.error = 0

    def feedForwardHidden(self):
        value = 0
        for connection in self.iconnections:
            value += (connection.fromNode.value / 255) * connection.weight  #summation of a(k) * W(k, j)
##        if value > 8.0:
##            self.value = 1.0
##        elif value < 8.0:
##            self.value = 0.0
##        else:
##        if epoch == 2 and counter > 5000:
##            print((connection.fromNode.value / 255) * connection.weight)
        self.value = 1 / (1 + math.exp(-value))     #sigmoid func
        self.dir = self.value * (1 - self.value)    #derivative of sigmoid func ( for back prop, g'(in(j)) )
    
    def feedForwardOutput(self):
        value = 0
        for connection in self.iconnections:
            value += (connection.fromNode.value) * connection.weight   #summation of a(j) * W(j, i)
##        if value > 8.0:
##            self.value = 1.0
##        elif value < 8.0:
##            self.value = 0.0
##        else:
        self.value = 1 / (1 + math.exp(-value))     #sigmoid func
        self.dir = self.value * (1 - self.value)  #derivative of sigmoid func ( for back prop, g'(in(i)) )
    
    def backPropogateOutput(self, correct):
        self.err = correct - self.value        # Err(i)
        for connection in self.iconnections:   
            value = connection.fromNode.value  #a(j)
            self.error = self.err * self.dir   #Delta(i) = Err(i) * g'(in(i))
            connection.weight = connection.weight + 0.1 * value * self.error    #W(j, i) = W(j, i) + alpha * a(j) * Delta(i)
            
    def backPropogateHidden(self):
        self.err = 0
        for connection in self.oconnections:
            self.err += connection.weight * connection.to.error  #summation of Delta(i) * W(j, i)
        for connection in self.iconnections:      
            value = connection.fromNode.value    #a(k)
            self.error = self.err * self.dir   #Delta(j) = g'(in(j)) * Delta Sum
            connection.weight = connection.weight + 0.1 * value * self.error    #W(k, j) = W(j, i) + alpha * a(k) * Delta(j)
        
#error = self.err * (math.exp(-value)/ math.pow((1+math.exp(-value)), 2))

class Connection:
    def __init__(self):
        self.fromNode = None
        self.to = None
        self.weight = 0

inputs = []
hidden = []
outputs = []

inputsize = 784
hiddensize = 50
outputsize = 5

inputdata = []
inputlabels = []

testdata = ""
testlabels = []

counter = 0

def getLabel():
    with open(r'''train_labels.txt''', 'r') as file:
        lines = file.readlines()
        for line in lines:
            inputlabels.append([int(line[0]), int(line[2]), int(line[4]), int(line[6]), int(line[8])])
            
data = [0] * 784
def getInput(file):
    line = file.read(784)
    for i in range(inputsize):
        inputs[i].value = line[i]
        data[i] = line[i]

def getLabelT():
    with open(r'''test_labels.txt''', 'r') as file:
        lines = file.readlines()
        for line in lines:
            testlabels.append([int(line[0]), int(line[2]), int(line[4]), int(line[6]), int(line[8])])
            
def setup():
    for i in range(inputsize):
        inputs.append(Node())
    for i in range(hiddensize):
        hidden.append(Node())
    for i in range(outputsize):
        outputs.append(Node())

    for node in hidden:
        bias = Node()
        bias.value = 1
        bconnection = Connection()
        bconnection.weight = -1
        bconnection.fromNode = bias
        bconnection.to = node
        node.iconnections.append(bconnection)
        bias.oconnections.append(bconnection)
        for input in inputs:
            connection = Connection()
            connection.weight = random.uniform(-1, 1)
            connection.fromNode = input
            connection.to = node

            node.iconnections.append(connection)
            input.oconnections.append(connection)

    for output in outputs:
        bias = Node()
        bias.value = 1
        bconnection = Connection()
        bconnection.weight = -1
        bconnection.fromNode = bias
        bconnection.to = output
        output.iconnections.append(bconnection)
        bias.oconnections.append(bconnection)
        for node in hidden:
            connection = Connection()
            connection.weight = random.uniform(-1, 1)
            connection.fromNode = node
            connection.to = output

            output.iconnections.append(connection)
            node.oconnections.append(connection)

setup() 
getLabel()

#for label in inputlabels:
#    print(label)
##
epoch = 1
lastSquareError = 1
squareDiff = 1
while abs(squareDiff) > 0.0005:
    squarederror = 0
    counter = 0
    file = open(r'''train_images.raw''', 'rb')
    while counter < 28038:
        getInput(file)
        for node in hidden:
            node.feedForwardHidden()
        for node in outputs:
            node.feedForwardOutput()
        label = inputlabels[counter]
        
        serror = 0
        for i in range(5):
            serror += math.pow(label[i] - outputs[i].value, 2)
            outputs[i].backPropogateOutput(label[i])
        for node in hidden:
            node.backPropogateHidden()
        if counter % 5000 == 0:
            print(counter)

        squarederror += serror / 2
        counter += 1
    file.close()
    meanSE = squarederror/28038
    print(meanSE)
    squareDiff = lastSquareError - meanSE
    lastSquareError = meanSE
    epoch += 1

counter = 0

getLabelT()
file = open(r'''test_images.raw''', 'rb')

correct = 0
confusion = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
while counter < 2561:
    getInput(file)
    for node in hidden:
        node.feedForwardHidden()
    for node in outputs:
        node.feedForwardOutput()
        
    label = testlabels[counter]
    maxout = 0
    for i in range(5):
        if outputs[i].value > outputs[maxout].value:
            maxout = i
    correctindex = label.index(max(label))
    if correctindex == maxout:
        correct += 1
    confusion[correctindex][maxout] += 1
    if counter % 500 == 0:
        print(counter)
    counter += 1

print(epoch)
print(str(correct) + " " + str(correct/2561))
print(confusion)
