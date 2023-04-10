import numpy as np

class nn:
    def __init__(self, w1, w2):
        # Define input layer size
        self.inputSize = 3

        # Define hidden layer size
        self.hiddenSize = 2

        # Define output layer size
        self.outputSize = 1

        self.w1 = w1
        self.w2 = w2


    def tanh(self, x):
        return np.tanh(x)

    def __call__(self, input):
        self.input = np.array(input).reshape(1, self.inputSize)
        return self.feedForward()

    def feedForward(self):
        # Hidden layer
        z1 = self.input.dot(self.w1)
        a1 = self.tanh(z1)

        # Output layer
        z2 = a1.dot(self.w2)
        a2 = self.tanh(z2)

        return a2


