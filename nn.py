import numpy as np

class nn:
    def __init__(self, weights):
        # Define input layer size
        self.inputSize = 3

        # Define hidden layer size
        self.hiddenSize = 24
        self.numberHiddenLayers = 3

        # Define output layer size
        self.outputSize = 1
        
        self.weights = weights


    def tanh(self, x):
        return 2 * np.tanh(x)

    def __call__(self, input):
        self.input = np.array(input).reshape(1, self.inputSize)
        return self.feedForward()

    def feedForward(self):
        # 1st Hidden layer
        z = self.input.dot(self.weights[0])
        a = self.tanh(z)

        # Hidden layers
        for i in range(1, self.numberHiddenLayers - 1):
            z = a.dot(self.weights[i])
            a = self.tanh(z)

        # Output layer
        z = a.dot(self.weights[-1])
        y = self.tanh(z)

        return y


