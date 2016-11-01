import numpy as np

class NeuralNet(object):
    def __init__(self, inputLayers, outputLayers, hiddenLayers):
        self.inputLayerSize = inputLayers
        self.outputLayerSize = outputLayers
        self.hiddenLayerSize = hiddenLayers
        self.performance = 0

        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
         self.z2 = np.dot(X, self.W1)
         self.a2 = self.sigmoid(self.z2)
         self.z3 = np.dot(self.a2, self.W2)
         yHat = self.sigmoid(self.z3)
         return yHat

    def sigmoid(self, z):
         return 1/(1 + np.exp(-z))

    def perturb_weights(self, perturbance):
        pW1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize) * perturbance
        self.W1 += pW1
        pW2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize) * perturbance
        self.W2 += pW2



"""IMPORTANT! Scale your inputs to 0-1"""
