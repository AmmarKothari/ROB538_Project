import numpy as np
import random

class NeuralNet(object):

	def __init__(self, inputLayers, outputLayers, hiddenLayers):
		self.inputLayerSize = inputLayers
		self.outputLayerSize = outputLayers
		self.hiddenLayerSize = hiddenLayers
		self.performance = 0

		self.W1 = np.zeros((self.inputLayerSize,self.hiddenLayerSize))
		self.W2 = np.zeros((self.hiddenLayerSize,self.outputLayerSize))

		for w in self.W1:
			w = random.gauss(0.0,1.0)

		for w in self.W2:
			w = random.gauss(0.0,1.0)

	def forward(self, X):
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.sigmoid(self.z3)
		return yHat

	def sigmoid(self, z):
		return 1/(1 + np.exp(-z))

	def perturb_weights(self, mutation_std):

		for w in self.W1:
			w = random.gauss(w,mutation_std)

		for w in self.W2:
			w = random.gauss(w,mutation_std)

"""IMPORTANT! Scale your inputs to 0-1"""
