import numpy as np
import random
import math

class NeuralNet(object):

	def __init__(self, inputLayers, outputLayers, hiddenLayers):
		self.inputLayerSize = inputLayers+1
		self.outputLayerSize = outputLayers
		self.hiddenLayerSize = hiddenLayers+1
		self.performance = 0

		self.W1 = np.zeros((self.inputLayerSize,self.hiddenLayerSize))
		self.W2 = np.zeros((self.hiddenLayerSize,self.outputLayerSize))

		for i in range(len(self.W1)):
			self.W1[i] = random.gauss(0.0,1.0)

		for i in range(len(self.W2)):
			self.W2[i] = random.gauss(0.0,1.0)

	def forward(self, X):

		# Bias input
		X.append(1)

		# Input weighting 
		self.z2 = np.dot(X, self.W1)

		# Sigmoid activation function
		self.a2 = self.sigmoid(self.z2)

		# Bias on hidden layer
		np.append(self.a2,1)

		# Output weighting 
		yHat = np.dot(self.a2, self.W2)

		return yHat

	def sigmoid(self, z):
		return 1/(1 + np.exp(-z))-0.5

	def perturb_weights(self, mutation_std):

		for w in self.W1:
			w = random.gauss(w,mutation_std)

		for w in self.W2:
			w = random.gauss(w,mutation_std)

	def store_weights(self, filename):
		np.savetxt(filename+"W1", self.W1, delimiter=',')
		np.savetxt(filename+"W2", self.W2, delimiter=',')
		print self.W1

	def load_weights(self, filename):
		self.W1 = np.loadtxt(filename+"W1", delimiter=',')
		self.W2 = np.loadtxt(filename+"W2", delimiter=',')




"""IMPORTANT! Scale your inputs to 0-1"""
